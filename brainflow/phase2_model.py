from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .models import BrainEncoder, migrate_input_proj
from .flow_clip_dit import FlowCLIPDiT
from .flow_unet import FlowUNet
from .ot_coupling import ot_minibatch_coupling

class BrainFlowPhase2(nn.Module):

    def __init__(self, cfg: Config, voxels: dict[int, int]):
        super().__init__()
        self.cfg = cfg
        self.brain_enc = BrainEncoder(cfg, voxels)
        self.flow_clip = FlowCLIPDiT(cfg)
        self.flow_vae = FlowUNet(cfg)

    @classmethod
    def from_v5_checkpoint(cls, cfg: Config, voxels: dict[int, int],
                           ckpt_path: str) -> "BrainFlowPhase2":
        model = cls(cfg, voxels)
        state = torch.load(ckpt_path, map_location="cpu")

        state = {k.replace("._orig_mod.", "."): v for k, v in state.items()}
        state = migrate_input_proj(state, model.brain_enc.max_vox)
        enc_state = {k[len("brain_enc."):]: v
                     for k, v in state.items() if k.startswith("brain_enc.")}
        missing, unexpected = model.brain_enc.load_state_dict(enc_state, strict=False)
        if missing:
            print(f"[Phase2] BrainEncoder missing keys: {missing}")
        if unexpected:
            print(f"[Phase2] BrainEncoder unexpected keys: {unexpected}")
        return model

    def set_stage_2a(self):
        for p in self.brain_enc.parameters():
            p.requires_grad_(False)
        for p in self.flow_clip.parameters():
            p.requires_grad_(True)
        for p in self.flow_vae.parameters():
            p.requires_grad_(False)

    def set_stage_2b(self):
        for p in self.parameters():
            p.requires_grad_(True)

    def encode_fmri(self, fmri: torch.Tensor,
                    subject: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.brain_enc(fmri, subject)

    def _to_clip_grid(self, clip_patches: torch.Tensor) -> torch.Tensor:
        B = clip_patches.shape[0]
        if self.flow_clip._standardization_fitted:
            clip_patches = self.flow_clip.standardize(clip_patches)
        return clip_patches.reshape(B, 16, 16, 1024).permute(0, 3, 1, 2).contiguous()

    def _from_clip_grid(self, clip_grid: torch.Tensor) -> torch.Tensor:
        B = clip_grid.shape[0]
        flat = clip_grid.permute(0, 2, 3, 1).reshape(B, 256, 1024)
        if self.flow_clip._standardization_fitted:
            flat = self.flow_clip.destandardize(flat)
        return flat

    def training_step(self, batch: dict, device: torch.device,
                      epoch: int = 0) -> dict[str, torch.Tensor]:
        stage = self.cfg.training_stage
        if stage == "2a":
            return self._step_2a(batch, device)
        if stage == "2b":
            return self._step_2b(batch, device, epoch)
        raise ValueError(f"Unknown training_stage: {stage!r}")

    def _step_2a(self, batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
        clip_patches = batch["clip_patches"].to(device).float()
        clip_cls = batch["clip_emb"].to(device).float()
        fmri = batch["fmri"].to(device)
        subject = batch["subject"].to(device)

        with torch.no_grad():
            tokens, _ = self.brain_enc(fmri, subject)

        clip_grid = self._to_clip_grid(clip_patches)
        loss_dict = self.flow_clip.flow_loss(clip_grid, tokens, clip_cls)
        return loss_dict

    def _step_2b(self, batch: dict, device: torch.device,
                 epoch: int) -> dict[str, torch.Tensor]:
        clip_patches = batch["clip_patches"].to(device).float()
        clip_cls = batch["clip_emb"].to(device).float()
        latents = batch["latent"].to(device)
        fmri = batch["fmri"].to(device)
        subject = batch["subject"].to(device)
        cfg = self.cfg
        B = clip_patches.shape[0]

        tokens, _ = self.brain_enc(fmri, subject)

        clip_grid = self._to_clip_grid(clip_patches)
        clip_loss_dict = self.flow_clip.flow_loss(clip_grid, tokens, clip_cls)

        ramp_epochs = max(1, cfg.clip_ramp_frac * cfg.num_epochs)
        clip_sample_prob = min(cfg.clip_sample_prob_max,
                               epoch / ramp_epochs * cfg.clip_sample_prob_max)

        sample_mask = torch.rand(B, device=device) < clip_sample_prob
        if sample_mask.any():
            with torch.no_grad():
                sampled = self.flow_clip.sample(tokens[sample_mask], n_steps=10, solver="euler")
            sampled_patches = self._from_clip_grid(sampled)
            clip_ctx = clip_patches.clone()
            clip_ctx[sample_mask] = sampled_patches
        else:
            clip_ctx = clip_patches

        t = torch.rand(B, device=device)
        x0 = torch.randn_like(latents)

        if cfg.use_ot_coupling:
            x0 = ot_minibatch_coupling(
                x0, latents,
                reg=cfg.ot_reg,
                n_iters=cfg.ot_iters,
            )

        t_exp = t[:, None, None, None]
        xt = (1 - t_exp) * x0 + t_exp * latents
        ut = latents - x0

        v_pred = self.flow_vae(xt, t, tokens, clip_ctx=clip_ctx)
        cfm_loss = F.mse_loss(v_pred, ut)

        total = (cfg.lambda_clip_flow * clip_loss_dict["loss"]
                 + cfg.lambda_cfm * cfm_loss)
        return {
            "loss": total,
            "cfm": cfm_loss,
            "clip_flow": clip_loss_dict["loss"],
            "loss_mse": clip_loss_dict["loss_mse"],
            "loss_cos": clip_loss_dict["loss_cos"],
        }

    @torch.no_grad()
    def sample(self, tokens: torch.Tensor, n_steps: int = 20,
               cfg_scale: float = 1.0, solver: str = "euler",
               cls=None) -> torch.Tensor:
        cfg = self.cfg
        B = tokens.shape[0]
        device = tokens.device

        clip_grid = self.flow_clip.sample(
            tokens, n_steps=max(1, n_steps // 2),
            cfg_scale=cfg_scale, solver=solver)
        clip_ctx = self._from_clip_grid(clip_grid)

        x = torch.randn(B, cfg.latent_ch, cfg.latent_res, cfg.latent_res,
                        device=device)
        dt = 1.0 / n_steps
        null_ctx = self.flow_vae.null_tokens.expand(B, -1, -1)

        for i in range(n_steps):
            t_cur = torch.full((B,), i * dt, device=device)
            if cfg_scale > 1.0:
                v_cond = self.flow_vae(x, t_cur, tokens, clip_ctx=clip_ctx)
                v_uncond = self.flow_vae(x, t_cur, null_ctx, clip_ctx=None)
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = self.flow_vae(x, t_cur, tokens, clip_ctx=clip_ctx)

            if solver == "euler":
                x = x + v * dt
            elif solver == "midpoint":
                t_mid = torch.full((B,), (i + 0.5) * dt, device=device)
                x_mid = x + v * (dt / 2)
                v_mid = self.flow_vae(x_mid, t_mid, tokens, clip_ctx=clip_ctx)
                x = x + v_mid * dt
            elif solver == "heun":
                t_next = torch.full((B,), (i + 1) * dt, device=device)
                x_euler = x + v * dt
                v2 = self.flow_vae(x_euler, t_next, tokens, clip_ctx=clip_ctx)
                x = x + (v + v2) * (dt / 2)
            else:
                raise ValueError(f"Unknown solver: {solver!r}")

        return x
