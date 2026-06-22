"""SDXL-unCLIP decoder (Step 1b) — MindEye2's frozen `sgm` DiffusionEngine.

Reimplements MindEye2's `unclip_recon` verbatim (see src/utils.py).  Takes a RAW
bigG token grid (B, 256, 1664) as the `crossattn` condition and renders 768x768
images, optionally resized to 256.

Requires the vendored sgm + config from the MindEyeV2 repo and the checkpoint
`unclip6_epoch0_step110000.ckpt` (see scripts/setup_step1b.sh).

Verified details baked in:
  * latent (B,4,96,96), scale_factor 0.13025 (from config).
  * sampler EulerEDMSampler, num_steps=38 (override), VanillaCFG scale=5.0.
  * unconditional branch uses RANDOM tokens (randn_like), same vector_suffix.
  * vector_suffix = conditioner(dummy jpg batch)["vector"]  (size/crop embedding).
  * first_stage target overridden to sgm.models.autoencoder.AutoencoderKL.
  * final image = clamp(samples_x*0.8 + 0.2, 0, 1).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F


class SDXLUnCLIPDecoder:
    def __init__(self, device, mindeye_src, ckpt_path, num_steps=38, out_size=256):
        self.device = device
        self.num_steps = num_steps
        self.out_size = out_size
        mindeye_src = Path(mindeye_src)
        gm = mindeye_src / "generative_models"
        for p in (str(mindeye_src), str(gm)):
            if p not in sys.path:
                sys.path.insert(0, p)

        from omegaconf import OmegaConf
        from generative_models.sgm.models.diffusion import DiffusionEngine
        from generative_models.sgm.util import append_dims
        self._append_dims = append_dims

        cfg_path = gm / "configs" / "unclip6.yaml"
        config = OmegaConf.to_container(OmegaConf.load(str(cfg_path)), resolve=True)
        p = config["model"]["params"]
        # Overrides applied by MindEye2 at inference time.
        p["first_stage_config"]["target"] = "sgm.models.autoencoder.AutoencoderKL"
        p["sampler_config"]["params"]["num_steps"] = num_steps
        self.offset_noise_level = p["loss_fn_config"]["params"]["offset_noise_level"]
        # The VAE decoder's attn_type 'vanilla-xformers' hard-crashes without
        # xformers (it doesn't fall back like the UNet does). Force native
        # attention — same weights, identical math, no xformers dependency.
        try:
            p["first_stage_config"]["params"]["ddconfig"]["attn_type"] = "vanilla"
        except (KeyError, TypeError):
            pass

        engine = DiffusionEngine(
            network_config=p["network_config"],
            denoiser_config=p["denoiser_config"],
            first_stage_config=p["first_stage_config"],
            conditioner_config=p["conditioner_config"],
            sampler_config=p["sampler_config"],
            scale_factor=p["scale_factor"],
            disable_first_stage_autocast=p["disable_first_stage_autocast"],
        )
        engine.eval().requires_grad_(False)
        engine.to(device)
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        engine.load_state_dict(ckpt["state_dict"])
        self.engine = engine

        # vector_suffix: size/crop conditioning (constant for all images).
        batch = {
            "jpg": torch.randn(1, 3, 1, 1, device=device),               # placeholder, unused
            "original_size_as_tuple": torch.ones(1, 2, device=device) * 768,
            "crop_coords_top_left": torch.zeros(1, 2, device=device),
        }
        out = engine.conditioner(batch)
        self.vector_suffix = out["vector"].to(device)
        print(f"✓ SDXL-unCLIP ready. vector_suffix {tuple(self.vector_suffix.shape)}")

    @torch.no_grad()
    def _recon_one(self, x, num_samples=1):
        """x: (1,256,1664) RAW bigG tokens. Verbatim port of MindEye2 unclip_recon."""
        eng = self.engine
        device = self.device
        append_dims = self._append_dims
        if x.ndim == 2:
            x = x[None]
        if x.shape[0] == 1:
            x = x[[0]]
        with torch.cuda.amp.autocast(dtype=torch.float16), eng.ema_scope():
            z = torch.randn(num_samples, 4, 96, 96, device=device)
            tokens = x
            c = {"crossattn": tokens.repeat(num_samples, 1, 1),
                 "vector": self.vector_suffix.repeat(num_samples, 1)}
            rand_tokens = torch.randn_like(x)
            uc = {"crossattn": rand_tokens.repeat(num_samples, 1, 1),
                  "vector": self.vector_suffix.repeat(num_samples, 1)}
            for k in c:
                c[k] = c[k][:num_samples].to(device)
                uc[k] = uc[k][:num_samples].to(device)

            noise = torch.randn_like(z)
            sigmas = eng.sampler.discretization(eng.sampler.num_steps)
            sigma = sigmas[0].to(z.device)
            if self.offset_noise_level > 0.0:
                noise = noise + self.offset_noise_level * append_dims(
                    torch.randn(z.shape[0], device=z.device), z.ndim)
            noised_z = z + noise * append_dims(sigma, z.ndim)
            noised_z = noised_z / torch.sqrt(1.0 + sigmas[0] ** 2.0)

            def denoiser(xx, ss, cc):
                return eng.denoiser(eng.model, xx, ss, cc)

            samples_z = eng.sampler(denoiser, noised_z, cond=c, uc=uc)
            samples_x = eng.decode_first_stage(samples_z)
            samples = torch.clamp(samples_x * 0.8 + 0.2, 0.0, 1.0)
        return samples            # (num_samples, 3, 768, 768) in [0,1]

    @torch.no_grad()
    def decode(self, tokens_raw: torch.Tensor) -> torch.Tensor:
        """tokens_raw: (B,256,1664) -> (B,3,out_size,out_size) in [0,1]."""
        imgs = []
        for i in range(tokens_raw.shape[0]):
            s = self._recon_one(tokens_raw[i:i + 1].to(self.device), num_samples=1)  # (1,3,768,768)
            if self.out_size and self.out_size != s.shape[-1]:
                s = F.interpolate(s, self.out_size, mode="bilinear", align_corners=False)
            imgs.append(s.float().cpu())
        return torch.cat(imgs)
