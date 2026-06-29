"""Frozen MindEye2 ``unclip6`` SDXL-unCLIP decoder with structured conditioning.

The decoder exposes two conditioning slots (paper Stage 5):

  * ``crossattn`` -- the reassembled raw patch tokens X in R^{256 x 1664};
  * ``vector``    -- the pooled image-embedding slot, into which we route the
    predicted global direction ``c_cls`` (concatenated with the size/crop
    micro-conditioning).

Routing ``c_cls`` into the pooled slot -- instead of the content-free
``vector_suffix`` obtained from a dummy image -- is the paper's "Change 5": it
supplies the global semantic conditioning the unCLIP model was trained on, which
the patch tokens alone do not provide.

The exact byte layout of ``vector`` is defined by the vendored sgm conditioner
(available only on the cluster). We therefore expose ``cls_vector_slot`` so the
1280-d pooled embedding can be placed correctly; the default ``(0, 1280)`` writes
it to the leading positions, which is the SDXL-unCLIP convention (pooled image
embedding first, then the size/crop Fourier features).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F


class SDXLUnCLIPDecoder:
    def __init__(self, device, mindeye_src, ckpt_path, num_steps=38, out_size=256,
                 cls_vector_slot=(0, 1280)):
        self.device = device
        self.num_steps = num_steps
        self.out_size = out_size
        self.cls_slot = tuple(cls_vector_slot)
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

        p["first_stage_config"]["target"] = "sgm.models.autoencoder.AutoencoderKL"
        p["sampler_config"]["params"]["num_steps"] = num_steps
        self.offset_noise_level = p["loss_fn_config"]["params"]["offset_noise_level"]

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

        batch = {
            "jpg": torch.randn(1, 3, 1, 1, device=device),
            "original_size_as_tuple": torch.ones(1, 2, device=device) * 768,
            "crop_coords_top_left": torch.zeros(1, 2, device=device),
        }
        out = engine.conditioner(batch)
        self.vector_suffix = out["vector"].to(device)
        V = self.vector_suffix.shape[-1]
        lo, hi = self.cls_slot
        # MindEye2's unclip6 carries ALL image semantics in the 256x1664 crossattn
        # tokens; its `vector` is size/crop micro-conditioning ONLY (no pooled-image
        # slot). Only inject c_cls if the requested slot actually fits the vector.
        self.cls_in_vector = self.cls_slot is not None and hi <= V
        if self.cls_in_vector:
            print(f"✓ SDXL-unCLIP ready. vector {tuple(self.vector_suffix.shape)} "
                  f"| c_cls -> vector[{lo}:{hi}]")
        else:
            print(f"✓ SDXL-unCLIP ready. vector width {V}: NO pooled-CLS slot "
                  f"(micro-conditioning only). Image semantics flow through the "
                  f"256x1664 crossattn tokens; c_cls is NOT routed to the decoder.")

    def _vector(self, cls_emb_row):
        """Build the ``vector`` slot; inject c_cls only if this decoder has a slot."""
        vec = self.vector_suffix.clone()
        if cls_emb_row is not None and self.cls_in_vector:
            lo, hi = self.cls_slot
            vec[:, lo:hi] = cls_emb_row.to(vec.device, vec.dtype).reshape(1, hi - lo)
        return vec

    @torch.no_grad()
    def _recon_one(self, x, vector, num_samples=1):
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
                 "vector": vector.repeat(num_samples, 1)}
            rand_tokens = torch.randn_like(x)
            uc = {"crossattn": rand_tokens.repeat(num_samples, 1, 1),
                  "vector": vector.repeat(num_samples, 1)}
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
        return samples

    @torch.no_grad()
    def decode(self, tokens_raw: torch.Tensor, cls_emb: torch.Tensor | None = None) -> torch.Tensor:
        """Decode raw patch tokens, optionally with predicted CLS in the pooled slot.

        ``tokens_raw``: (B, 256, 1664) reassembled raw tokens.
        ``cls_emb``:    (B, 1280) predicted global directions, or None to fall
                        back to the (content-free) dummy suffix.
        """
        imgs = []
        for i in range(tokens_raw.shape[0]):
            vec = self._vector(None if cls_emb is None else cls_emb[i:i + 1])
            s = self._recon_one(tokens_raw[i:i + 1].to(self.device), vec, num_samples=1)
            if self.out_size and self.out_size != s.shape[-1]:
                s = F.interpolate(s, self.out_size, mode="bilinear", align_corners=False)
            imgs.append(s.float().cpu())
        return torch.cat(imgs)
