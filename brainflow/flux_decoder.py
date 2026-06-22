from __future__ import annotations

import gc
import logging
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flux_adapter import BrainIPAdapter
from .config import Config

log = logging.getLogger(__name__)

_FLUX_VAE_SCALE = 0.3611

def _flux_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    cc = torch.cuda.get_device_capability()

    if cc[0] >= 8:
        return torch.bfloat16
    return torch.float16

def _sample_flux_time(B: int, device: torch.device, m: float = 0.0,
                      s: float = 1.0) -> torch.Tensor:
    u = torch.randn(B, device=device) * s + m
    return torch.sigmoid(u)

class _NullTextCache:

    def __init__(self):
        self._prompt_embeds    = None
        self._pooled_embeds    = None
        self._text_ids         = None

    def get(self, pipe, device, dtype):
        if self._prompt_embeds is None:
            with torch.no_grad():
                (
                    self._prompt_embeds,
                    self._pooled_embeds,
                    self._text_ids,
                ) = pipe.encode_prompt(
                    prompt="",
                    prompt_2="",
                    device=device,
                    num_images_per_prompt=1,
                )
            self._prompt_embeds  = self._prompt_embeds.to(dtype)
            self._pooled_embeds  = self._pooled_embeds.to(dtype)
            self._text_ids       = self._text_ids.to(dtype)
        return (
            self._prompt_embeds.to(device),
            self._pooled_embeds.to(device),
            self._text_ids.to(device),
        )

class FLUXBrainDecoder(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.ip_adapter = BrainIPAdapter(
            brain_dim=cfg.brain_dim,
            clip_dim=cfg.clip_dim,
            ip_dim=cfg.ip_dim,
            n_blocks=cfg.ip_n_blocks,
            use_gradient_checkpointing=getattr(cfg, "flux_grad_ckpt", False),
        )

        self._pipe        = None
        self._null_cache  = _NullTextCache()
        self._hook_handles: list = []
        self._flux_dtype  = _flux_dtype()

        self._adapter_inputs: dict = {}

        self.register_buffer("_device_anchor", torch.zeros(1), persistent=False)

    def _load_pipe(self):
        if self._pipe is not None:
            return self._pipe
        try:
            from diffusers import FluxPipeline
        except ImportError as e:
            raise RuntimeError(
                "diffusers >= 0.30 with FLUX support required. "
                "Run: pip install diffusers>=0.30"
            ) from e

        cache_dir   = Path(self.cfg.data_dir) / "hf_cache"
        flux_dtype  = self._flux_dtype
        device      = self._device_anchor.device
        offload_t5  = getattr(self.cfg, "flux_offload_t5", True)

        n_gpu       = torch.cuda.device_count()
        device_map  = "balanced" if n_gpu > 1 else None

        log.info(
            "Loading FLUX.1-dev | dtype=%s | device_map=%s | T5_offload=%s",
            flux_dtype, device_map, offload_t5,
        )

        load_kwargs: dict = dict(
            torch_dtype=flux_dtype,
            cache_dir=cache_dir,
        )
        if device_map is not None:
            load_kwargs["device_map"] = device_map

        pipe = FluxPipeline.from_pretrained(self.cfg.flux_model_id, **load_kwargs)

        if device_map is None:
            pipe = pipe.to(device)

        if offload_t5 and hasattr(pipe, "text_encoder_2"):
            pipe.text_encoder_2.to("cpu")
            log.info("T5 encoder offloaded to CPU (flux_offload_t5=True).")

        pipe.transformer.eval().requires_grad_(False)
        pipe.vae.eval().requires_grad_(False)
        pipe.text_encoder.eval().requires_grad_(False)
        pipe.text_encoder_2.eval().requires_grad_(False)

        if hasattr(pipe.transformer, "enable_xformers_memory_efficient_attention"):
            try:
                pipe.transformer.enable_xformers_memory_efficient_attention()
                log.info("xformers memory-efficient attention enabled.")
            except Exception:
                pass

        self._pipe = pipe
        log.info(
            "FLUX.1-dev loaded. Transformer params: %dM (frozen) | dtype: %s",
            sum(p.numel() for p in pipe.transformer.parameters()) // 1_000_000,
            flux_dtype,
        )
        return self._pipe

    def _remove_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def _register_hooks(self, pipe):
        transformer = pipe.transformer

        def _make_hook(block_idx: int):
            def hook(module, args, output):
                if block_idx not in self._adapter_inputs:
                    return output

                cond_tokens, gate = self._adapter_inputs[block_idx]

                if isinstance(output, tuple):
                    img_out, *rest = output
                else:
                    img_out = output
                    rest    = []

                residual = self.ip_adapter.forward_block(
                    block_idx, img_out, cond_tokens, gate
                )
                img_out  = img_out + residual.to(img_out.dtype)

                return (img_out, *rest) if rest else img_out

            return hook

        blocks = getattr(transformer, "transformer_blocks", None)
        if blocks is None:
            raise AttributeError(
                "Cannot find transformer.transformer_blocks on FLUX pipeline. "
                "Check your diffusers version."
            )

        n = min(self.cfg.ip_n_blocks, len(blocks))
        for i in range(n):
            h = blocks[i].register_forward_hook(_make_hook(i))
            self._hook_handles.append(h)

    def _set_adapter_inputs(
        self,
        brain_tokens: torch.Tensor,
        mu_clip: torch.Tensor,
        log_sigma: torch.Tensor,
    ):
        cond_tokens, gate = self.ip_adapter.build_conditioning(
            brain_tokens, mu_clip, log_sigma
        )
        self._adapter_inputs = {
            i: (cond_tokens, gate)
            for i in range(self.cfg.ip_n_blocks)
        }

    @torch.no_grad()
    def _vae_encode(self, images: torch.Tensor) -> torch.Tensor:
        pipe  = self._load_pipe()
        dtype = self._flux_dtype

        x = images.to(dtype=dtype, device=pipe.vae.device) * 2.0 - 1.0
        latents = pipe.vae.encode(x).latent_dist.sample()
        return latents * _FLUX_VAE_SCALE

    def compute_loss(
        self,
        brain_tokens: torch.Tensor,
        mu_clip: torch.Tensor,
        log_sigma: torch.Tensor,
        target_images: torch.Tensor,
    ) -> torch.Tensor:
        pipe   = self._load_pipe()
        device = self._device_anchor.device
        dtype  = self._flux_dtype

        B = brain_tokens.shape[0]

        with torch.no_grad():
            z = self._vae_encode(target_images.to(device))

        t = _sample_flux_time(B, device)

        eps  = torch.randn_like(z)
        t4   = t.view(B, 1, 1, 1)
        z_t  = (1.0 - t4) * eps + t4 * z

        v_target = z - eps

        self._set_adapter_inputs(
            brain_tokens.to(device),
            mu_clip.to(device),
            log_sigma.to(device),
        )
        self._register_hooks(pipe)

        with torch.no_grad():
            prompt_embeds, pooled_embeds, text_ids = self._null_cache.get(
                pipe, device, dtype
            )

            prompt_embeds = prompt_embeds.expand(B, -1, -1)
            pooled_embeds = pooled_embeds.expand(B, -1)

        latent_image_ids = pipe._prepare_latent_image_ids(
            B, z_t.shape[2], z_t.shape[3], device, dtype
        )
        packed = pipe._pack_latents(z_t.to(dtype), *z_t.shape[2:])

        try:
            with torch.cuda.amp.autocast(dtype=dtype, enabled=torch.cuda.is_available()):
                noise_pred = pipe.transformer(
                    hidden_states=packed,
                    timestep=t.to(dtype),
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_embeds,
                    txt_ids=text_ids.expand(B, -1, -1),
                    img_ids=latent_image_ids.expand(B, -1, -1),
                    guidance=torch.full((B,), 1.0, device=device, dtype=dtype),
                    return_dict=False,
                )[0]
        finally:

            self._remove_hooks()
            self._adapter_inputs.clear()

        v_pred = pipe._unpack_latents(
            noise_pred, z_t.shape[2], z_t.shape[3], pipe.vae_scale_factor
        )

        loss = F.mse_loss(v_pred.float(), v_target.float())
        return loss

    @torch.no_grad()
    def generate(
        self,
        brain_tokens: torch.Tensor,
        mu_clip: torch.Tensor,
        log_sigma: torch.Tensor,
        height: int = 512,
        width:  int = 512,
        n_steps: Optional[int] = None,
        guidance: Optional[float] = None,
    ) -> torch.Tensor:
        pipe    = self._load_pipe()
        device  = self._device_anchor.device
        dtype   = torch.bfloat16
        B       = brain_tokens.shape[0]
        n_steps = n_steps or self.cfg.flux_steps
        guidance_val = guidance if guidance is not None else self.cfg.flux_guidance

        self._set_adapter_inputs(
            brain_tokens.to(device),
            mu_clip.to(device),
            log_sigma.to(device),
        )
        self._register_hooks(pipe)

        prompt_embeds, pooled_embeds, text_ids = self._null_cache.get(
            pipe, device, dtype
        )
        prompt_embeds = prompt_embeds.expand(B, -1, -1)
        pooled_embeds = pooled_embeds.expand(B, -1)

        try:
            result = pipe(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_embeds,
                height=height,
                width=width,
                num_inference_steps=n_steps,
                guidance_scale=guidance_val,
                num_images_per_prompt=1,
                output_type="pt",
            )
        finally:
            self._remove_hooks()
            self._adapter_inputs.clear()

        images = result.images.clamp(0.0, 1.0)
        return images

    def to(self, *args, **kwargs):
        out = super().to(*args, **kwargs)
        if self._pipe is not None:
            self._pipe = self._pipe.to(*args, **kwargs)
        return out

    def trainable_parameters(self):
        return self.ip_adapter.parameters()

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def save_adapter(self, path: str | Path):
        torch.save(self.ip_adapter.state_dict(), path)
        log.info("Adapter saved → %s", path)

    def load_adapter(self, path: str | Path, strict: bool = True):
        sd = torch.load(path, map_location="cpu")
        self.ip_adapter.load_state_dict(sd, strict=strict)
        log.info("Adapter loaded ← %s", path)
