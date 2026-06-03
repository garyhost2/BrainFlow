"""FLUXBrainDecoder — frozen FLUX.1-dev + BrainIPAdapter.

Architecture
------------
* FLUX.1-dev (12B, bf16) is loaded fully frozen.  No gradients, no weight
  updates.  Only the BrainIPAdapter (~120M fp32) is trained.

Adapter injection
-----------------
FLUX.1-dev uses MM-DiT "double-stream" blocks where image tokens and text
tokens evolve in parallel.  We inject the adapter into each double-stream
block by registering a forward hook on the block's image-attention output.
The hook fires **after** the block's own attention but **before** the MLP,
so the adapter cross-attends to brain tokens at every resolution of the
image stream.

Hook safety
-----------
* Hooks store state via a per-block side-channel (self._adapter_inputs)
  populated once before the FLUX forward pass from the encoded brain signals.
* Hooks are removed and re-registered every call to avoid accumulation.

Training loss
-------------
FLUX is itself a Rectified Flow model.  The training objective is the same
flow-matching velocity MSE used in Phase 1/2, now applied in FLUX's 16-ch
latent space (FLUX VAE, scale factor 0.3611).

    z  = flux_vae.encode(images)  ×  0.3611
    t  ~ logit_normal(0, 1)       (FLUX's own time sampling distribution)
    ε  ~ N(0, I)
    z_t = (1-t)*ε + t*z
    v_pred = flux_transformer(z_t, t, txt="", brain_conditioning)
    loss   = MSE(v_pred, z - ε)

FLUX.1-dev uses guidance distillation: we pass guidance=1.0 during training
(no CFG split needed) and guidance=3.5 at inference.

Inference
---------
Euler integration with FLUX's own scheduler, 28 steps default.  The adapter
conditioning is injected identically to training.  No text prompt needed —
we pass an empty string that FLUX's T5/CLIP encoders map to a fixed null
embedding.

Memory layout on 2× A100 80GB
------------------------------
  FLUX transformer (bf16) : ~23 GB
  FLUX VAE (bf16)          :  ~1 GB
  T5 encoder (bf16)        :  ~9 GB  (loaded once, kept resident)
  CLIP-L encoder (bf16)    :  ~0.5 GB
  BrainIPAdapter (fp32)    :  ~2 GB
  Activations  (bs=4)      :  ~8 GB
  ──────────────────────────────────
  Total                    : ~44 GB  → fits 2× A100 80GB or 1× A100 80GB
  Low-VRAM variant         : offload T5 after encoding, ip_n_blocks=10 → ~32 GB
"""
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


# ── FLUX VAE scale factor (matches FLUX.1-dev training) ────────────────────
_FLUX_VAE_SCALE = 0.3611


# ── logit-normal time sampling (matches FLUX training distribution) ─────────

def _sample_flux_time(B: int, device: torch.device, m: float = 0.0,
                      s: float = 1.0) -> torch.Tensor:
    """Sample t from logit-normal(m, s) — FLUX's own time distribution."""
    u = torch.randn(B, device=device) * s + m
    return torch.sigmoid(u)


# ── null-text embedding cache ────────────────────────────────────────────────

class _NullTextCache:
    """Cache the null-text embeddings so T5/CLIP are only called once."""

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


# ── main module ──────────────────────────────────────────────────────────────

class FLUXBrainDecoder(nn.Module):
    """Frozen FLUX.1-dev with a trainable BrainIPAdapter.

    Parameters
    ----------
    cfg : Config
        BrainFlow config dataclass — reads flux_model_id, ip_dim, ip_n_blocks,
        flux_guidance, flux_steps, data_dir.
    """

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

        # Lazily loaded FLUX pipeline (avoids import at module load time)
        self._pipe        = None
        self._null_cache  = _NullTextCache()
        self._hook_handles: list = []

        # Side-channel for hooks: populated before each FLUX forward pass.
        # Keys: block_idx → (cond_tokens, uncertainty_gate)
        self._adapter_inputs: dict = {}

        # register a buffer so .to(device) moves this module cleanly
        self.register_buffer("_device_anchor", torch.zeros(1), persistent=False)

    # ── lazy load ─────────────────────────────────────────────────────────────

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

        cache_dir = Path(self.cfg.data_dir) / "hf_cache"
        log.info("Loading FLUX.1-dev from %s (this may take a few minutes)…",
                 self.cfg.flux_model_id)

        pipe = FluxPipeline.from_pretrained(
            self.cfg.flux_model_id,
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
        )
        pipe = pipe.to(self._device_anchor.device)

        # Freeze everything
        pipe.transformer.eval().requires_grad_(False)
        pipe.vae.eval().requires_grad_(False)
        pipe.text_encoder.eval().requires_grad_(False)
        pipe.text_encoder_2.eval().requires_grad_(False)

        # Enable memory-efficient attention in the transformer if available
        if hasattr(pipe.transformer, "enable_xformers_memory_efficient_attention"):
            try:
                pipe.transformer.enable_xformers_memory_efficient_attention()
                log.info("xformers memory-efficient attention enabled.")
            except Exception:
                pass

        self._pipe = pipe
        log.info("FLUX.1-dev loaded. Transformer params: %dM (frozen)",
                 sum(p.numel() for p in pipe.transformer.parameters()) // 1_000_000)
        return self._pipe

    # ── adapter hook management ──────────────────────────────────────────────

    def _remove_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def _register_hooks(self, pipe):
        """Register a forward hook on each FLUX double-stream block.

        The hook adds the adapter residual to the image-token stream
        immediately after the block's own attention output, before the MLP.
        FLUX double-stream blocks expose `hidden_states` as the first return
        value (img tokens) and `encoder_hidden_states` as the second (txt).
        """
        transformer = pipe.transformer

        def _make_hook(block_idx: int):
            def hook(module, args, output):
                if block_idx not in self._adapter_inputs:
                    return output

                cond_tokens, gate = self._adapter_inputs[block_idx]

                # FLUX double-stream blocks return (img_out, txt_out) or a
                # single tensor depending on diffusers version.
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

        # FLUX.1-dev double-stream blocks are at transformer.transformer_blocks
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
        """Pre-compute conditioning tokens and gate; store for hooks."""
        cond_tokens, gate = self.ip_adapter.build_conditioning(
            brain_tokens, mu_clip, log_sigma
        )
        self._adapter_inputs = {
            i: (cond_tokens, gate)
            for i in range(self.cfg.ip_n_blocks)
        }

    # ── VAE encode / decode ───────────────────────────────────────────────────

    @torch.no_grad()
    def _vae_encode(self, images: torch.Tensor) -> torch.Tensor:
        """images (B,3,H,W) ∈ [0,1] → FLUX latents (B,16,H//8,W//8)."""
        pipe  = self._load_pipe()
        dtype = torch.bfloat16
        # FLUX expects [-1,1]
        x = images.to(dtype=dtype, device=pipe.vae.device) * 2.0 - 1.0
        latents = pipe.vae.encode(x).latent_dist.sample()
        return latents * _FLUX_VAE_SCALE

    # ── training loss ─────────────────────────────────────────────────────────

    def compute_loss(
        self,
        brain_tokens: torch.Tensor,   # (B, N, brain_dim)
        mu_clip: torch.Tensor,         # (B, clip_dim)
        log_sigma: torch.Tensor,       # (B, clip_dim)
        target_images: torch.Tensor,   # (B, 3, H, W)  ∈ [0,1]
    ) -> torch.Tensor:
        """FLUX flow-matching denoising loss with brain IP-Adapter conditioning.

        Only the BrainIPAdapter receives gradients.  FLUX is always in eval
        mode and no_grad is applied to its parameters.

        Returns
        -------
        loss : scalar tensor
        """
        pipe   = self._load_pipe()
        device = self._device_anchor.device
        dtype  = torch.bfloat16

        B = brain_tokens.shape[0]

        # 1. Encode target images to FLUX latent space (no grad)
        with torch.no_grad():
            z = self._vae_encode(target_images.to(device))  # (B,16,H/8,W/8)

        # 2. Logit-normal time sampling (FLUX distribution)
        t = _sample_flux_time(B, device)

        # 3. Linear interpolant  z_t = (1-t)*eps + t*z
        eps  = torch.randn_like(z)
        t4   = t.view(B, 1, 1, 1)
        z_t  = (1.0 - t4) * eps + t4 * z
        # velocity target
        v_target = z - eps                                    # (B,16,H/8,W/8)

        # 4. Pre-compute adapter conditioning and register hooks
        self._set_adapter_inputs(
            brain_tokens.to(device),
            mu_clip.to(device),
            log_sigma.to(device),
        )
        self._register_hooks(pipe)

        # 5. Null-text embeddings (fixed; T5 + CLIP-L)
        with torch.no_grad():
            prompt_embeds, pooled_embeds, text_ids = self._null_cache.get(
                pipe, device, dtype
            )
            # expand to batch
            prompt_embeds = prompt_embeds.expand(B, -1, -1)
            pooled_embeds = pooled_embeds.expand(B, -1)

        # 6. Patchify latents for FLUX transformer (pack into sequence)
        latent_image_ids = pipe._prepare_latent_image_ids(
            B, z_t.shape[2], z_t.shape[3], device, dtype
        )
        packed = pipe._pack_latents(z_t.to(dtype), *z_t.shape[2:])

        # 7. FLUX transformer forward — hooks fire inside here
        try:
            with torch.cuda.amp.autocast(dtype=dtype, enabled=True):
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
            # Always remove hooks even if forward fails
            self._remove_hooks()
            self._adapter_inputs.clear()

        # 8. Unpack prediction back to latent shape
        v_pred = pipe._unpack_latents(
            noise_pred, z_t.shape[2], z_t.shape[3], pipe.vae_scale_factor
        )

        # 9. Flow-matching MSE loss in fp32 for numerical stability
        loss = F.mse_loss(v_pred.float(), v_target.float())
        return loss

    # ── inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        brain_tokens: torch.Tensor,    # (B, N, brain_dim)
        mu_clip: torch.Tensor,          # (B, clip_dim)
        log_sigma: torch.Tensor,        # (B, clip_dim)
        height: int = 512,
        width:  int = 512,
        n_steps: Optional[int] = None,
        guidance: Optional[float] = None,
    ) -> torch.Tensor:
        """Generate images conditioned on brain signals.

        Returns
        -------
        images : (B, 3, H, W)  float32  ∈ [0, 1]
        """
        pipe    = self._load_pipe()
        device  = self._device_anchor.device
        dtype   = torch.bfloat16
        B       = brain_tokens.shape[0]
        n_steps = n_steps or self.cfg.flux_steps
        guidance_val = guidance if guidance is not None else self.cfg.flux_guidance

        # Pre-compute adapter conditioning
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

        images = result.images.clamp(0.0, 1.0)  # (B, 3, H, W)
        return images

    # ── device / dtype plumbing ───────────────────────────────────────────────

    def to(self, *args, **kwargs):
        out = super().to(*args, **kwargs)
        if self._pipe is not None:
            self._pipe = self._pipe.to(*args, **kwargs)
        return out

    def trainable_parameters(self):
        """Iterator over only the adapter parameters (for optimizer)."""
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
