from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class DecoderUnavailableError(RuntimeError):
    """Raised when the frozen decoder weights cannot be loaded."""


class FrozenImageEmbeddingDecoder(nn.Module):
    """
    Frozen CLIP-image-embedding -> image decoder wrapper.

    Uses diffusers StableUnCLIP image-variation pipeline and keeps all weights frozen.
    Loading is lazy so importing this module never triggers network access.
    """

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-2-1-unclip",
        cache_dir: str | Path = "models/hf_cache",
        torch_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.model_id = model_id
        self.cache_dir = Path(cache_dir)
        self.torch_dtype = torch_dtype
        self._pipe = None
        self.register_buffer("_device_anchor", torch.zeros(1), persistent=False)

    def _load_pipe(self):
        if self._pipe is not None:
            return self._pipe
        try:
            from diffusers import StableUnCLIPImg2ImgPipeline
        except Exception as e:  # pragma: no cover - dependency/runtime-specific
            raise DecoderUnavailableError(
                "FrozenImageEmbeddingDecoder requires diffusers with "
                "StableUnCLIPImg2ImgPipeline available."
            ) from e
        try:
            pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                cache_dir=self.cache_dir,
            )
        except Exception as e:
            raise DecoderUnavailableError(
                "Failed to load frozen decoder weights "
                f"'{self.model_id}'. If running offline, pre-cache this model under "
                f"'{self.cache_dir}'. Original error: {e}"
            ) from e

        pipe.set_progress_bar_config(disable=True)
        pipe = pipe.to(self._device_anchor.device)
        pipe.unet.eval().requires_grad_(False)
        pipe.vae.eval().requires_grad_(False)
        if getattr(pipe, "text_encoder", None) is not None:
            pipe.text_encoder.eval().requires_grad_(False)
        self._pipe = pipe
        return self._pipe

    def to(self, *args, **kwargs):
        out = super().to(*args, **kwargs)
        if self._pipe is not None:
            self._pipe = self._pipe.to(*args, **kwargs)
        return out

    @torch.no_grad()
    def generate(
        self,
        clip_image_embedding: torch.Tensor,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
    ) -> torch.Tensor:
        pipe = self._load_pipe()
        emb = clip_image_embedding.to(dtype=self.torch_dtype, device=pipe._execution_device)
        out = pipe(
            image_embeds=emb,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            output_type="pt",
        )
        images = out.images
        return images.clamp(0.0, 1.0)
