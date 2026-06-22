from __future__ import annotations

from pathlib import Path

import torch

UNCLIP_REPO = "stabilityai/stable-diffusion-2-1-unclip"

class UnCLIPDecoder:
    def __init__(self, device: torch.device, hf_cache: Path | None = None,
                 dtype: torch.dtype = torch.float16):
        from diffusers import StableUnCLIPImg2ImgPipeline

        pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
            UNCLIP_REPO, torch_dtype=dtype, variant="fp16" if dtype == torch.float16 else None,
            cache_dir=str(hf_cache) if hf_cache else None,
            safety_checker=None, requires_safety_checker=False,
        )
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=True)

        pipe.vae = pipe.vae.to(torch.float32)
        try:
            pipe.vae.config.force_upcast = True
        except Exception:
            pass
        self.pipe = pipe
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def decode(self, embeds_raw: torch.Tensor, *, num_inference_steps: int = 25,
               guidance_scale: float = 10.0, noise_level: int = 0,
               prompt: str = "", generator: torch.Generator | None = None,
               batch_size: int = 16) -> torch.Tensor:
        imgs = []
        for i in range(0, len(embeds_raw), batch_size):
            chunk = embeds_raw[i:i + batch_size].to(self.device, self.dtype)
            n = chunk.shape[0]
            out = self.pipe(
                image=None,
                prompt=[prompt] * n,
                image_embeds=chunk,
                noise_level=noise_level,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type="pt",
            ).images
            imgs.append(out.float().cpu())
        return torch.cat(imgs)
