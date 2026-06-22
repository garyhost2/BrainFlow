from pathlib import Path
import torch
import torch.nn as nn
from diffusers import AutoencoderKL

class FrozenVAE(nn.Module):
    def __init__(self, cache_dir: Path | str | None = None,
                 dtype: torch.dtype = torch.float16):
        super().__init__()
        kwargs = {"torch_dtype": dtype}
        if cache_dir is not None:
            kwargs["cache_dir"] = str(cache_dir)
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse", **kwargs
        )
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.scale = 0.18215
        self.dtype = dtype

    @torch.no_grad()
    def encode(self, x):

        return (self.vae.encode((x * 2 - 1).to(self.dtype))
                .latent_dist.sample().float() * self.scale)

    @torch.no_grad()
    def decode(self, z):

        return ((self.vae.decode((z / self.scale).to(self.dtype))
                 .sample.float().clamp(-1, 1) + 1) / 2)

    def decode_grad(self, z):
        return ((self.vae.decode((z / self.scale).to(self.dtype))
                 .sample.float().clamp(-1, 1) + 1) / 2)
