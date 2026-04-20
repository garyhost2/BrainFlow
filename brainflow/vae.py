"""Frozen Stable Diffusion VAE wrapper (sd-vae-ft-mse)."""
from pathlib import Path
import torch
import torch.nn as nn
from diffusers import AutoencoderKL


class FrozenVAE(nn.Module):
    def __init__(self, cache_dir: Path | str | None = None):
        super().__init__()
        kwargs = {"torch_dtype": torch.float32}
        if cache_dir is not None:
            kwargs["cache_dir"] = str(cache_dir)
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse", **kwargs
        )
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.scale = 0.18215

    @torch.no_grad()
    def encode(self, x):
        return self.vae.encode(x * 2 - 1).latent_dist.sample() * self.scale

    @torch.no_grad()
    def decode(self, z):
        return (self.vae.decode(z / self.scale).sample.clamp(-1, 1) + 1) / 2
