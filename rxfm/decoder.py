from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def quiet_benign_warnings():
    import warnings
    import logging
    warnings.filterwarnings("ignore", message=".*use_reentrant.*")
    warnings.filterwarnings("ignore", message=".*None of the inputs have requires_grad.*")
    for name in ("sgm", "sgm.modules.attention", "sgm.modules.diffusionmodules.model"):
        logging.getLogger(name).setLevel(logging.ERROR)


class SDXLUnCLIPDecoder:
    def __init__(self, device, mindeye_src, ckpt_path, num_steps=38, out_size=256,
                 cls_vector_slot=(0, 1280), img2img_cfg=5.0):
        self.device = device
        self.num_steps = num_steps
        self.out_size = out_size
        self.cls_slot = tuple(cls_vector_slot)
        self.img2img_cfg = img2img_cfg
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
        self.cls_in_vector = self.cls_slot is not None and hi <= V
        if self.cls_in_vector:
            print(f"✓ SDXL-unCLIP ready. vector {tuple(self.vector_suffix.shape)} "
                  f"| c_cls -> vector[{lo}:{hi}]")
        else:
            print(f"✓ SDXL-unCLIP ready. vector width {V}: NO pooled-CLS slot "
                  f"(micro-conditioning only). Image semantics flow through the "
                  f"256x1664 crossattn tokens; c_cls is NOT routed to the decoder.")

    def _vector(self, cls_emb_row):
        vec = self.vector_suffix.clone()
        if cls_emb_row is not None and self.cls_in_vector:
            lo, hi = self.cls_slot
            vec[:, lo:hi] = cls_emb_row.to(vec.device, vec.dtype).reshape(1, hi - lo)
        return vec

    LATENT_SHAPE = (4, 96, 96)

    @torch.no_grad()
    def _encode_init(self, img):
        x = F.interpolate(img.to(self.device).float(), 768, mode="bilinear",
                          align_corners=False).clamp(0, 1)
        with torch.cuda.amp.autocast(dtype=torch.float16), self.engine.ema_scope():
            return self.engine.encode_first_stage(x * 2.0 - 1.0)

    @torch.no_grad()
    def encode_blurry(self, img, ll_size: int | None = None):
        x = img.to(self.device).float()
        if x.dtype == torch.uint8:
            x = x / 255.0
        if ll_size:
            x = F.interpolate(x.clamp(0, 1), ll_size, mode="bilinear", align_corners=False)
        return self._encode_init(x.clamp(0, 1))

    @torch.no_grad()
    def _recon_one(self, x, vector, init_latent=None, strength=1.0, num_samples=1):
        eng = self.engine
        device = self.device
        append_dims = self._append_dims
        if x.ndim == 2:
            x = x[None]
        x = x[[0]]
        with torch.cuda.amp.autocast(dtype=torch.float16), eng.ema_scope():
            c = {"crossattn": x.repeat(num_samples, 1, 1), "vector": vector.repeat(num_samples, 1)}
            uc = {"crossattn": torch.randn_like(x).repeat(num_samples, 1, 1),
                  "vector": vector.repeat(num_samples, 1)}
            for k in c:
                c[k] = c[k][:num_samples].to(device)
                uc[k] = uc[k][:num_samples].to(device)

            sigmas = eng.sampler.discretization(eng.sampler.num_steps).to(device)

            def denoiser(xx, ss, cc):
                return eng.denoiser(eng.model, xx, ss, cc)

            if init_latent is None or strength >= 1.0:
                z = torch.randn(num_samples, 4, 96, 96, device=device)
                noise = torch.randn_like(z)
                if self.offset_noise_level > 0.0:
                    noise = noise + self.offset_noise_level * append_dims(
                        torch.randn(z.shape[0], device=device), z.ndim)
                noised_z = (z + noise * append_dims(sigmas[0], z.ndim)) / torch.sqrt(1.0 + sigmas[0] ** 2)
                samples_z = eng.sampler(denoiser, noised_z, cond=c, uc=uc)
            else:
                z0 = init_latent.repeat(num_samples, 1, 1, 1).to(device).float()
                N = len(sigmas) - 1
                n_steps = max(1, int(round(strength * N)))
                i0 = N - n_steps
                x_cur = z0 + torch.randn_like(z0) * append_dims(sigmas[i0], z0.ndim)
                for i in range(i0, N):
                    s, s_next = sigmas[i], sigmas[i + 1]
                    ss = s.repeat(num_samples)
                    dc = denoiser(x_cur, ss, c)
                    du = denoiser(x_cur, ss, uc)
                    denoised = du + self.img2img_cfg * (dc - du)
                    d = (x_cur - denoised) / append_dims(s, x_cur.ndim)
                    x_cur = x_cur + d * append_dims(s_next - s, x_cur.ndim)
                samples_z = x_cur

            samples_x = eng.decode_first_stage(samples_z)
            samples = torch.clamp(samples_x * 0.8 + 0.2, 0.0, 1.0)
        return samples

    @torch.no_grad()
    def decode(self, tokens_raw: torch.Tensor, cls_emb: torch.Tensor | None = None,
               init_image: torch.Tensor | None = None, strength: float = 1.0,
               init_latent: torch.Tensor | None = None) -> torch.Tensor:
        have_init = init_latent is not None or init_image is not None
        use_init = have_init and strength < 1.0
        imgs = []
        for i in range(tokens_raw.shape[0]):
            vec = self._vector(None if cls_emb is None else cls_emb[i:i + 1])
            if not use_init:
                il = None
            elif init_latent is not None:
                il = init_latent[i:i + 1].to(self.device).float()
            else:
                il = self._encode_init(init_image[i:i + 1])
            s = self._recon_one(tokens_raw[i:i + 1].to(self.device), vec,
                                init_latent=il, strength=strength, num_samples=1)
            if self.out_size and self.out_size != s.shape[-1]:
                s = F.interpolate(s, self.out_size, mode="bilinear", align_corners=False)
            imgs.append(s.float().cpu())
        return torch.cat(imgs)
