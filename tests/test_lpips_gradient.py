import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brainflow.config import Config
from brainflow.models import BrainFlowV5
from brainflow.vae import FrozenVAE


class _DummyVAE:
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z[:, :3].clamp(-1.0, 1.0)

    def decode_grad(self, z: torch.Tensor) -> torch.Tensor:
        return z[:, :3].clamp(-1.0, 1.0)


class _TinyAutoencoder:
    class _EncOut:
        def __init__(self, x):
            self.latent_dist = type("LatentDist", (), {"sample": lambda self_: x})()

    class _DecOut:
        def __init__(self, x):
            self.sample = x

    def __init__(self):
        self._weight = torch.nn.Parameter(torch.tensor(1.0))

    def eval(self):
        return self

    def parameters(self):
        return [self._weight]

    def encode(self, x):
        return self._EncOut(x)

    def decode(self, z):
        return self._DecOut(z)


def test_frozen_vae_decode_grad_and_decode_no_grad(monkeypatch):
    monkeypatch.setattr(
        "brainflow.vae.AutoencoderKL.from_pretrained",
        lambda *args, **kwargs: _TinyAutoencoder(),
    )
    vae = FrozenVAE(dtype=torch.float32)
    z = torch.randn(2, 3, 4, 4, requires_grad=True)
    out_no_grad = vae.decode(z)
    out_grad = vae.decode_grad(z)

    assert out_no_grad.requires_grad is False
    assert out_grad.requires_grad is True
    assert out_grad.grad_fn is not None
    assert all(not p.requires_grad for p in vae.vae.parameters())


def test_lpips_path_keeps_gradient():
    torch.manual_seed(0)
    cfg = Config()
    cfg.subjects = [1]
    cfg.method = "baseline"
    cfg.enc_hidden = 64
    cfg.enc_blocks = 2
    cfg.brain_dim = 64
    cfg.clip_dim = 64
    cfg.n_tokens = 8
    cfg.time_emb_dim = 64
    cfg.unet_base_ch = 32
    cfg.attn_heads = 4
    cfg.lambda_cfm = 0.0
    cfg.lambda_align = 0.1
    cfg.lambda_percep = 0.15
    cfg.lambda_pixel = 0.3
    cfg.percep_t_min = 0.0
    cfg.percep_t_power = 1.0
    cfg.percep_loss = "lpips"
    cfg.token_drop_prob = 0.0
    cfg.cfg_drop_prob = 0.0
    cfg.mixup_alpha = 0.0

    model = BrainFlowV5(cfg, {1: 64})
    model.train()

    bsz = 32
    batch = {
        "fmri": torch.randn(bsz, 64),
        "latent": torch.randn(bsz, 4, 32, 32),
        "clip_emb": F.normalize(torch.randn(bsz, cfg.clip_dim), dim=-1),
        "subject": torch.ones(bsz, dtype=torch.long),
        "image": torch.rand(bsz, 3, 32, 32),
    }
    loss_dict = model.training_step(
        batch=batch,
        device=torch.device("cpu"),
        vae=_DummyVAE(),
        percep_loss_fn=lambda pred, tgt: ((pred - tgt) ** 2).flatten(1).mean(1),
        epoch=0,
    )
    assert loss_dict["percep"].requires_grad
    assert loss_dict["percep"].grad_fn is not None
    assert loss_dict["pixel"].requires_grad
    assert loss_dict["pixel"].grad_fn is not None
    loss_dict["loss"].backward()

    assert model.brain_enc.input_proj.weight.grad is not None
    assert model.flow_unet.op.weight.grad is not None
    assert model.brain_enc.input_proj.weight.grad.abs().sum().item() > 0.0
    assert model.flow_unet.op.weight.grad.abs().sum().item() > 0.0


def test_context_length_matches_train_and_sample():
    cfg = Config()
    cfg.subjects = [1]
    cfg.method = "baseline"
    cfg.use_clip_prior = True
    cfg.enc_hidden = 64
    cfg.enc_blocks = 1
    cfg.brain_dim = 64
    cfg.clip_dim = 64
    cfg.n_tokens = 8
    cfg.time_emb_dim = 64
    cfg.unet_base_ch = 32
    cfg.attn_heads = 4
    model = BrainFlowV5(cfg, {1: 64})
    assert model._expected_context_tokens(True) == cfg.n_tokens + 1
    assert model._expected_context_tokens(False) == cfg.n_tokens
