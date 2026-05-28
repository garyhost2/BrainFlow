import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brainflow.config import Config
from brainflow.models import BrainFlowV5


class _DummyVAE:
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z[:, :3].clamp(-1.0, 1.0)


def test_lpips_path_keeps_gradient():
    torch.manual_seed(0)
    cfg = Config()
    cfg.subjects = [1]
    cfg.method = "baseline"
    cfg.enc_hidden = 64
    cfg.enc_blocks = 2
    cfg.brain_dim = 64
    cfg.n_tokens = 8
    cfg.time_emb_dim = 64
    cfg.unet_base_ch = 32
    cfg.attn_heads = 4
    cfg.lambda_cfm = 0.0
    cfg.lambda_align = 0.0
    cfg.lambda_percep = 0.15
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
        percep_loss_fn=lambda pred, tgt: F.mse_loss(pred, tgt),
        epoch=0,
    )
    loss_dict["loss"].backward()

    assert model.brain_enc.input_proj.weight.grad is not None
    assert model.flow_unet.op.weight.grad is not None
    assert model.brain_enc.input_proj.weight.grad.abs().sum().item() > 0.0
    assert model.flow_unet.op.weight.grad.abs().sum().item() > 0.0
