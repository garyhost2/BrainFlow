import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brainflow.config import Config
from brainflow.models import BrainEncoder, BrainFlowV5, migrate_input_proj


def _wide_cfg() -> Config:
    cfg = Config()
    cfg.subjects = [1, 2]
    cfg.method = "baseline"
    cfg.enc_hidden = 64
    cfg.enc_blocks = 1
    cfg.enc_drop = 0.0
    cfg.stem_drop = 0.0
    cfg.n_tokens = 64
    cfg.brain_dim = 1024
    cfg.clip_dim = 768
    cfg.time_emb_dim = 64
    cfg.unet_base_ch = 32
    cfg.attn_heads = 4
    cfg.token_drop_prob = 0.0
    cfg.cfg_drop_prob = 0.0
    cfg.mixup_alpha = 0.0
    cfg.use_mixco = True
    cfg.use_subject_heads = True
    cfg.subject_head_rank = 4
    cfg.use_ridge_init = True
    cfg.ridge_init_cache = "missing-test-ridge-init.pt"
    cfg.lambda_percep = 0.0
    cfg.lambda_pixel = 0.0
    cfg.percep_loss = "none"
    return cfg


def test_brain_encoder_outputs_wide_tokens_and_clip_aligned_cls():
    cfg = _wide_cfg()
    enc = BrainEncoder(cfg, {1: 32, 2: 32})
    out = enc(torch.randn(3, 32), torch.tensor([1, 1, 1]))

    assert out.tokens.shape == (3, 64, 1024)
    assert out.cls.shape == (3, 1024)
    assert out.cls_align.shape == (3, 768)
    assert torch.allclose(out.cls_align.norm(dim=-1), torch.ones(3), atol=1e-5)


def test_training_step_uses_cls_align_for_wide_brain_dim():
    torch.manual_seed(0)
    cfg = _wide_cfg()
    model = BrainFlowV5(cfg, {1: 32, 2: 32})
    model.train()

    batch = {
        "fmri": torch.randn(2, 32),
        "latent": torch.randn(2, 4, 32, 32),
        "clip_emb": F.normalize(torch.randn(2, cfg.clip_dim), dim=-1),
        "subject": torch.tensor([1, 1], dtype=torch.long),
        "image": torch.rand(2, 3, 32, 32),
    }
    enc_out = model.encode_fmri(batch["fmri"], batch["subject"])
    logits = enc_out.cls_align @ batch["clip_emb"].T
    assert logits.shape == (2, 2)

    loss_dict = model.training_step(batch, torch.device("cpu"))

    assert torch.isfinite(loss_dict["loss"])
    assert torch.isfinite(loss_dict["align"])


def test_subject_adapter_routes_by_subject_id():
    cfg = _wide_cfg()
    enc = BrainEncoder(cfg, {1: 16, 2: 16})
    with torch.no_grad():
        enc.subject_adapter.down.weight.zero_()
        enc.subject_adapter.up.weight.zero_()
        enc.subject_adapter.bias.weight.zero_()
        enc.subject_adapter.down.weight[0, 0] = 1.0
        enc.subject_adapter.up.weight[0, 0] = 1.0
        enc.subject_adapter.bias.weight[0, 0] = 0.25
        enc.subject_adapter.down.weight[1, 1] = -1.0
        enc.subject_adapter.up.weight[1, enc.subject_adapter.rank + 1] = 1.0
        enc.subject_adapter.bias.weight[1, 1] = -0.25

    fmri = torch.randn(1, 16)
    out_a = enc(fmri, torch.tensor([1]))
    out_b = enc(fmri, torch.tensor([2]))

    assert not torch.allclose(out_a.cls, out_b.cls)
    assert not torch.allclose(out_a.cls_align, out_b.cls_align)


def test_old_checkpoint_loads_with_noop_new_paths():
    cfg = _wide_cfg()
    model = BrainFlowV5(cfg, {1: 16, 2: 16})
    old_state = {
        "brain_enc.input_proj.1.weight": torch.randn(cfg.enc_hidden, 16),
        "brain_enc.input_proj.1.bias": torch.randn(cfg.enc_hidden),
    }
    migrated = migrate_input_proj(old_state, model.brain_enc.max_vox)
    missing, unexpected = model.load_state_dict(migrated, strict=False)

    assert unexpected == []
    assert any(key.startswith("brain_enc.subject_adapter.") for key in missing)
    assert any(key.startswith("brain_enc.ridge_skip.") for key in missing)
    with torch.no_grad():
        assert torch.count_nonzero(model.brain_enc.subject_adapter.down.weight) == 0
        assert torch.count_nonzero(model.brain_enc.subject_adapter.up.weight) == 0
        assert torch.count_nonzero(model.brain_enc.subject_adapter.bias.weight) == 0
        assert torch.count_nonzero(model.brain_enc.ridge_skip.weight) == 0
        assert torch.count_nonzero(model.brain_enc.ridge_skip.bias) == 0

    fmri = torch.randn(1, 16)
    out_a = model.encode_fmri(fmri, torch.tensor([1]))
    out_b = model.encode_fmri(fmri, torch.tensor([2]))
    assert torch.allclose(out_a.cls, out_b.cls, atol=1e-6)
    assert torch.allclose(out_a.cls_align, out_b.cls_align, atol=1e-6)


def test_token_self_attention_is_identity_at_init():
    torch.manual_seed(0)
    cfg_base = _wide_cfg()
    cfg_base.enc_token_attn_layers = 0
    cfg_base.enc_token_attn_heads = 4
    enc_base = BrainEncoder(cfg_base, {1: 32, 2: 32})

    cfg_attn = _wide_cfg()
    cfg_attn.enc_token_attn_layers = 2
    cfg_attn.enc_token_attn_heads = 4
    enc_attn = BrainEncoder(cfg_attn, {1: 32, 2: 32})
    missing, unexpected = enc_attn.load_state_dict(enc_base.state_dict(), strict=False)
    enc_base.eval()
    enc_attn.eval()

    assert unexpected == []
    assert len(enc_attn.token_attn) == 2
    assert sum(p.numel() for p in enc_attn.parameters()) > sum(p.numel() for p in enc_base.parameters())
    assert all(key.startswith("token_attn.") for key in missing)

    fmri = torch.randn(3, 32)
    subject = torch.tensor([1, 1, 1], dtype=torch.long)
    out_base = enc_base(fmri, subject)
    out_attn = enc_attn(fmri, subject)

    assert torch.allclose(out_attn.tokens, out_base.tokens, atol=1e-6, rtol=1e-6)
    assert torch.allclose(out_attn.cls, out_base.cls, atol=1e-6, rtol=1e-6)
    assert torch.allclose(out_attn.cls_align, out_base.cls_align, atol=1e-6, rtol=1e-6)


def test_softclip_curriculum_switches_at_configured_epoch():
    torch.manual_seed(0)
    cfg = _wide_cfg()
    cfg.enc_token_attn_layers = 0
    cfg.use_mixco = False
    cfg.mixup_alpha = 0.0
    cfg.softclip_start_epoch = 1
    cfg.softclip_temp = 0.006

    model = BrainFlowV5(cfg, {1: 32, 2: 32})
    model.eval()
    batch = {
        "fmri": torch.randn(3, 32),
        "latent": torch.randn(3, 4, 32, 32),
        "clip_emb": F.normalize(torch.randn(3, cfg.clip_dim), dim=-1),
        "subject": torch.tensor([1, 1, 1], dtype=torch.long),
        "image": torch.rand(3, 3, 32, 32),
    }

    with torch.no_grad():
        enc_out = model.encode_fmri(batch["fmri"], batch["subject"])
        logits = ((enc_out.cls_align @ batch["clip_emb"].T) / cfg.infonce_temp).float()
        labels = torch.arange(batch["fmri"].shape[0], dtype=torch.long)
        soft_target = F.softmax(
            (batch["clip_emb"].float() @ batch["clip_emb"].float().T) / cfg.softclip_temp,
            dim=-1,
        )
        hard_align = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        soft_align = (
            -(soft_target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            - (soft_target.T * F.log_softmax(logits.T, dim=-1)).sum(dim=-1).mean()
        ) / 2

    assert torch.allclose(soft_target.sum(dim=-1), torch.ones(soft_target.shape[0]), atol=1e-6)

    loss_hard = model.training_step(batch, torch.device("cpu"), epoch=0)
    loss_soft = model.training_step(batch, torch.device("cpu"), epoch=1)
    loss_soft["loss"].backward()

    assert torch.allclose(loss_hard["align"], hard_align, atol=1e-6, rtol=1e-6)
    assert torch.allclose(loss_soft["align"], soft_align, atol=1e-6, rtol=1e-6)
    assert torch.isfinite(loss_soft["align"])
    assert model.brain_enc.align_proj.weight.grad is not None
    assert torch.isfinite(model.brain_enc.align_proj.weight.grad).all()
