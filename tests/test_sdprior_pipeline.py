import types

import torch
from torch.utils.data import DataLoader, Dataset

from brainflow.clip_prior import ClipPrior
from brainflow.config import Config
from brainflow.decoder import DecoderUnavailableError, FrozenImageEmbeddingDecoder
from brainflow.models import BrainEncoder
from scripts.eval_sdprior import evaluate_sdprior
from scripts.train_sdprior import compute_sdprior_losses


def _tiny_cfg() -> Config:
    cfg = Config()
    cfg.enc_hidden = 32
    cfg.enc_blocks = 1
    cfg.enc_drop = 0.0
    cfg.stem_drop = 0.0
    cfg.n_tokens = 4
    cfg.brain_dim = 16
    cfg.clip_dim = 8
    cfg.attn_heads = 2
    cfg.subjects = [1]
    cfg.cfg_drop_prob = 0.0
    cfg.sigma_min = 1e-4
    cfg.infonce_temp = 0.07
    cfg.lambda_cfm = 1.0
    cfg.lambda_align = 0.1
    cfg.prior_ode_steps = 2
    cfg.cfg_scale = 1.0
    cfg.decoder_num_inference_steps = 2
    cfg.decoder_guidance_scale = 1.0
    return cfg


def test_decoder_import_and_clear_load_error(monkeypatch):
    class _FailPipe:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise OSError("offline")

    monkeypatch.setitem(
        __import__("sys").modules,
        "diffusers",
        types.SimpleNamespace(StableUnCLIPImg2ImgPipeline=_FailPipe),
    )

    dec = FrozenImageEmbeddingDecoder(model_id="bad/model")
    with torch.no_grad():
        try:
            dec.generate(torch.randn(1, 768))
        except DecoderUnavailableError as e:
            assert "Failed to load frozen decoder weights" in str(e)
        else:
            raise AssertionError("Expected DecoderUnavailableError")


def test_sdprior_tiny_training_loss_is_finite_and_differentiable():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    enc = BrainEncoder(cfg, {1: 12})
    prior = ClipPrior(
        clip_dim=cfg.clip_dim,
        ctx_dim=cfg.brain_dim,
        dim=32,
        depth=1,
        heads=2,
        cfg_drop=0.0,
    )

    fmri = torch.randn(3, 12)
    subject = torch.ones(3, dtype=torch.long)
    clip_gt = torch.randn(3, cfg.clip_dim)
    brain_out = enc(fmri, subject)
    loss, _, _ = compute_sdprior_losses(prior, brain_out.tokens, brain_out.cls_align, clip_gt, cfg)

    assert torch.isfinite(loss)
    assert loss.grad_fn is not None
    loss.backward()
    assert enc.input_proj.weight.grad is not None
    assert prior.x_proj.weight.grad is not None


def test_sdprior_patch_target_loss_is_finite_and_differentiable():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    cfg.prior_target = "patches"
    enc = BrainEncoder(cfg, {1: 12})
    prior = ClipPrior(
        clip_dim=cfg.clip_dim,
        ctx_dim=cfg.brain_dim,
        dim=32,
        depth=1,
        heads=2,
        cfg_drop=0.0,
        prior_target="patches",
        patch_tokens=4,
        patch_dim=6,
        patch_valid_tokens=4,
    )

    fmri = torch.randn(3, 12)
    subject = torch.ones(3, dtype=torch.long)
    clip_gt = torch.randn(3, cfg.clip_dim)
    clip_patches = torch.randn(3, 4, 6)
    brain_out = enc(fmri, subject)
    loss, _, _ = compute_sdprior_losses(
        prior, brain_out.tokens, brain_out.cls_align, clip_gt, cfg, clip_patches=clip_patches
    )

    assert torch.isfinite(loss)
    assert loss.grad_fn is not None
    loss.backward()
    assert enc.input_proj.weight.grad is not None
    assert prior.x_proj.weight.grad is not None


def test_clip_prior_cross_attention_receives_full_token_sequence():
    prior = ClipPrior(clip_dim=8, ctx_dim=6, dim=16, depth=1, heads=2, cfg_drop=0.0)
    ctx = torch.randn(2, 7, 6)
    clip_gt = torch.randn(2, 8)
    seen = {}

    orig = prior.blocks[0].cross_attn.forward

    def _wrapped(*args, **kwargs):
        # args: (self, query, key, value, ...)
        seen["ctx_len"] = args[2].shape[1]
        return orig(*args[1:], **kwargs)

    prior.blocks[0].cross_attn.forward = _wrapped
    _ = prior.flow_loss(clip_gt, ctx)
    assert seen["ctx_len"] == ctx.shape[1]


class _EvalDS(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return {
            "fmri": torch.randn(12),
            "subject": torch.tensor(1),
            "clip_emb": torch.randn(8),
            "image": torch.full((3, 64, 64), 0.5),
        }


def test_eval_sdprior_outputs_cosine_and_2way_keys(monkeypatch):
    cfg = _tiny_cfg()

    class _Brain(torch.nn.Module):
        def eval(self):
            return self

        def forward(self, fmri, subject):
            b = fmri.shape[0]
            return types.SimpleNamespace(tokens=torch.randn(b, 4, 16))

    class _Prior(torch.nn.Module):
        def eval(self):
            return self

        def sample(self, tokens, n_steps, cfg_scale, normalize_output):
            return torch.nn.functional.normalize(torch.randn(tokens.shape[0], 8), dim=-1)

    class _Decoder:
        def generate(self, clip_image_embedding=None, clip_patch_tokens=None, num_inference_steps=1, guidance_scale=1.0):
            b = (clip_image_embedding if clip_image_embedding is not None else clip_patch_tokens).shape[0]
            return torch.full((b, 3, 64, 64), 0.25)

    monkeypatch.setattr(
        "scripts.eval_sdprior.evaluate_full",
        lambda pred, tgt, device=None: {
            "PixCorr": 0.1,
            "SSIM": 0.2,
            "CLIP_2way": 0.3,
            "Inception_2way": 0.4,
            "EffNet-B_2way": 0.5,
            "SwAV_2way": 0.6,
        },
    )

    out = evaluate_sdprior(
        brain_enc=_Brain(),
        prior=_Prior(),
        decoder=_Decoder(),
        loader=DataLoader(_EvalDS(), batch_size=2),
        cfg=cfg,
        device=torch.device("cpu"),
    )
    assert "cosine" in out
    for k in ("CLIP_2way", "Inception_2way", "EffNet-B_2way", "SwAV_2way"):
        assert k in out
