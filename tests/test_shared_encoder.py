"""Cross-subject encoder: one shared input projection + per-subject LoRA.

The property that motivates the change is parameter scaling. Under the legacy
per-subject ModuleDict, subject k costs a full nn.Linear(V_k, enc_hidden) and
shares nothing at the input stage; under the shared encoder it costs
2*h*r + h. These tests pin that, plus the invariants that make the swap safe:
the adapter is the identity at initialisation, padded columns take no gradient,
and an old checkpoint migrates instead of silently starting fresh.
"""
import re

import pytest
import torch

from brainflow.step1.model_tokens import TokenStep1Config, TokenBackbone

# Real NSD voxel counts, so the parameter arithmetic below is the true one.
VOX = {1: 15724, 2: 14278, 5: 13039, 7: 12682}


def _backbone(shared, subs=None, rank=16, h=256):
    subs = dict(VOX) if subs is None else subs
    cfg = TokenStep1Config(subjects=sorted(subs), enc_hidden=h,
                           shared_encoder=shared, subject_rank=rank)
    torch.manual_seed(0)
    return TokenBackbone(cfg, subs)


def _nparams(m):
    return sum(p.numel() for p in m.parameters())


def test_shared_is_the_default():
    assert TokenStep1Config().shared_encoder is True


@pytest.mark.parametrize("shared", [False, True])
def test_forward_shape_unchanged(shared):
    bb = _backbone(shared).eval()
    with torch.no_grad():
        out = bb(torch.randn(4, VOX[2]), 2)
    assert out.shape == (4, bb.cfg.n_brain_tokens, bb.cfg.brain_dim)


def test_extra_subject_is_cheap_shared_and_expensive_per_subject():
    one = {1: VOX[1]}
    per_1, per_4 = _nparams(_backbone(False, one)), _nparams(_backbone(False))
    shr_1, shr_4 = _nparams(_backbone(True, one)), _nparams(_backbone(True))

    # Legacy: three more subjects buy three more voxel matrices.
    assert per_4 - per_1 == sum(VOX[s] for s in (2, 5, 7)) * 256 + 3 * 256

    # Shared: three more subjects buy three LoRA rows (2*h*r + h each).
    assert shr_4 - shr_1 == 3 * (2 * 256 * 16 + 256)
    assert (shr_4 - shr_1) < (per_4 - per_1) / 100


def test_adapter_is_identity_at_init():
    """Zero-init means adding a subject cannot perturb the others at step 0."""
    bb = _backbone(True).eval()
    x = torch.randn(4, bb.max_vox)
    with torch.no_grad():
        h = bb.input_proj(x)
        idx = bb._subject_indices(2, 4, x.device)
        assert torch.equal(bb.subject_adapter(h, idx), h)


def test_scalar_and_tensor_subject_agree():
    bb = _backbone(True).eval()
    x = torch.randn(4, VOX[2])
    with torch.no_grad():
        assert torch.equal(bb(x, 2), bb(x, torch.tensor([2, 2, 2, 2])))


def test_mixed_subject_batch_matches_per_row_forward():
    """A batch spanning subjects must equal each row run on its own."""
    bb = _backbone(True).eval()
    ids = [1, 5, 7]
    xb = torch.zeros(len(ids), bb.max_vox)
    for i, s in enumerate(ids):
        xb[i, :VOX[s]] = torch.randn(VOX[s])
    with torch.no_grad():
        mixed = bb(xb, torch.tensor(ids))
        solo = torch.cat([bb(xb[i:i + 1], s) for i, s in enumerate(ids)])
    assert torch.allclose(mixed, solo, atol=1e-5)


def test_padded_columns_receive_no_gradient():
    bb = _backbone(True)
    bb(torch.randn(2, VOX[7]), 7).sum().backward()
    g = bb.input_proj.weight.grad
    assert g[:, VOX[7]:].abs().sum().item() == 0.0
    assert g[:, :VOX[7]].abs().sum().item() > 0.0


def test_unknown_subject_raises():
    bb = _backbone(True).eval()
    with pytest.raises(KeyError):
        bb(torch.randn(2, VOX[1]), 3)


def test_too_wide_input_raises():
    bb = _backbone(True).eval()
    with pytest.raises(ValueError):
        bb(torch.randn(2, bb.max_vox + 1), 1)


# --------------------------------------------------------------------------
#  Checkpoint migration
# --------------------------------------------------------------------------
class _Holder(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone


def _old_state(subs):
    bb = _backbone(False, subs)
    return {f"backbone.{k}": v for k, v in bb.state_dict().items()}


def test_migration_seeds_shared_layer_from_checkpoint():
    from scripts.train_step1b import _migrate_input_proj

    old = _old_state({1: VOX[1]})
    model = _Holder(_backbone(True, VOX))
    mig = _migrate_input_proj(model, dict(old), "test")

    assert "backbone.input_proj.weight" in mig
    assert not any(re.fullmatch(r"backbone\.input_proj\.\d+\.(weight|bias)", k)
                   for k in mig)
    seeded = mig["backbone.input_proj.weight"][:, :VOX[1]]
    assert torch.equal(seeded, old["backbone.input_proj.1.weight"])

    missing, unexpected = model.load_state_dict(mig, strict=False)
    assert not unexpected
    # Only the three zero-init adapter tensors should be fresh.
    assert set(missing) == {
        "backbone.subject_adapter.down.weight",
        "backbone.subject_adapter.up.weight",
        "backbone.subject_adapter.bias.weight",
    }


def test_migration_prefers_widest_subject_present_in_run():
    """Voxel index j only means the same thing for the same subject, so an
    in-run subject beats a wider out-of-run one."""
    from scripts.train_step1b import _migrate_input_proj

    old = _old_state({1: VOX[1], 7: VOX[7]})          # 1 is wider than 7
    model = _Holder(_backbone(True, {7: VOX[7]}))     # but only 7 is in the run
    mig = _migrate_input_proj(model, dict(old), "test")
    assert torch.equal(mig["backbone.input_proj.weight"][:, :VOX[7]],
                       old["backbone.input_proj.7.weight"])


def test_migration_is_noop_for_per_subject_target():
    from scripts.train_step1b import _migrate_input_proj

    old = _old_state({1: VOX[1]})
    model = _Holder(_backbone(False, {1: VOX[1]}))
    assert _migrate_input_proj(model, dict(old), "test") == old
