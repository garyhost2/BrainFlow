import torch
import torch.nn.functional as F

from brainflow.step1.sphere import (
    random_sphere, project_tangent, exp_map, log_map,
    slerp, slerp_velocity, polar_encode, polar_decode, tangent_noise,
)
from brainflow.step1.model_tokens import TokenStep1Config, TokenStep1Model

torch.manual_seed(0)


# --------------------------------------------------------------------------
#  Geometry primitives (section 5 of the paper)
# --------------------------------------------------------------------------
def test_random_sphere_unit_norm():
    z = random_sphere((5, 7, 16), "cpu")
    assert torch.allclose(z.norm(dim=-1), torch.ones(5, 7), atol=1e-5)


def test_project_tangent_orthogonal():
    z = random_sphere((4, 16), "cpu")
    v = torch.randn(4, 16)
    w = project_tangent(v, z)
    assert torch.allclose((w * z).sum(-1), torch.zeros(4), atol=1e-5)


def test_exp_map_preserves_norm_and_identity():
    z = random_sphere((4, 16), "cpu")
    w = project_tangent(torch.randn(4, 16), z)
    out = exp_map(z, w)
    assert torch.allclose(out.norm(dim=-1), torch.ones(4), atol=1e-5)
    assert torch.allclose(exp_map(z, torch.zeros_like(z)), z, atol=1e-5)


def test_exp_log_inverse():
    z = random_sphere((6, 32), "cpu")
    y = random_sphere((6, 32), "cpu")
    assert torch.allclose(exp_map(z, log_map(z, y)), y, atol=1e-4)


def test_slerp_endpoints_and_norm():
    z0 = random_sphere((8, 24), "cpu")
    z1 = random_sphere((8, 24), "cpu")
    t0 = torch.zeros(8, 1)
    t1 = torch.ones(8, 1)
    assert torch.allclose(slerp(z0, z1, t0), z0, atol=1e-5)
    assert torch.allclose(slerp(z0, z1, t1), z1, atol=1e-5)
    zt = slerp(z0, z1, torch.full((8, 1), 0.37))
    assert torch.allclose(zt.norm(dim=-1), torch.ones(8), atol=1e-5)


def test_slerp_velocity_is_tangent_and_matches_finite_diff():
    z0 = random_sphere((8, 24), "cpu").double()
    z1 = random_sphere((8, 24), "cpu").double()
    t = torch.full((8, 1), 0.41, dtype=torch.float64)
    zt = slerp(z0, z1, t)
    v = slerp_velocity(z0, z1, t)
    assert torch.allclose((v * zt).sum(-1), torch.zeros(8, dtype=torch.float64), atol=1e-6)
    h = 1e-6
    fd = (slerp(z0, z1, t + h) - slerp(z0, z1, t - h)) / (2 * h)
    assert torch.allclose(v, fd, atol=1e-4)


def test_polar_round_trip():
    x = torch.randn(3, 5, 16)
    mean = torch.randn(16)
    z, r = polar_encode(x, mean)
    assert torch.allclose(z.norm(dim=-1), torch.ones(3, 5), atol=1e-5)
    assert torch.allclose(polar_decode(z, r, mean), x, atol=1e-4)


def test_tangent_noise_stays_on_sphere():
    z = random_sphere((4, 32), "cpu")
    out = tangent_noise(z, kappa=0.1)
    assert torch.allclose(out.norm(dim=-1), torch.ones(4), atol=1e-5)
    assert torch.allclose(tangent_noise(z, 0.0), z)


# --------------------------------------------------------------------------
#  Two-head model (sections 7-9)
# --------------------------------------------------------------------------
def _tiny_cfg(geometry, two_head=True, low_level=True):
    return TokenStep1Config(
        token_len=4, token_dim=8, cls_dim=10, brain_dim=6, n_brain_tokens=3,
        enc_hidden=16, enc_blocks=1, reg_depth=1, reg_heads=2,
        prior_width=16, prior_depth=1, prior_heads=2, time_dim=8,
        cls_width=16, cls_depth=1, cls_heads=2,
        low_level=low_level, ll_size=16, ll_base=32,
        geometry=geometry, two_head=two_head, lambda_radius=1.0, n_steps=4, subjects=[1],
    )


def _model(geometry, two_head=True, train=False, low_level=True):
    cfg = _tiny_cfg(geometry, two_head, low_level)
    model = TokenStep1Model(cfg, {1: 10})
    model.set_target_mean(torch.zeros(8))
    return model if train else model.eval()


def test_two_head_sphere_training_step_trains_both_flows():
    model = _model("sphere", two_head=True, train=True)
    fmri = torch.randn(2, 10)
    out = model.training_step(fmri, 1, None,
                              target_raw=torch.randn(2, 4, 8),
                              target_cls=torch.randn(2, 10))
    assert torch.isfinite(out["loss"]) and out["loss"].requires_grad
    for k in ("flow", "rcfm", "reg", "cos", "clip", "radius"):
        assert torch.isfinite(out[k])
    assert out["rcfm"] > 0          # the CLS head actually receives a signal
    assert out["clip"] > 0          # SoftCLIP anchored on the true CLS
    out["loss"].backward()


def test_two_head_predict_returns_unit_cls():
    model = _model("sphere", two_head=True)
    fmri = torch.randn(2, 10)
    for cond in ("regression", "prior", "blend"):
        tok, cls_hat = model.predict_tokens(fmri, 1, None, cond_source=cond)
        assert tok.shape == (2, 4, 8) and torch.isfinite(tok).all()
        assert cls_hat.shape == (2, 10)
        assert torch.allclose(cls_hat.norm(dim=-1), torch.ones(2), atol=1e-5)


def test_cls_conditioning_changes_patch_velocity():
    """The patch prior must actually use c_cls (not ignore the conditioning)."""
    model = _model("sphere", two_head=True)
    # The output head and AdaLN are zero-initialised; perturb them so the
    # conditioning pathways (extra K/V token + AdaLN modulation) are live.
    torch.nn.init.normal_(model.prior.out.weight, std=0.1)
    torch.nn.init.normal_(model.prior.out.bias, std=0.1)
    torch.nn.init.normal_(model.prior.cls_ada.weight, std=0.1)
    for blk in model.prior.blocks:
        torch.nn.init.normal_(blk.ada[-1].weight, std=0.1)
    z = random_sphere((2, 4, 8), "cpu")
    t = torch.full((2,), 0.5)
    brain = model.backbone(torch.randn(2, 10), 1)
    a = model.prior(z, t, brain, torch.randn(2, 10))
    b = model.prior(z, t, brain, torch.randn(2, 10))
    assert not torch.allclose(a, b)


def test_geodesic_euler_and_heun_both_finite_on_sphere():
    model = _model("sphere", two_head=True)
    fmri = torch.randn(2, 10)
    for solver in ("euler", "heun"):
        tok, cls_hat = model.predict_tokens(fmri, 1, None, cond_source="prior", solver=solver)
        assert torch.isfinite(tok).all() and torch.isfinite(cls_hat).all()


def test_single_head_predict_has_no_cls():
    model = _model("sphere", two_head=False)
    tok, cls_hat = model.predict_tokens(torch.randn(2, 10), 1, None, cond_source="prior")
    assert tok.shape == (2, 4, 8) and torch.isfinite(tok).all()
    assert cls_hat is None


def test_euclidean_two_head_path_trains():
    model = _model("euclidean", two_head=True, train=True)
    out = model.training_step(torch.randn(2, 10), 1, torch.randn(2, 4, 8),
                              target_cls=torch.randn(2, 10))
    assert torch.isfinite(out["loss"]) and out["loss"].requires_grad
    out["loss"].backward()


def test_euclidean_single_head_baseline_still_works():
    model = _model("euclidean", two_head=False, train=True)
    out = model.training_step(torch.randn(2, 10), 1, torch.randn(2, 4, 8))
    assert torch.isfinite(out["loss"]) and out["loss"].requires_grad
    assert out["rcfm"] == 0          # no CLS head -> no RCFM term
    out["loss"].backward()


# --------------------------------------------------------------------------
#  Low-level (blurry-image / img2img) pathway
# --------------------------------------------------------------------------
def test_low_level_head_predicts_blurry_image():
    model = _model("sphere", two_head=True)
    blur = model.predict_lowlevel(torch.randn(2, 10), 1)
    assert blur.shape == (2, 3, 16, 16)
    assert blur.min() >= 0.0 and blur.max() <= 1.0          # sigmoid output


def test_low_level_loss_trains():
    model = _model("sphere", two_head=True, train=True)
    out = model.training_step(torch.randn(2, 10), 1, None,
                              target_raw=torch.randn(2, 4, 8),
                              target_cls=torch.randn(2, 10),
                              target_img=torch.rand(2, 3, 32, 32))
    assert "low" in out and torch.isfinite(out["low"]) and out["low"] > 0
    out["loss"].backward()
    assert model.low_head.out.weight.grad is not None


def test_low_level_disabled():
    model = _model("sphere", two_head=True, low_level=False)
    assert model.low_head is None
    assert model.predict_lowlevel(torch.randn(2, 10), 1) is None
    out = model.training_step(torch.randn(2, 10), 1, None,
                              target_raw=torch.randn(2, 4, 8),
                              target_cls=torch.randn(2, 10),
                              target_img=torch.rand(2, 3, 32, 32))
    assert "low" not in out          # no head -> no term


def test_ll_loss_modes_all_train():
    import dataclasses
    for mode in ("l1", "huber", "mse"):
        cfg = dataclasses.replace(_tiny_cfg("sphere"), ll_loss=mode)
        model = TokenStep1Model(cfg, {1: 10}); model.train()
        out = model.training_step(torch.randn(2, 10), 1, None,
                                  target_raw=torch.randn(2, 4, 8),
                                  target_cls=torch.randn(2, 10),
                                  target_img=torch.rand(2, 3, 32, 32), low_only=True)
        assert torch.isfinite(out["low"]) and out["low"] > 0, mode
        out["loss"].backward()
        assert model.low_head.out.weight.grad is not None, mode


def test_low_head_can_overfit_two_distinct_layouts():
    # The MSE->mean failure would show as inability to separate two targets. With
    # the voxel-fed head + L1, a few steps must drive the loss well below the
    # constant-prediction floor (proves it learns structure, not the average).
    torch.manual_seed(0)
    model = _model("sphere", train=True)
    fmri = torch.randn(2, 10)
    tgt = torch.rand(2, 3, 32, 32)
    tgt[0] *= 0.2; tgt[1] = 1.0 - tgt[1] * 0.2          # two very different images
    opt = torch.optim.Adam(model.low_head.parameters(), lr=1e-2)
    first = None
    for _ in range(150):
        opt.zero_grad()
        out = model.training_step(fmri, 1, None, target_img=tgt, low_only=True)
        out["loss"].backward(); opt.step()
        first = first if first is not None else out["low"].item()
    assert out["low"].item() < 0.5 * first    # loss must drop, i.e. it fits the layouts


def test_low_only_fast_path_trains_only_the_head():
    # --freeze-token: freeze everything except low_head, then low_only training
    # must (a) skip the prior/CLS/contrastive terms, (b) produce a finite low-only
    # loss, and (c) send gradients to low_head but NOT to the frozen token model.
    model = _model("sphere", two_head=True, train=True)
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith("low_head"))

    out = model.training_step(torch.randn(2, 10), 1, None,
                              target_raw=torch.randn(2, 4, 8),
                              target_cls=torch.randn(2, 10),
                              target_img=torch.rand(2, 3, 32, 32),
                              low_only=True)
    assert set(out) == {"low", "loss"}                 # only the low-level term
    assert torch.isfinite(out["loss"]) and out["loss"] > 0
    out["loss"].backward()
    assert model.low_head.out.weight.grad is not None  # head learns
    # every frozen token-model param stayed grad-free (no prior drift)
    assert all(p.grad is None for n, p in model.named_parameters()
               if not n.startswith("low_head"))
