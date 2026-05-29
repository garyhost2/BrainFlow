"""Tests for unified solver API: each solver integrates dy/dt = -y correctly."""
import math
import sys
from pathlib import Path
import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brainflow.solvers import solve, make_t_grid, euler, midpoint, heun, dpm_pp_2m


def _vel_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """dy/dt = -y  →  exact solution: y(t) = y0 * exp(-t)."""
    return -x


def _exact(x0: torch.Tensor, t_end: float = 1.0) -> torch.Tensor:
    return x0 * math.exp(-t_end)


class TestMakeTGrid:
    def test_shape(self):
        grid = make_t_grid(10)
        assert grid.shape == (11,)

    def test_range(self):
        grid = make_t_grid(20)
        assert abs(grid[0].item()) < 1e-6
        assert abs(grid[-1].item() - 1.0) < 1e-6

    def test_uniform_spacing(self):
        grid = make_t_grid(10)
        diffs = (grid[1:] - grid[:-1])
        assert torch.allclose(diffs, diffs[0].expand_as(diffs), atol=1e-6)

    def test_linear_identical_to_linspace(self):
        """'linear' must be byte-for-byte identical to torch.linspace."""
        ref = torch.linspace(0.0, 1.0, 21)
        out = make_t_grid(20, schedule="linear")
        assert torch.allclose(out, ref)

    def test_cosine_endpoints(self):
        grid = make_t_grid(20, schedule="cosine")
        assert abs(grid[0].item()) < 1e-6
        assert abs(grid[-1].item() - 1.0) < 1e-6

    def test_cosine_shape(self):
        grid = make_t_grid(15, schedule="cosine")
        assert grid.shape == (16,)

    def test_cosine_monotone(self):
        grid = make_t_grid(20, schedule="cosine")
        diffs = grid[1:] - grid[:-1]
        assert (diffs > 0).all(), "cosine grid must be strictly increasing"

    def test_logit_normal_endpoints(self):
        grid = make_t_grid(20, schedule="logit_normal")
        assert abs(grid[0].item()) < 1e-6
        assert abs(grid[-1].item() - 1.0) < 1e-6

    def test_logit_normal_shape(self):
        grid = make_t_grid(15, schedule="logit_normal")
        assert grid.shape == (16,)

    def test_logit_normal_monotone(self):
        grid = make_t_grid(20, schedule="logit_normal")
        diffs = grid[1:] - grid[:-1]
        assert (diffs > 0).all(), "logit_normal grid must be strictly increasing"

    def test_logit_normal_custom_params(self):
        grid = make_t_grid(20, schedule="logit_normal", logit_normal_m=0.5, logit_normal_s=2.0)
        assert abs(grid[0].item()) < 1e-6
        assert abs(grid[-1].item() - 1.0) < 1e-6
        assert (grid[1:] > grid[:-1]).all()

    def test_unknown_schedule_raises(self):
        with pytest.raises(ValueError, match="Unknown schedule"):
            make_t_grid(10, schedule="polynomial")


class TestEulerSolver:
    def test_integrates_linear_ode(self):
        """Euler should approximate dy/dt=-y within reasonable tolerance (n=200)."""
        x0 = torch.ones(4, 8)
        grid = make_t_grid(200)
        x_pred = solve(_vel_fn, x0, grid, method="euler")
        x_exact = _exact(x0)
        err = (x_pred - x_exact).abs().mean().item()
        assert err < 0.01, f"Euler error {err:.4e} too large"

    def test_output_shape(self):
        x0 = torch.randn(2, 4)
        grid = make_t_grid(5)
        out = solve(_vel_fn, x0, grid, method="euler")
        assert out.shape == x0.shape


class TestMidpointSolver:
    def test_integrates_linear_ode(self):
        """Midpoint should be more accurate than Euler with same NFE."""
        x0 = torch.ones(4, 8)
        grid = make_t_grid(50)
        x_pred = solve(_vel_fn, x0, grid, method="midpoint")
        x_exact = _exact(x0)
        err = (x_pred - x_exact).abs().mean().item()
        assert err < 0.005, f"Midpoint error {err:.4e} too large"

    def test_more_accurate_than_euler(self):
        x0 = torch.ones(4, 4)
        grid = make_t_grid(10)
        err_euler = (solve(_vel_fn, x0.clone(), grid, method="euler") - _exact(x0)).abs().mean().item()
        err_mid   = (solve(_vel_fn, x0.clone(), grid, method="midpoint") - _exact(x0)).abs().mean().item()
        assert err_mid <= err_euler, "Midpoint should not be worse than Euler"


class TestHeunSolver:
    def test_integrates_linear_ode(self):
        """Heun should be quite accurate (2nd-order)."""
        x0 = torch.ones(4, 8)
        grid = make_t_grid(20)
        x_pred = solve(_vel_fn, x0, grid, method="heun")
        x_exact = _exact(x0)
        err = (x_pred - x_exact).abs().mean().item()
        assert err < 0.001, f"Heun error {err:.4e} too large"


class TestDPMPP2MSolver:
    def test_integrates_linear_ode(self):
        """DPM-Solver++ 2M should converge to within tolerance."""
        x0 = torch.ones(4, 8)
        grid = make_t_grid(50)
        x_pred = solve(_vel_fn, x0, grid, method="dpm_pp_2m")
        x_exact = _exact(x0)
        err = (x_pred - x_exact).abs().mean().item()
        assert err < 0.05, f"DPM++ 2M error {err:.4e} too large"

    def test_output_shape(self):
        x0 = torch.randn(3, 6)
        grid = make_t_grid(10)
        out = solve(_vel_fn, x0, grid, method="dpm_pp_2m")
        assert out.shape == x0.shape


class TestAdaptiveRK45Solver:
    def test_integrates_linear_ode(self):
        """RK45 should be highly accurate."""
        try:
            import torchdiffeq
        except ImportError:
            pytest.skip("torchdiffeq not installed")

        x0 = torch.ones(4, 8)
        grid = make_t_grid(5)  # t_grid[0..1] used; RK45 adapts internally
        x_pred = solve(_vel_fn, x0, grid, method="adaptive_rk45", rtol=1e-5, atol=1e-5)
        x_exact = _exact(x0)
        err = (x_pred - x_exact).abs().mean().item()
        assert err < 1e-3, f"RK45 error {err:.4e} too large"


class TestSolveDispatch:
    def test_unknown_solver_raises(self):
        x0 = torch.randn(2, 4)
        grid = make_t_grid(5)
        with pytest.raises(ValueError, match="Unknown solver"):
            solve(_vel_fn, x0, grid, method="nonsense_solver")

    def test_all_solvers_run_without_error(self):
        x0 = torch.randn(2, 4)
        grid = make_t_grid(10)
        for method in ["euler", "midpoint", "heun", "dpm_pp_2m"]:
            out = solve(_vel_fn, x0.clone(), grid, method=method)
            assert out.shape == x0.shape, f"Solver {method} returned wrong shape"

    def test_tolerances_to_within_threshold(self):
        """All deterministic solvers should integrate dy/dt=-y within 5% at n=100."""
        x0 = torch.ones(2, 2)
        grid = make_t_grid(100)
        x_exact = _exact(x0)
        for method in ["euler", "midpoint", "heun", "dpm_pp_2m"]:
            x_pred = solve(_vel_fn, x0.clone(), grid, method=method)
            err = (x_pred - x_exact).abs().mean().item()
            assert err < 0.05, f"Solver {method} has error {err:.4e} > 5%"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
