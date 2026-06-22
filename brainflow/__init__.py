"""BrainFlow v5.1 — Conditional Flow Matching for fMRI-to-Image Reconstruction.

v5.1: Phase 1 bugfix patch — data leak (B1), DDP (B3), trial averaging (B4),
      token dropout (B5), midpoint/Heun solver (B6), CFG/ODE defaults (B7),
      mixup fix, unsharded eval loader, GPU throughput improvements.

v5.2 (Phase 2): Hierarchical VFM — Flow_CLIP DiT + VFM objective +
      CLIP-conditioned Flow_VAE + unified solver API.

Top-level names are imported LAZILY (PEP 562) so that lightweight subpackages
(e.g. ``brainflow.step1``) can be imported without pulling in the entire v5
stack (diffusers, Phase-2 chain, …).  ``from brainflow import BrainFlowV5`` still
works exactly as before — the import just happens on first access.
"""
import importlib

_LAZY = {
    "load_config": (".config", "load_config"),
    "Config": (".config", "Config"),
    "BrainFlowV5": (".models", "BrainFlowV5"),
    "BrainEncoder": (".models", "BrainEncoder"),
    "FlowUNet": (".models", "FlowUNet"),
    "EMA": (".ema", "EMA"),
    "FrozenVAE": (".vae", "FrozenVAE"),
    "vfm_loss": (".vfm", "vfm_loss"),
    "cfm_loss": (".vfm", "cfm_loss"),
    "velocity_from_posterior": (".vfm", "velocity_from_posterior"),
    "flow_loss": (".vfm", "flow_loss"),
    "FlowCLIPDiT": (".flow_clip_dit", "FlowCLIPDiT"),
    "FlowUNetV2": (".flow_unet", "FlowUNet"),
    "BrainFlowPhase2": (".phase2_model", "BrainFlowPhase2"),
    "ClipPrior": (".clip_prior", "ClipPrior"),
    "solve": (".solvers", "solve"),
    "make_t_grid": (".solvers", "make_t_grid"),
}

__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        module, attr = _LAZY[name]
        mod = importlib.import_module(module, __name__)
        value = getattr(mod, attr)
        globals()[name] = value          # cache so subsequent access is direct
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY))
