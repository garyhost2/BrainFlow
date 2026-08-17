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
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY))
