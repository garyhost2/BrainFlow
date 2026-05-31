"""Compute and cache the optional shared ridge warm-start for BrainEncoder."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brainflow.config import load_config
from brainflow.config_overrides import apply_env_overrides
from brainflow.data import build_or_load_tensors, compute_or_load_clip
from brainflow.ridge_init import ridge_init_path, save_ridge_init


def main():
    cfg = apply_env_overrides(load_config())
    tensors = build_or_load_tensors(cfg)
    clips = compute_or_load_clip(tensors, cfg)
    path = save_ridge_init(tensors, clips, cfg)
    print(f"✓ ridge init saved: {path}")
    print(f"  expected cache path: {ridge_init_path(cfg)}")


if __name__ == "__main__":
    main()
