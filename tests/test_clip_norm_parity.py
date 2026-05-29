import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brainflow.config import Config
from brainflow.data import compute_or_load_clip
from brainflow import metrics


class _FakeClipModel(nn.Module):
    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(2, 3))
        std = x.std(dim=(2, 3))
        return torch.cat([mean, std], dim=-1)


def test_clip_cache_normalization_matches_metrics_pipeline(tmp_path, monkeypatch):
    def _create_model_and_transforms(*args, **kwargs):
        return _FakeClipModel(), None, None

    monkeypatch.setitem(
        sys.modules,
        "open_clip",
        types.SimpleNamespace(create_model_and_transforms=_create_model_and_transforms),
    )

    cfg = Config()
    cfg.subjects = [1]
    cfg.data_dir = tmp_path
    cfg.force_rebuild = True
    cfg.clip_cache = "all_subjects_clip_v2.pt"

    batch = torch.rand(16, 3, 32, 32)
    tensors = {
        "imgs_train_1": batch.clone(),
        "imgs_test_1": batch[:2].clone(),
    }

    out = compute_or_load_clip(tensors, cfg)
    encoded = out["clip_train_1"]

    clip_enc = _FakeClipModel()
    expected = F.normalize(
        clip_enc.encode_image(metrics._clip_preprocess(batch, torch.device("cpu"))).float(),
        dim=-1,
    )
    cos = F.cosine_similarity(encoded, expected, dim=-1)
    assert torch.all(cos >= 0.9999), f"min cosine={cos.min().item():.6f}"
