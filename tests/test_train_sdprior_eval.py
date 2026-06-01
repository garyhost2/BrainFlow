import types

import torch
from torch.utils.data import DataLoader, Dataset

from scripts.train_sdprior import evaluate


class _EvalDS(Dataset):
    def __len__(self):
        return 5

    def __getitem__(self, idx):
        clip = torch.zeros(8)
        clip[idx % 8] = 1.0
        fmri = clip.clone()
        return {
            "fmri": fmri,
            "subject": torch.tensor(1),
            "clip_emb": clip,
        }


class _Brain(torch.nn.Module):
    def eval(self):
        return self

    def train(self):
        return self

    def forward(self, fmri, subject):
        return (fmri.unsqueeze(1), None)


class _Prior(torch.nn.Module):
    prior_target = "cls"

    def eval(self):
        return self

    def train(self):
        return self

    def sample(self, tokens, n_steps, cfg_scale, normalize_output):
        return torch.nn.functional.normalize(tokens.squeeze(1), dim=-1)


def test_train_sdprior_evaluate_reports_aggregated_clip_2way(monkeypatch):
    seen = {}

    def _fake_two_way(pred, gt):
        seen["shape"] = (pred.shape, gt.shape)
        return 0.77

    monkeypatch.setattr("scripts.train_sdprior.two_way_identification", _fake_two_way)

    metrics = evaluate(
        brain_enc=_Brain(),
        prior=_Prior(),
        loader=DataLoader(_EvalDS(), batch_size=2),
        device=torch.device("cpu"),
        cfg=types.SimpleNamespace(eval_ode_steps=2, cfg_scale=1.0),
        n_batches=None,
    )

    assert "CLIP_Cos" in metrics
    assert metrics["CLIP_2way"] == 0.77
    assert seen["shape"] == ((5, 8), (5, 8))
