from __future__ import annotations

from dataclasses import dataclass

NSD_TEST_IMAGES = 982
BIGG_TOKEN_LEN = 256
BIGG_TOKEN_DIM = 1664


@dataclass(frozen=True)
class ReconstructionScores:
    pixcorr: float | None
    ssim: float | None
    clip_two_way: float | None
    inception_two_way: float | None = None
    efficientnet_distance: float | None = None
    swav_distance: float | None = None


PUBLISHED = {
    "mindeye2": ReconstructionScores(0.322, 0.431, 0.930, 0.954, 0.618, 0.344),
    "mindeye1": ReconstructionScores(None, None, 0.933, 0.946, 0.622, 0.343),
    "brain_diffuser": ReconstructionScores(None, None, 0.909, 0.913, 0.728, 0.421),
    "takagi_nishimoto_2023": ReconstructionScores(0.246, 0.410, 0.821),
    "brain_it": ReconstructionScores(0.386, 0.486, None),
}

OURS_REGRESSION_ONLY = ReconstructionScores(0.164, 0.348, 0.703)
OURS_BLEND_HALF = ReconstructionScores(0.131, 0.276, 0.738)

ANCHOR_TOKEN_COSINE = 0.37

GROUND_TRUTH_TOKEN_CEILING = ReconstructionScores(0.64, 0.25, None)
GROUND_TRUTH_BLURRY_INIT_CEILING = ReconstructionScores(0.91, 0.48, None)

LOW_LEVEL_L1_COLLAPSE_FLOOR = 0.798
LOW_LEVEL_BLUR_MSE_COLLAPSE_FLOOR = 1.0
BLURRY_INIT_STRENGTH = 0.5


def low_level_head_has_collapsed(l1: float, tol: float = 1e-3) -> bool:
    return l1 >= LOW_LEVEL_L1_COLLAPSE_FLOOR - tol


def beats(scores: ReconstructionScores, reference: str) -> dict[str, bool]:
    ref = PUBLISHED[reference]
    out: dict[str, bool] = {}
    for field, lower_is_better in (("pixcorr", False), ("ssim", False),
                                   ("clip_two_way", False),
                                   ("inception_two_way", False),
                                   ("efficientnet_distance", True),
                                   ("swav_distance", True)):
        ours, theirs = getattr(scores, field), getattr(ref, field)
        if ours is None or theirs is None:
            continue
        out[field] = ours < theirs if lower_is_better else ours > theirs
    return out
