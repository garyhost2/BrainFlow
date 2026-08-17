"""Alignment contract for the per-subject tensor cache.

Deliberately dependency-free: every script that reads the cache imports this,
and pulling in ``brainflow.data`` for a guard would drag ``webdataset`` and
``h5py`` into scripts that need neither.
"""
from __future__ import annotations

# behav column 5 is MindEye2's ``global_trial = (SESSION-1)*750 + jj``, a 0-based
# positional index that MindEye2 consumes unshifted. We indexed the betas with
# ``all_trial - 1`` while indexing the COCO images unshifted, pairing every image
# with the PREVIOUS trial's fMRI. scripts/check_offset measured it on subject 1:
# top-1 retrieval 0.1305 at +0 against 0.0070 at -1, chance 0.0005 -- the -1
# pairing destroyed 19/20 of the stimulus-locked signal. Every result produced
# before 2026-08-17 came from that pairing.
BEHAV_TO_BETAS_OFFSET = 0

# The old cache carried no version stamp at all, so an UNSTAMPED cache is by
# definition the misaligned one. That is why the check is equality against this
# string rather than a >= comparison.
TENSOR_CACHE_FORMAT = "tensors_v4_offset0_fullsplit"

# NSD shows each image about three times. max_train/max_test capped TRIALS while
# being set to IMAGE counts (8859 / 982), so the test split repeat-averaged down
# to 552 unique images and training saw roughly a third of the available trials.
NSD_TEST_IMAGES = 982
NSD_REPEATS_PER_IMAGE = 3


def tensor_cache_meta(git_sha: str = "unknown") -> dict:
    return {"format": TENSOR_CACHE_FORMAT,
            "behav_to_betas_offset": BEHAV_TO_BETAS_OFFSET,
            "git_sha": git_sha}


def assert_tensor_cache_alignment(cache, payload: dict) -> dict:
    """Refuse a tensor cache built before the offset and split were corrected.

    Every entry point must call this: train_step1b and the eval/sweep/smoke
    scripts load the cache with a bare ``torch.load`` rather than through
    ``build_or_load_tensors``, so a guard in the builder alone protects nothing.
    """
    meta = payload.get("_meta")
    got = meta.get("format") if isinstance(meta, dict) else None
    if got != TENSOR_CACHE_FORMAT:
        raise RuntimeError(
            f"{cache} predates the current split contract "
            f"(_meta.format={got!r}, expected {TENSOR_CACHE_FORMAT!r}). Caches built "
            f"before this stamp are wrong in one of two ways: offset -1 paired each "
            f"image with the PREVIOUS trial's fMRI (subject-1 top-1 retrieval 0.007 "
            f"against 0.131), and the trial caps truncated the test split to 552 "
            f"unique images and training to a third of its trials. Rebuild it "
            f"(force_rebuild=true). The bigG target caches derive from images only "
            f"and stay valid."
        )
    return payload
