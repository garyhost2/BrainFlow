from __future__ import annotations

BEHAV_TO_BETAS_OFFSET = 0

TENSOR_CACHE_FORMAT = "tensors_v4_offset0_fullsplit"

NSD_TEST_IMAGES = 982
NSD_REPEATS_PER_IMAGE = 3


def tensor_cache_meta(git_sha: str = "unknown") -> dict:
    return {"format": TENSOR_CACHE_FORMAT,
            "behav_to_betas_offset": BEHAV_TO_BETAS_OFFSET,
            "git_sha": git_sha}


def assert_tensor_cache_alignment(cache, payload: dict) -> dict:
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
