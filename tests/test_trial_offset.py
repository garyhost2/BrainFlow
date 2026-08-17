"""The behav->betas trial offset, and the guard that stops a stale cache loading.

scripts/check_offset measured the pairing on subject 1 rather than arguing it
from source: top-1 retrieval 0.1305 at offset +0 against 0.0070 at -1, chance
0.0005. The repo indexed betas with `all_trial - 1` while indexing images
unshifted, so every image was paired with the previous trial's fMRI and 19/20 of
the stimulus-locked signal was destroyed.

The cache that shipped carried no `_meta` at all, so an unstamped cache IS the
misaligned one. These tests pin both halves: the constant, and the refusal.
"""
import pytest

from brainflow.tensor_cache import (BEHAV_TO_BETAS_OFFSET, TENSOR_CACHE_FORMAT,
                                    assert_tensor_cache_alignment,
                                    tensor_cache_meta)


def test_offset_is_zero():
    # MindEye2 defines behav column 5 as a 0-based global_trial and consumes it
    # unshifted; check_offset confirms it empirically. Anything else is the bug.
    assert BEHAV_TO_BETAS_OFFSET == 0


def test_unstamped_cache_is_refused():
    with pytest.raises(RuntimeError, match="offset -1"):
        assert_tensor_cache_alignment("cache.pt", {"voxels": {}, "fmri_train_1": None})


def test_cache_with_foreign_stamp_is_refused():
    payload = {"_meta": {"format": "tensors_v2"}}
    with pytest.raises(RuntimeError, match="expected"):
        assert_tensor_cache_alignment("cache.pt", payload)


def test_meta_is_not_a_dict_is_refused():
    with pytest.raises(RuntimeError):
        assert_tensor_cache_alignment("cache.pt", {"_meta": "tensors_v3_offset0"})


def test_correctly_stamped_cache_passes_through():
    payload = {"_meta": tensor_cache_meta("deadbeef"), "voxels": {1: 15724}}
    assert assert_tensor_cache_alignment("cache.pt", payload) is payload


def test_meta_records_the_offset_that_built_it():
    meta = tensor_cache_meta()
    assert meta["format"] == TENSOR_CACHE_FORMAT
    assert meta["behav_to_betas_offset"] == BEHAV_TO_BETAS_OFFSET


def test_the_refusal_message_tells_you_what_to_do():
    # A guard that fires without saying "rebuild" costs someone an afternoon.
    with pytest.raises(RuntimeError) as e:
        assert_tensor_cache_alignment("all_subjects_tensors.pt", {})
    msg = str(e.value)
    assert "force_rebuild" in msg
    assert "bigG" in msg
    assert "all_subjects_tensors.pt" in msg


def test_every_cache_reader_goes_through_the_guard():
    # train_step1b loads the cache with a bare torch.load rather than through
    # build_or_load_tensors, so a guard in the builder alone protects nothing.
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    offenders = []
    for p in (root / "scripts").glob("*.py"):
        src = p.read_text(encoding="utf-8")
        if "args.tensor_cache" not in src or "torch.load" not in src:
            continue
        for line in src.splitlines():
            if "torch.load" in line and "tensor_cache" in line:
                if "assert_tensor_cache_alignment" not in line:
                    offenders.append(f"{p.name}: {line.strip()}")
    assert not offenders, "unguarded tensor-cache loads:\n" + "\n".join(offenders)
