import pytest

from rxfm.tensor_cache import (BEHAV_TO_BETAS_OFFSET, TENSOR_CACHE_FORMAT,
                                    assert_tensor_cache_alignment,
                                    tensor_cache_meta)


def test_offset_is_zero():
    assert BEHAV_TO_BETAS_OFFSET == 0


def test_unstamped_cache_is_refused():
    with pytest.raises(RuntimeError, match="predates the current split contract"):
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
    with pytest.raises(RuntimeError) as e:
        assert_tensor_cache_alignment("all_subjects_tensors.pt", {})
    msg = str(e.value)
    assert "force_rebuild" in msg
    assert "bigG" in msg
    assert "all_subjects_tensors.pt" in msg


def test_every_cache_reader_goes_through_the_guard():
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


def test_splits_are_not_capped_by_default():
    from rxfm.config import Config
    cfg = Config()
    assert cfg.max_train is None
    assert cfg.max_test is None


def test_nsd_split_constants():
    from rxfm.tensor_cache import NSD_TEST_IMAGES, NSD_REPEATS_PER_IMAGE
    assert NSD_TEST_IMAGES == 982
    assert NSD_REPEATS_PER_IMAGE == 3


def test_a_v3_cache_is_now_refused_too():
    with pytest.raises(RuntimeError, match="552"):
        assert_tensor_cache_alignment("c.pt", {"_meta": {"format": "tensors_v3_offset0"}})
