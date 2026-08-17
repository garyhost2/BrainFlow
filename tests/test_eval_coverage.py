import pytest

from rxfm.dataset import SubjectSampler


def _batches(sampler):
    return list(iter(sampler))


def test_tail_is_kept_for_the_nsd_test_split():
    sampler = SubjectSampler([982], batch_size=32, shuffle=False, drop_last=False)
    batches = _batches(sampler)
    covered = [i for b in batches for i in b]
    assert len(covered) == 982
    assert sorted(covered) == list(range(982))
    assert len(batches[-1]) == 982 - (982 // 32) * 32 == 22


@pytest.mark.parametrize("lengths", [[982], [982, 982], [982, 800, 40], [7]])
@pytest.mark.parametrize("batch_size", [16, 32, 48])
def test_every_index_covered_exactly_once(lengths, batch_size):
    sampler = SubjectSampler(lengths, batch_size=batch_size, shuffle=False, drop_last=False)
    covered = [i for b in _batches(sampler) for i in b]
    assert sorted(covered) == list(range(sum(lengths)))


@pytest.mark.parametrize("lengths", [[982], [982, 800, 40], [7]])
@pytest.mark.parametrize("batch_size", [16, 32, 48])
@pytest.mark.parametrize("drop_last", [True, False])
def test_len_matches_what_iter_yields(lengths, batch_size, drop_last):
    sampler = SubjectSampler(lengths, batch_size=batch_size, shuffle=False, drop_last=drop_last)
    assert len(sampler) == len(_batches(sampler))


def test_batches_never_mix_subjects():
    lengths = [982, 800]
    sampler = SubjectSampler(lengths, batch_size=32, shuffle=True, drop_last=False)
    for b in _batches(sampler):
        assert all(i < lengths[0] for i in b) or all(i >= lengths[0] for i in b)


def test_drop_last_still_truncates_for_training():
    sampler = SubjectSampler([982], batch_size=32, shuffle=False, drop_last=True)
    covered = [i for b in _batches(sampler) for i in b]
    assert len(covered) == 960
