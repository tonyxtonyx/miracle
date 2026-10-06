import pytest
from miracle.selection import SubsetSpec, select

INST = [{"instance_id": f"{r}-{i}", "repo": r, "difficulty": "d1" if i % 2 else "d2"}
        for r in ("a/a", "b/b") for i in range(10)]


def test_sampling_is_deterministic_and_order_independent():
    s = SubsetSpec(n=5, seed=3)
    assert select(INST, s) == select(list(reversed(INST)), s)
    assert select(INST, s) != select(INST, SubsetSpec(n=5, seed=4))


def test_filters_and_ids():
    assert all(i.startswith("a/a") for i in select(INST, SubsetSpec(repos=["a/a"])))
    assert select(INST, SubsetSpec(ids=["b/b-2", "a/a-1"])) == ["b/b-2", "a/a-1"]   # explicit order kept
    assert "a/a-1" not in select(INST, SubsetSpec(exclude=["a/a-1"]))
    with pytest.raises(KeyError):
        select(INST, SubsetSpec(ids=["nope"]))
