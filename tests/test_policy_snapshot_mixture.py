from __future__ import annotations

import numpy as np

from build_policy_snapshot_mixture import concatenate_parts, load_parts


def make_part(group_count: int, period: int, version: float | None = None):
    part = {
        "features": np.zeros((group_count, 2, 3), dtype=np.float32),
        "rewards": np.zeros((group_count, 2), dtype=np.float32),
        "costs": np.zeros((group_count, 2), dtype=np.float32),
        "scores": np.zeros((group_count, 2), dtype=np.float32),
        "periods": np.full(group_count, period, dtype=np.int64),
        "groups": np.arange(group_count, dtype=np.int64),
    }
    if version is not None:
        part["policy_versions"] = np.full(group_count, version, dtype=np.float32)
    return part


def test_load_parts_assigns_default_policy_version(tmp_path):
    path = tmp_path / "base.npz"
    np.savez_compressed(path, **make_part(3, 24))
    loaded = load_parts([str(path)], default_version=0.0)
    np.testing.assert_array_equal(loaded["policy_versions"], np.zeros(3))


def test_concatenate_parts_reindexes_groups_and_preserves_versions():
    recent = make_part(2, 24, version=1.0)
    base = make_part(1, 24, version=0.0)
    mixed = concatenate_parts([recent, base])
    np.testing.assert_array_equal(mixed["groups"], np.arange(3))
    np.testing.assert_array_equal(mixed["policy_versions"], [1.0, 1.0, 0.0])
