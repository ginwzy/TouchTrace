import json

import numpy as np
import pytest

from sensor.split import (
    build_user_split,
    load_user_split,
    partition_indices,
    rows_for_partition,
    validate_user_split,
    write_user_split,
)


def test_build_user_split_creates_deterministic_40_5_5_groups():
    users = [str(index) for index in range(50) for _ in range(3)]
    first = build_user_split(users, seed=42)
    second = build_user_split(users, seed=42)
    assert first == second
    assert len(first["train_users"]) == 40
    assert len(first["validation_users"]) == 5
    assert len(first["test_users"]) == 5
    assert not (set(first["train_users"]) & set(first["validation_users"]))
    assert not (set(first["train_users"]) & set(first["test_users"]))
    assert not (set(first["validation_users"]) & set(first["test_users"]))


def test_partition_indices_assigns_every_row_without_overlap():
    users = np.asarray([str(index) for index in range(10) for _ in range(2)])
    split = build_user_split(users, validation_fraction=0.2, test_fraction=0.2, seed=7)
    indices = partition_indices(users, split)
    combined = np.concatenate(list(indices.values()))
    assert sorted(combined.tolist()) == list(range(len(users)))
    assert len(set(combined.tolist())) == len(users)


def test_rows_for_partition_uses_manifest_user_ids():
    rows = [{"meta": {"user_id": str(index)}, "value": index} for index in range(10)]
    split = build_user_split([str(index) for index in range(10)], validation_fraction=0.2, test_fraction=0.2)
    picked = rows_for_partition(rows, split, "test")
    assert {row["meta"]["user_id"] for row in picked} == set(split["test_users"])


def test_split_manifest_round_trip_and_overlap_rejection(tmp_path):
    split = build_user_split([str(index) for index in range(10)], validation_fraction=0.2, test_fraction=0.2)
    path = write_user_split(tmp_path / "split.json", split)
    assert load_user_split(path) == split

    invalid = json.loads(path.read_text())
    invalid["test_users"].append(invalid["train_users"][0])
    with pytest.raises(ValueError, match="both train and test"):
        validate_user_split(invalid)
