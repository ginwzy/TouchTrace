"""Deterministic user-grouped splits shared by sensor training and evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PARTITIONS = ("train", "validation", "test")


def build_user_split(
    user_ids,
    *,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> dict:
    users = sorted({str(user_id) for user_id in user_ids if str(user_id).strip()})
    if len(users) < 3:
        raise ValueError("user-grouped split requires at least three non-empty user IDs")
    if validation_fraction <= 0 or test_fraction <= 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("validation/test fractions must be positive and sum to less than one")
    order = np.random.default_rng(seed).permutation(len(users))
    n_validation = max(1, int(round(len(users) * validation_fraction)))
    n_test = max(1, int(round(len(users) * test_fraction)))
    if n_validation + n_test >= len(users):
        raise ValueError("validation/test fractions leave no training users")
    shuffled = [users[int(index)] for index in order]
    test_users = sorted(shuffled[:n_test])
    validation_users = sorted(shuffled[n_test : n_test + n_validation])
    train_users = sorted(shuffled[n_test + n_validation :])
    return {
        "strategy": "user_id",
        "seed": int(seed),
        "validation_fraction": float(validation_fraction),
        "test_fraction": float(test_fraction),
        "train_users": train_users,
        "validation_users": validation_users,
        "test_users": test_users,
    }


def validate_user_split(split: dict) -> dict:
    if split.get("strategy") != "user_id":
        raise ValueError("split manifest strategy must be user_id")
    groups = {partition: {str(value) for value in split.get(f"{partition}_users", [])} for partition in PARTITIONS}
    if any(not users for users in groups.values()):
        raise ValueError("split manifest must contain non-empty train/validation/test users")
    for index, left in enumerate(PARTITIONS):
        for right in PARTITIONS[index + 1 :]:
            overlap = groups[left] & groups[right]
            if overlap:
                raise ValueError(f"split manifest has users in both {left} and {right}: {sorted(overlap)}")
    return split


def partition_indices(user_ids, split: dict) -> dict[str, np.ndarray]:
    validate_user_split(split)
    values = np.asarray([str(user_id) for user_id in user_ids], dtype=object)
    result = {}
    assigned = np.zeros(len(values), dtype=bool)
    for partition in PARTITIONS:
        users = set(split[f"{partition}_users"])
        mask = np.asarray([value in users for value in values], dtype=bool)
        result[partition] = np.flatnonzero(mask)
        assigned |= mask
    if not assigned.all():
        missing = sorted(set(values[~assigned].tolist()))
        raise ValueError(f"split manifest does not assign users: {missing}")
    return result


def rows_for_partition(rows: list[dict], split: dict, partition: str) -> list[dict]:
    validate_user_split(split)
    if partition not in PARTITIONS:
        raise ValueError(f"partition must be one of {PARTITIONS}")
    users = set(split[f"{partition}_users"])
    return [row for row in rows if str((row.get("meta") or {}).get("user_id") or "") in users]


def write_user_split(path: str | Path, split: dict) -> Path:
    validate_user_split(split)
    output = Path(path)
    output.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    return output


def load_user_split(path: str | Path) -> dict:
    return validate_user_split(json.loads(Path(path).read_text(encoding="utf-8")))
