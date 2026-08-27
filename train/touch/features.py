"""Encode swipe jsonl into (dx, dy, dt, remaining-distance) training sequences."""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np

DEFAULT_MAX_STEPS = 128


def encode_trajectory(path: list[dict], target: dict) -> tuple[list[list[float]], list[list[float]]]:
    target_x = float(target["x"])
    target_y = float(target["y"])
    dx_prev = 0.0
    dy_prev = 0.0
    dt_prev = 0.0
    X: list[list[float]] = []
    Y: list[list[float]] = []

    for i in range(1, len(path)):
        curr = path[i]
        prev = path[i - 1]
        dx_curr = float(curr["x"]) - float(prev["x"])
        dy_curr = float(curr["y"]) - float(prev["y"])
        dt_curr = float(curr["timestamp"]) - float(prev["timestamp"])
        dist_x = target_x - float(prev["x"])
        dist_y = target_y - float(prev["y"])
        X.append([dx_prev, dy_prev, dt_prev, dist_x, dist_y])
        Y.append([dx_curr, dy_curr, dt_curr])
        dx_prev = dx_curr
        dy_prev = dy_curr
        dt_prev = dt_curr

    return X, Y


def _open_jsonl(filepath: str | Path):
    path = Path(filepath)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open()


def iter_encoded_trajectories(
    filepath: str | Path, max_steps: int | None = DEFAULT_MAX_STEPS
) -> Iterator[tuple[list[list[float]], list[list[float]]]]:
    with _open_jsonl(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            x_seq, y_seq = encode_trajectory(data.get("path", []), data["target"])
            if max_steps is not None:
                x_seq = x_seq[:max_steps]
                y_seq = y_seq[:max_steps]
            if not x_seq:
                continue
            yield x_seq, y_seq


def load_trajectory_sequences(
    filepath: str | Path, max_steps: int | None = DEFAULT_MAX_STEPS
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for x_seq, y_seq in iter_encoded_trajectories(filepath, max_steps=max_steps):
        xs.append(np.asarray(x_seq, dtype=np.float32))
        ys.append(np.asarray(y_seq, dtype=np.float32))
    return xs, ys


def load_trajectory_jsonl(
    filepath: str | Path,
    pad: float = -999999.0,
    max_steps: int | None = DEFAULT_MAX_STEPS,
) -> tuple[np.ndarray, np.ndarray]:
    X_all: list[list[list[float]]] = []
    Y_all: list[list[list[float]]] = []
    for x_seq, y_seq in iter_encoded_trajectories(filepath, max_steps=max_steps):
        X_all.append(x_seq)
        Y_all.append(y_seq)
    return _pad(X_all, pad), _pad(Y_all, pad)


def _pad(sequences: list[list[list[float]]], pad: float) -> np.ndarray:
    if not sequences:
        return np.zeros((0, 0, 0), dtype=np.float32)
    max_len = max(len(seq) for seq in sequences)
    dims = len(sequences[0][0])
    out = np.full((len(sequences), max_len, dims), pad, dtype=np.float32)
    for i, seq in enumerate(sequences):
        arr = np.asarray(seq, dtype=np.float32)
        out[i, : len(seq)] = arr
    return out
