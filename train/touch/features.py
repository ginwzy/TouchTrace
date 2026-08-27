"""Encode swipe jsonl into (dx, dy, dt, remaining-distance) training sequences."""

from __future__ import annotations

import gzip
import json
import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np

DEFAULT_MAX_STEPS = 128
ANGLE_H_DEG = 20.0
ANGLE_V_DEG = 60.0

# D4: identity, 90/180/270 rotations, and four reflections.
GEOM_TRANSFORMS = (
    lambda x, y: (x, y),
    lambda x, y: (-y, x),
    lambda x, y: (-x, -y),
    lambda x, y: (y, -x),
    lambda x, y: (-x, y),
    lambda x, y: (x, -y),
    lambda x, y: (y, x),
    lambda x, y: (-y, -x),
)


def swipe_angle_bucket(
    dx: float,
    dy: float,
    h_deg: float = ANGLE_H_DEG,
    v_deg: float = ANGLE_V_DEG,
) -> str:
    """Classify a swipe as H / D / V from its remaining-distance vector."""
    ang = math.degrees(math.atan2(abs(dy), abs(dx)))
    if ang < h_deg:
        return "H"
    if ang > v_deg:
        return "V"
    return "D"


def angle_multipliers(buckets: list[str], max_mult: float = 8.0) -> np.ndarray:
    """Inverse-frequency weights so H/D/V contribute equally, clipped at max_mult."""
    if not buckets:
        return np.zeros(0, dtype=np.float32)
    counts: dict[str, int] = {}
    for b in buckets:
        counts[b] = counts.get(b, 0) + 1
    n = len(buckets)
    n_classes = max(len(counts), 1)
    out = np.empty(n, dtype=np.float32)
    for i, b in enumerate(buckets):
        out[i] = min(max_mult, n / (n_classes * counts[b]))
    return out


def scale_sample_weights_by_angle(W: np.ndarray, X: np.ndarray, max_mult: float = 8.0) -> np.ndarray:
    buckets = [swipe_angle_bucket(float(X[i, 0, 3]), float(X[i, 0, 4])) for i in range(len(X))]
    return (W * angle_multipliers(buckets, max_mult)[:, None]).astype(np.float32)


def apply_geom_transform(x: np.ndarray, y: np.ndarray, fn) -> tuple[np.ndarray, np.ndarray]:
    """Apply a 2D linear map to spatial channels of one encoded sequence."""
    x = np.array(x, dtype=np.float32, copy=True)
    y = np.array(y, dtype=np.float32, copy=True)
    dx, dy = x[..., 0].copy(), x[..., 1].copy()
    x[..., 0], x[..., 1] = fn(dx, dy)
    rx, ry = x[..., 3].copy(), x[..., 4].copy()
    x[..., 3], x[..., 4] = fn(rx, ry)
    ox, oy = y[..., 0].copy(), y[..., 1].copy()
    y[..., 0], y[..., 1] = fn(ox, oy)
    return x, y


def subsample_path(path: list[dict], min_step_px: float = 0.0) -> list[dict]:
    """Keep endpoints; drop intermediate points until movement >= min_step_px."""
    if min_step_px <= 0 or len(path) < 2:
        return path

    out = [path[0]]
    anchor = path[0]
    ax, ay = float(anchor["x"]), float(anchor["y"])

    for pt in path[1:]:
        px, py = float(pt["x"]), float(pt["y"])
        if math.hypot(px - ax, py - ay) >= min_step_px:
            out.append(pt)
            anchor = pt
            ax, ay = px, py

    if out[-1] is not path[-1]:
        out.append(path[-1])
    return out


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


def compute_sample_weights(
    X: np.ndarray,
    Y: np.ndarray,
    pad: float,
    step_weight: float,
    dist_weight: float,
) -> np.ndarray:
    """Per-timestep loss weights: emphasize larger steps and far-from-target states."""
    mask = Y[:, :, 0] != pad
    step_mag = np.sqrt(Y[:, :, 0] ** 2 + Y[:, :, 1] ** 2)
    dist = np.sqrt(X[:, :, 3] ** 2 + X[:, :, 4] ** 2)
    weights = 1.0 + step_weight * step_mag + dist_weight * dist
    return (weights * mask).astype(np.float32)


def _open_jsonl(filepath: str | Path):
    path = Path(filepath)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open()


def iter_encoded_trajectories(
    filepath: str | Path,
    max_steps: int | None = DEFAULT_MAX_STEPS,
    min_step_px: float = 0.0,
) -> Iterator[tuple[list[list[float]], list[list[float]]]]:
    with _open_jsonl(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            path = subsample_path(data.get("path", []), min_step_px=min_step_px)
            if len(path) < 2:
                continue
            x_seq, y_seq = encode_trajectory(path, data["target"])
            if max_steps is not None:
                x_seq = x_seq[:max_steps]
                y_seq = y_seq[:max_steps]
            if not x_seq:
                continue
            yield x_seq, y_seq


def load_trajectory_sequences(
    filepath: str | Path,
    max_steps: int | None = DEFAULT_MAX_STEPS,
    min_step_px: float = 0.0,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for x_seq, y_seq in iter_encoded_trajectories(
        filepath, max_steps=max_steps, min_step_px=min_step_px
    ):
        xs.append(np.asarray(x_seq, dtype=np.float32))
        ys.append(np.asarray(y_seq, dtype=np.float32))
    return xs, ys


def load_trajectory_jsonl(
    filepath: str | Path,
    pad: float = -999999.0,
    max_steps: int | None = DEFAULT_MAX_STEPS,
    min_step_px: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    X_all: list[list[list[float]]] = []
    Y_all: list[list[list[float]]] = []
    for x_seq, y_seq in iter_encoded_trajectories(
        filepath, max_steps=max_steps, min_step_px=min_step_px
    ):
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


def summarize_encoded_lengths(filepath: str | Path, min_step_px: float = 0.0) -> dict[str, float]:
    lengths = [len(y) for _, y in iter_encoded_trajectories(filepath, min_step_px=min_step_px)]
    arr = np.array(lengths, dtype=np.float32)
    return {
        "count": float(len(arr)),
        "len_min": float(arr.min()) if len(arr) else 0.0,
        "len_mean": float(arr.mean()) if len(arr) else 0.0,
        "len_max": float(arr.max()) if len(arr) else 0.0,
    }
