"""Encode fused touch+IMU jsonl into (imu_prev, kinematics, condition) sequences."""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from touch.features import (
    DEFAULT_MAX_STEPS,
    _pad,
    iter_jsonl,
    subsample_keep_indices,
    to_remaining_frame,
)

CONDITIONS = ("seated", "walking", "stress")
IMU_DIMS = 6


def condition_onehot(condition: str | None) -> list[float]:
    vec = [0.0] * len(CONDITIONS)
    if condition in CONDITIONS:
        vec[CONDITIONS.index(condition)] = 1.0
    return vec


def subsample_paired(
    path: list[dict],
    sensors: list[dict],
    min_step_px: float = 0.0,
) -> tuple[list[dict], list[dict]]:
    keep = subsample_keep_indices(path, min_step_px)
    return [path[i] for i in keep], [sensors[i] for i in keep]


def pack_sensor_step(
    imu_prev: list[float],
    prev: dict,
    curr: dict,
    target: dict,
    condition: str | None,
    remaining_frame: bool = True,
) -> list[float]:
    """13-d model input: prev IMU + remaining-frame kinematics + condition one-hot."""
    dx = float(curr["x"]) - float(prev["x"])
    dy = float(curr["y"]) - float(prev["y"])
    dt = float(curr["timestamp"]) - float(prev["timestamp"])
    rem_x = float(target["x"]) - float(prev["x"])
    rem_y = float(target["y"]) - float(prev["y"])
    if remaining_frame:
        tan, nrm, rem = to_remaining_frame(dx, dy, rem_x, rem_y)
        kin = [tan, nrm, dt, rem]
    else:
        kin = [dx, dy, dt, math.hypot(rem_x, rem_y)]
    return [*imu_prev, *kin, *condition_onehot(condition)]


def encode_sensor_trajectory(
    path: list[dict],
    sensors: list[dict],
    condition: str | None,
    remaining_frame: bool = True,
    target: dict | None = None,
) -> tuple[list[list[float]], list[list[float]]]:
    """Y is current − previous accel+gyro in device frame; z-score is applied later."""
    if len(path) != len(sensors) or len(path) < 2:
        return [], []
    tgt = target or path[-1]
    X: list[list[float]] = []
    Y: list[list[float]] = []
    for i in range(1, len(path)):
        imu_prev = _imu(sensors[i - 1])
        imu_curr = _imu(sensors[i])
        if imu_prev is None or imu_curr is None:
            return [], []
        X.append(pack_sensor_step(imu_prev, path[i - 1], path[i], tgt, condition, remaining_frame))
        Y.append([c - p for p, c in zip(imu_prev, imu_curr)])
    return X, Y


def iter_encoded_sensor_trajectories(
    filepath: str | Path,
    max_steps: int | None = DEFAULT_MAX_STEPS,
    min_step_px: float = 0.0,
    remaining_frame: bool = True,
) -> Iterator[tuple[list[list[float]], list[list[float]]]]:
    for data in iter_jsonl(filepath):
        path = data.get("path") or []
        sensors = data.get("sensors") or []
        if len(path) != len(sensors) or len(path) < 2:
            continue
        path, sensors = subsample_paired(path, sensors, min_step_px=min_step_px)
        if len(path) < 2:
            continue
        x_seq, y_seq = encode_sensor_trajectory(
            path,
            sensors,
            (data.get("meta") or {}).get("condition"),
            remaining_frame=remaining_frame,
            target=data.get("target"),
        )
        if max_steps is not None:
            x_seq = x_seq[:max_steps]
            y_seq = y_seq[:max_steps]
        if not x_seq:
            continue
        yield x_seq, y_seq


def load_sensor_sequences(
    filepath: str | Path,
    max_steps: int | None = DEFAULT_MAX_STEPS,
    min_step_px: float = 0.0,
    remaining_frame: bool = True,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for x_seq, y_seq in iter_encoded_sensor_trajectories(
        filepath, max_steps=max_steps, min_step_px=min_step_px, remaining_frame=remaining_frame
    ):
        xs.append(np.asarray(x_seq, dtype=np.float32))
        ys.append(np.asarray(y_seq, dtype=np.float32))
    return xs, ys


def load_sensor_jsonl(
    filepath: str | Path,
    pad: float = -999999.0,
    max_steps: int | None = DEFAULT_MAX_STEPS,
    min_step_px: float = 0.0,
    remaining_frame: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    X_all: list[list[list[float]]] = []
    Y_all: list[list[list[float]]] = []
    for x_seq, y_seq in iter_encoded_sensor_trajectories(
        filepath, max_steps=max_steps, min_step_px=min_step_px, remaining_frame=remaining_frame
    ):
        X_all.append(x_seq)
        Y_all.append(y_seq)
    return _pad(X_all, pad), _pad(Y_all, pad)


def collect_init_by_condition(filepath: str | Path) -> dict[str, dict[str, list[float]]]:
    """Mean/std of the first IMU sample per swipe, grouped by condition."""
    buckets: dict[str, list[list[float]]] = {c: [] for c in CONDITIONS}
    for data in iter_jsonl(filepath):
        sensors = data.get("sensors") or []
        if not sensors:
            continue
        imu = _imu(sensors[0])
        if imu is None:
            continue
        cond = (data.get("meta") or {}).get("condition")
        if cond in buckets:
            buckets[cond].append(imu)
    out: dict[str, dict[str, list[float]]] = {}
    for cond, rows in buckets.items():
        if not rows:
            continue
        arr = np.asarray(rows, dtype=np.float64)
        std = arr.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        out[cond] = {"mean": arr.mean(axis=0).tolist(), "std": std.tolist(), "n": int(len(rows))}
    return out


def _channel_mean_std(vals: np.ndarray, width: int) -> tuple[list[float], list[float]]:
    if vals.size == 0:
        return [0.0] * width, [1.0] * width
    mean = vals.mean(axis=0).astype(np.float64)
    std = vals.std(axis=0).astype(np.float64)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.tolist(), std.tolist()


def fit_sensor_norm(X: np.ndarray, Y: np.ndarray, pad: float) -> dict:
    """Fit absolute-IMU stats for X[..., :6] and ΔIMU stats for Y."""
    mask = Y[:, :, 0] != pad
    imu_mean, imu_std = _channel_mean_std(X[mask][..., :IMU_DIMS], IMU_DIMS)
    delta_mean, delta_std = _channel_mean_std(Y[mask], IMU_DIMS)
    return {
        "mean": imu_mean,
        "std": imu_std,
        "delta_mean": delta_mean,
        "delta_std": delta_std,
        "target": "delta",
        "axes": ["ax", "ay", "az", "gx", "gy", "gz"],
    }


def apply_sensor_norm(
    X: np.ndarray,
    Y: np.ndarray,
    norm: dict,
    pad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Z-score prev IMU on X[..., :6] and ΔIMU on Y; leave pad cells untouched."""
    imu_mean = np.asarray(norm["mean"], dtype=np.float32)
    imu_std = np.asarray(norm["std"], dtype=np.float32)
    d_mean = np.asarray(norm["delta_mean"], dtype=np.float32)
    d_std = np.asarray(norm["delta_std"], dtype=np.float32)
    mask = (Y[:, :, 0] != pad)[..., None]
    x = np.array(X, dtype=np.float32, copy=True)
    y = np.array(Y, dtype=np.float32, copy=True)
    x[..., :IMU_DIMS] = np.where(mask, (x[..., :IMU_DIMS] - imu_mean) / imu_std, x[..., :IMU_DIMS])
    y = np.where(mask, (y - d_mean) / d_std, y)
    return x, y


def abs_z_from_delta_z(prev_abs_z: np.ndarray, delta_z: np.ndarray, norm: dict) -> np.ndarray:
    """Integrate a z-scored ΔIMU onto a z-scored previous IMU."""
    mean = np.asarray(norm["mean"], dtype=np.float64)
    std = np.asarray(norm["std"], dtype=np.float64)
    d_mean = np.asarray(norm["delta_mean"], dtype=np.float64)
    d_std = np.asarray(norm["delta_std"], dtype=np.float64)
    prev = prev_abs_z.astype(np.float64) * std + mean
    delta = delta_z.astype(np.float64) * d_std + d_mean
    return ((prev + delta - mean) / std).astype(np.float32)


def require_delta_norm(norm: dict) -> dict:
    if norm.get("target") != "delta" or "delta_mean" not in norm or "delta_std" not in norm:
        raise ValueError(
            "sensor_norm.json is missing delta stats; retrain after the ΔIMU target change"
        )
    return norm


def _imu(sample: dict) -> list[float] | None:
    accel = sample.get("accel")
    gyro = sample.get("gyro")
    if not accel or not gyro or len(accel) != 3 or len(gyro) != 3:
        return None
    return [float(accel[0]), float(accel[1]), float(accel[2]), float(gyro[0]), float(gyro[1]), float(gyro[2])]
