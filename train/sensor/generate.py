"""Autoregressive IMU generation along a frozen touch path."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

HERE = Path(__file__).resolve().parent
_TRAIN_ROOT = HERE.parent
if str(_TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAIN_ROOT))

from sensor.config import model_config
from sensor.features import IMU_DIMS, pack_sensor_step, require_delta_norm
from touch.features import resolve_jsonl

if TYPE_CHECKING:
    import onnxruntime as ort

COMPONENTS = int(model_config["components"])
INPUT_DIMS = int(model_config["input_dims"])
OUTPUT_DIMS = int(model_config["output_dims"])
PARAMS_SIZE = COMPONENTS * (1 + 2 * OUTPUT_DIMS)


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[-1]


def default_onnx_path() -> Path:
    name = model_config["onnx_model"]
    return _first_existing(HERE / name, HERE.parent.parent / "inference" / "public" / name)


def default_norm_path() -> Path:
    name = model_config["norm"]
    return _first_existing(HERE / name, HERE.parent.parent / "inference" / "public" / name)


def default_data_path() -> Path:
    return resolve_jsonl(HERE, "sensor_data")


def load_norm(path: str | Path | None = None) -> dict:
    norm_path = Path(path) if path else default_norm_path()
    if not norm_path.exists():
        raise SystemExit(f"Missing {norm_path}")
    try:
        return require_delta_norm(json.loads(norm_path.read_text()))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def load_session(model_path: str | Path) -> ort.InferenceSession:
    import onnxruntime as ort

    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def sample_imu_mdn(params: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    logits = params[:COMPONENTS]
    u = rng.random(COMPONENTS)
    gumbel = logits - np.log(-np.log(np.clip(u, 1e-12, 1.0)))
    best = int(np.argmax(gumbel))
    offset = COMPONENTS + best * (2 * OUTPUT_DIMS)
    means = params[offset : offset + OUTPUT_DIMS]
    raw_scale = params[offset + OUTPUT_DIMS : offset + 2 * OUTPUT_DIMS]
    scales = np.where(raw_scale > 20, raw_scale, np.log1p(np.exp(np.minimum(raw_scale, 20.0))))
    return rng.normal(means, scales)


def mixture_mean(params: np.ndarray) -> np.ndarray:
    """E[x] of the 6-d mixture (softmax over component means)."""
    logits = np.asarray(params[:COMPONENTS], dtype=np.float64)
    logits = logits - logits.max()
    weights = np.exp(logits)
    weights = weights / weights.sum()
    means = np.asarray(params[COMPONENTS:], dtype=np.float64).reshape(COMPONENTS, 2 * OUTPUT_DIMS)[:, :OUTPUT_DIMS]
    return weights @ means


def mix_mdn_draw(mean: np.ndarray, sample: np.ndarray, temp: float) -> np.ndarray:
    if temp <= 0:
        return mean
    if temp >= 1:
        return sample
    return mean + float(temp) * (sample - mean)


def decode_imu_mdn(params: np.ndarray, rng: np.random.Generator, temp: float = 1.0) -> np.ndarray:
    """temp=0 is the mixture mean; temp=1 is a full MDN draw; in between interpolates."""
    center = mixture_mean(params)
    if temp <= 0:
        return center
    return mix_mdn_draw(center, sample_imu_mdn(params, rng), temp)


def decode_correlated_imu_mdn(
    params: np.ndarray,
    rng: np.random.Generator,
    temp: float,
    rho: float,
    previous_innovation: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Decode an MDN draw while optionally correlating only its innovation."""
    if not 0.0 <= rho < 1.0:
        raise ValueError("innovation_rho must be in [0, 1)")
    center = mixture_mean(params)
    if temp <= 0:
        return center, previous_innovation
    innovation = sample_imu_mdn(params, rng) - center
    if rho > 0 and previous_innovation is not None:
        innovation = rho * previous_innovation + math.sqrt(1.0 - rho * rho) * innovation
    return center + float(temp) * innovation, innovation


def sensor_mags(sensors: list[dict]) -> tuple[list[float], list[float], list[float]]:
    ts, acc, gyro = [], [], []
    for s in sensors:
        a, g = s["accel"], s["gyro"]
        ts.append(float(s["timestamp"]))
        acc.append(math.hypot(a[0], a[1], a[2]))
        gyro.append(math.hypot(g[0], g[1], g[2]))
    return ts, acc, gyro


def generate_sensor_along_path(
    session: ort.InferenceSession,
    path: list[dict],
    imu0: list[float],
    condition: str | None,
    norm: dict,
    rng: np.random.Generator,
    remaining_frame: bool | None = None,
    *,
    temp: float | None = None,
    teacher_sensors: list[dict] | None = None,
    innovation_rho: float = 0.0,
) -> list[dict]:
    """Roll out accel+gyro on a fixed touch path, starting from imu0.

    The ONNX head predicts z-scored ΔIMU; this integrates onto the previous
    absolute sample. temp: 0 = mixture mean, 1 = full MDN sample
    (default: model_config['mdn_temp']). innovation_rho correlates only the
    stochastic MDN residual; 0 preserves independent per-step draws.
    teacher_sensors: if set, each step is conditioned on the real previous IMU.
    """
    if len(path) < 2:
        return [_sensor_point(path[0]["timestamp"], imu0)] if path else []
    if teacher_sensors is not None and len(teacher_sensors) != len(path):
        raise ValueError("teacher_sensors must align with path")
    remaining_frame = model_config["remaining_frame"] if remaining_frame is None else remaining_frame
    temp = float(model_config["mdn_temp"] if temp is None else temp)
    innovation_rho = float(innovation_rho)
    if not 0.0 <= innovation_rho < 1.0:
        raise ValueError("innovation_rho must be in [0, 1)")
    mean = np.asarray(norm["mean"], dtype=np.float64)
    std = np.asarray(norm["std"], dtype=np.float64)
    d_mean = np.asarray(norm["delta_mean"], dtype=np.float64)
    d_std = np.asarray(norm["delta_std"], dtype=np.float64)
    target = path[-1]
    pred = np.asarray(imu0, dtype=np.float64)
    out = [_sensor_point(path[0]["timestamp"], pred.tolist())]
    sequence: list[list[float]] = []
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    innovation = None

    for i, (prev, curr) in enumerate(zip(path, path[1:])):
        if teacher_sensors is not None:
            s = teacher_sensors[i]
            prev_imu = np.asarray(s["accel"] + s["gyro"], dtype=np.float64)
        else:
            prev_imu = pred
        z_prev = ((prev_imu - mean) / std).tolist()
        sequence.append(pack_sensor_step(z_prev, prev, curr, target, condition, remaining_frame))
        arr = np.asarray(sequence, dtype=np.float32).reshape(1, len(sequence), INPUT_DIMS)
        params = session.run([output_name], {input_name: arr})[0][0, -1, :PARAMS_SIZE]
        delta_z, innovation = decode_correlated_imu_mdn(
            params,
            rng,
            temp,
            innovation_rho,
            innovation,
        )
        delta = delta_z * d_std + d_mean
        pred = prev_imu + delta
        out.append(_sensor_point(curr["timestamp"], pred.tolist()))
    return out


def _sensor_point(timestamp: float, imu: list[float]) -> dict:
    return {
        "timestamp": float(timestamp),
        "accel": [float(v) for v in imu[:3]],
        "gyro": [float(v) for v in imu[3:IMU_DIMS]],
    }
