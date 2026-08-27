"""Autoregressive touch swipe generation from touch.onnx."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from features import from_remaining_frame, remaining_frame_axes, to_remaining_frame

if TYPE_CHECKING:
    import onnxruntime as ort

COMPONENTS = 5
INPUT_DIMS = 5
OUTPUT_DIMS = 3
MIN_DELAY_MS = 2.0
PARAMS_SIZE = COMPONENTS * (1 + 2 * OUTPUT_DIMS)

# Training-set step stats (CSD4CA): median step ~5px, p90 ~32px.
DEFAULT_MIN_STEP_PX = 3.0
DEFAULT_MAX_STEP_PX = 35.0
DEFAULT_AVG_STEP_PX = 13.0
# Loop stop / optional target append. Distinct from min_step_px (training subsample).
ARRIVE_PX = 3.0


@dataclass(frozen=True)
class TouchStep:
    x: int
    y: int
    t: float


@dataclass
class GenerateOptions:
    max_steps: int = 500
    min_step_px: float = DEFAULT_MIN_STEP_PX
    max_step_px: float = DEFAULT_MAX_STEP_PX
    avg_step_px: float = DEFAULT_AVG_STEP_PX
    no_backtrack: bool = True
    seed: int | None = None
    remaining_frame: bool = True


def inference_public_onnx(name: str = "touch.onnx") -> Path:
    return Path(__file__).resolve().parent.parent.parent / "inference" / "public" / name


def default_onnx_path() -> Path:
    here = Path(__file__).resolve().parent
    candidates = (here / "touch.onnx", inference_public_onnx())
    for path in candidates:
        if path.exists():
            return path
    return candidates[-1]


def default_data_path() -> Path:
    here = Path(__file__).resolve().parent
    gz = here / "touch_data.jsonl.gz"
    return gz if gz.exists() else here / "touch_data.jsonl"


def softplus(x: float) -> float:
    return x if x > 20 else math.log1p(math.exp(x))


def sample_from_mdn(params: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    logits = params[:COMPONENTS]
    u = rng.random(COMPONENTS)
    gumbel = logits - np.log(-np.log(u))
    best = int(np.argmax(gumbel))
    offset = COMPONENTS + best * (2 * OUTPUT_DIMS)
    means = params[offset : offset + OUTPUT_DIMS]
    scales = np.array([softplus(v) for v in params[offset + OUTPUT_DIMS : offset + 2 * OUTPUT_DIMS]])
    sample = rng.normal(means, scales)
    return float(sample[0]), float(sample[1]), float(sample[2])


def _no_backtrack_step(
    cx: float,
    cy: float,
    end: tuple[float, float],
    dx: float,
    dy: float,
    min_step_px: float,
) -> tuple[float, float]:
    """If the sampled step points away from the target, redirect it along the remaining vector."""
    c, s, dist = remaining_frame_axes(end[0] - cx, end[1] - cy)
    if dist < 1.0:
        return dx, dy
    if dx * c + dy * s >= 0:
        return dx, dy
    mag = max(math.hypot(dx, dy), min_step_px)
    return c * mag, s * mag


def _clamp_step_mag(dx: float, dy: float, max_step_px: float) -> tuple[float, float]:
    """Cap a step to max_step_px, keeping its direction."""
    mag = math.hypot(dx, dy)
    if mag > max_step_px > 0:
        scale = max_step_px / mag
        return dx * scale, dy * scale
    return dx, dy


def generate_touch_path(
    session: ort.InferenceSession,
    start: tuple[float, float],
    end: tuple[float, float],
    options: GenerateOptions | None = None,
    rng: np.random.Generator | None = None,
) -> list[TouchStep]:
    opts = options or GenerateOptions()
    rng = rng if rng is not None else np.random.default_rng(opts.seed)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    cx, cy = start
    dx_prev = dy_prev = dt_prev = 0.0
    elapsed_ms = 0.0
    last_dt = MIN_DELAY_MS
    sequence: list[list[float]] = []
    path = [TouchStep(x=round(cx), y=round(cy), t=elapsed_ms)]

    dist0 = math.hypot(end[0] - start[0], end[1] - start[1])
    step_budget = min(opts.max_steps, max(20, int(dist0 / opts.avg_step_px * 3)))

    for _ in range(step_budget):
        dist = math.hypot(end[0] - cx, end[1] - cy)
        if dist < ARRIVE_PX:
            break

        rem_x = end[0] - cx
        rem_y = end[1] - cy
        if opts.remaining_frame:
            tan_prev, nrm_prev, rem = to_remaining_frame(dx_prev, dy_prev, rem_x, rem_y)
            sequence.append([tan_prev, nrm_prev, dt_prev, rem, 0.0])
        else:
            sequence.append([dx_prev, dy_prev, dt_prev, rem_x, rem_y])
        arr = np.array(sequence, dtype=np.float32).reshape(1, len(sequence), INPUT_DIMS)
        params = session.run([output_name], {input_name: arr})[0][0, -1, :PARAMS_SIZE]
        d0, d1, dt = sample_from_mdn(params, rng)
        if opts.remaining_frame:
            dx, dy = from_remaining_frame(d0, d1, rem_x, rem_y)
        else:
            dx, dy = d0, d1
        dt_step = dt if dt > 0 else MIN_DELAY_MS

        if opts.no_backtrack:
            dx, dy = _no_backtrack_step(cx, cy, end, dx, dy, opts.min_step_px)
        dx, dy = _clamp_step_mag(dx, dy, opts.max_step_px)

        cx += dx
        cy += dy
        elapsed_ms += dt_step
        path.append(TouchStep(x=round(cx), y=round(cy), t=elapsed_ms))
        dx_prev, dy_prev, dt_prev = dx, dy, dt_step
        last_dt = dt_step

    if math.hypot(path[-1].x - end[0], path[-1].y - end[1]) >= ARRIVE_PX:
        elapsed_ms += last_dt
        path.append(TouchStep(x=round(end[0]), y=round(end[1]), t=elapsed_ms))

    return path


def load_session(model_path: str | Path) -> ort.InferenceSession:
    import onnxruntime as ort

    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
