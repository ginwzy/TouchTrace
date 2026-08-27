"""Autoregressive touch swipe generation from touch.onnx."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

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
DEFAULT_SNAP_PX = 15.0
DEFAULT_AVG_STEP_PX = 13.0


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
    snap_px: float = DEFAULT_SNAP_PX
    avg_step_px: float = DEFAULT_AVG_STEP_PX
    guided: bool = True
    smooth: bool = True
    smooth_window: int = 7
    seed: int | None = None


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


def smooth_path(path: list[TouchStep], window_size: int = 7) -> list[TouchStep]:
    if len(path) < window_size:
        return path
    half = window_size // 2
    smoothed = [path[0]]
    for i in range(1, len(path) - 1):
        start = max(0, i - half)
        end = min(len(path), i + half + 1)
        window = path[start:end]
        smoothed.append(
            TouchStep(
                x=round(sum(p.x for p in window) / len(window)),
                y=round(sum(p.y for p in window) / len(window)),
                t=path[i].t,
            )
        )
    smoothed.append(path[-1])
    return smoothed


def _guided_step(
    cx: float,
    cy: float,
    end: tuple[float, float],
    dx: float,
    dy: float,
    *,
    min_step_px: float,
    max_step_px: float,
    snap_px: float,
) -> tuple[float, float, bool]:
    dist = math.hypot(end[0] - cx, end[1] - cy)
    if dist < snap_px:
        return end[0] - cx, end[1] - cy, True

    tx = (end[0] - cx) / dist
    ty = (end[1] - cy) / dist
    mag = math.hypot(dx, dy)
    dot = dx * tx + dy * ty

    if dot < 0 or mag < min_step_px:
        mag = min(max(min_step_px, mag), max_step_px, dist * 0.35)
        dx, dy = tx * mag, ty * mag
    elif mag > max(dist, max_step_px):
        step = min(dist, max_step_px)
        dx, dy = tx * step, ty * step

    return dx, dy, False


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
        if dist < 3.0:
            break

        sequence.append([dx_prev, dy_prev, dt_prev, end[0] - cx, end[1] - cy])
        arr = np.array(sequence, dtype=np.float32).reshape(1, len(sequence), INPUT_DIMS)
        params = session.run([output_name], {input_name: arr})[0][0, -1, :PARAMS_SIZE]
        dx, dy, dt = sample_from_mdn(params, rng)
        dt_step = dt if dt > 0 else MIN_DELAY_MS

        if opts.guided:
            dx, dy, snapped = _guided_step(
                cx,
                cy,
                end,
                dx,
                dy,
                min_step_px=opts.min_step_px,
                max_step_px=opts.max_step_px,
                snap_px=opts.snap_px,
            )
            if snapped:
                cx, cy = end[0], end[1]
                elapsed_ms += max(dt_step, MIN_DELAY_MS)
                path.append(TouchStep(x=round(cx), y=round(cy), t=elapsed_ms))
                break
        else:
            snapped = False

        cx += dx
        cy += dy
        elapsed_ms += dt_step
        path.append(TouchStep(x=round(cx), y=round(cy), t=elapsed_ms))
        dx_prev, dy_prev, dt_prev = dx, dy, dt_step
        last_dt = dt_step

    elapsed_ms += last_dt
    path.append(TouchStep(x=round(end[0]), y=round(end[1]), t=elapsed_ms))

    if opts.smooth:
        path = smooth_path(path, opts.smooth_window)
    return path


def load_session(model_path: str | Path) -> ort.InferenceSession:
    import onnxruntime as ort

    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
