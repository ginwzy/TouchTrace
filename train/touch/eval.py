"""Evaluate touch.onnx generation quality, split by swipe angle."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_TRAIN_ROOT = HERE.parent
if str(_TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAIN_ROOT))

from touch.config import model_config
from touch.features import iter_jsonl, swipe_angle_bucket
from touch.generate import ARRIVE_PX, GenerateOptions, TouchStep, default_data_path, default_onnx_path, generate_touch_path, load_session

REACH_PX = ARRIVE_PX
CLOSE_PX = 15.0


@dataclass(frozen=True)
class _Pt:
    x: float
    y: float
    t: float


def _load_val_swipes(data_path: Path, val_fraction: float, seed: int = 42, limit: int | None = 200):
    rows = list(iter_jsonl(data_path))
    order = np.random.default_rng(seed).permutation(len(rows))
    n_val = max(1, int(len(rows) * val_fraction))
    val_rows = [rows[i] for i in order[:n_val]]
    if limit is not None:
        val_rows = val_rows[:limit]
    swipes = []
    for row in val_rows:
        path = row.get("path", [])
        if len(path) < 2:
            continue
        start = (float(path[0]["x"]), float(path[0]["y"]))
        end = (float(row["target"]["x"]), float(row["target"]["y"]))
        swipes.append((start, end, path))
    return swipes


def _as_steps(path: list[dict] | list[TouchStep]) -> list[_Pt]:
    if path and isinstance(path[0], TouchStep):
        return [_Pt(float(p.x), float(p.y), p.t) for p in path]
    t0 = float(path[0]["timestamp"]) if path else 0.0
    return [_Pt(float(p["x"]), float(p["y"]), float(p["timestamp"]) - t0) for p in path]


def _drop_trailing_on_target(path: list[_Pt], end: tuple[float, float]) -> list[_Pt]:
    if path and math.hypot(path[-1].x - end[0], path[-1].y - end[1]) < 1.0:
        return path[:-1]
    return path


def _interior(path: list[_Pt], end: tuple[float, float]) -> list[_Pt]:
    """Drop start and a trailing forced target snap so bow is measured on free motion."""
    pts = path[1:] if len(path) > 1 else path
    return _drop_trailing_on_target(pts, end)


def _model_end(path: list[_Pt], end: tuple[float, float]) -> _Pt:
    """Last free point: drop a trailing target snap only if the previous point was still far."""
    if len(path) < 2:
        return path[-1]
    last = path[-1]
    if math.hypot(last.x - end[0], last.y - end[1]) >= 1.0:
        return last
    prev = path[-2]
    if math.hypot(prev.x - end[0], prev.y - end[1]) >= REACH_PX:
        return prev
    return last


def _signed_perps(path: list[_Pt], start: tuple[float, float], end: tuple[float, float]) -> np.ndarray:
    sx, sy = start
    ex, ey = end
    length = math.hypot(ex - sx, ey - sy) or 1.0
    tx, ty = (ex - sx) / length, (ey - sy) / length
    pts = _interior(path, end)
    return np.array([(p.x - sx) * (-ty) + (p.y - sy) * tx for p in pts], dtype=np.float64)


def _step_mags(path: list[_Pt], end: tuple[float, float]) -> list[float]:
    pts = _drop_trailing_on_target(path, end)
    mags = []
    for a, b in zip(pts, pts[1:]):
        mag = math.hypot(b.x - a.x, b.y - a.y)
        if mag >= 0.5:
            mags.append(mag)
    return mags


def path_metrics(
    path: list[dict] | list[TouchStep],
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    drop_forced_end: bool = True,
) -> dict:
    steps = _as_steps(path)
    perps = _signed_perps(steps, start, end)
    if len(perps) == 0:
        bow_signed = 0.0
    else:
        bow_signed = float(perps[int(np.argmax(np.abs(perps)))])
    last = _model_end(steps, end) if drop_forced_end else steps[-1]
    end_err = math.hypot(end[0] - last.x, end[1] - last.y)
    step_px = _step_mags(steps, end)
    return {
        "steps": len(steps),
        "ms": float(steps[-1].t) if steps else 0.0,
        "bow_px": abs(bow_signed),
        "bow_signed": bow_signed,
        "step_px": float(np.median(step_px)) if step_px else 0.0,
        "end_err": end_err,
        "reach": end_err < REACH_PX,
        "close": end_err < CLOSE_PX,
    }


def _summarize(rows: list[dict]) -> dict:
    empty = {
        "n": 0,
        "steps_median": 0.0,
        "ms_median": 0.0,
        "step_px_median": 0.0,
        "bow_abs_median": 0.0,
        "bow_signed_median": 0.0,
        "end_err_median": 0.0,
        "reach_rate": 0.0,
        "close_rate": 0.0,
    }
    if not rows:
        return empty
    return {
        "n": len(rows),
        "steps_median": float(np.median([r["steps"] for r in rows])),
        "ms_median": float(np.median([r["ms"] for r in rows])),
        "step_px_median": float(np.median([r["step_px"] for r in rows])),
        "bow_abs_median": float(np.median([r["bow_px"] for r in rows])),
        "bow_signed_median": float(np.median([r["bow_signed"] for r in rows])),
        "end_err_median": float(np.median([r["end_err"] for r in rows])),
        "reach_rate": sum(1 for r in rows if r["reach"]) / len(rows),
        "close_rate": sum(1 for r in rows if r["close"]) / len(rows),
    }


def evaluate(model_path: Path, data_path: Path, limit: int | None = 200, seed: int = 42) -> dict:
    session = load_session(model_path)
    swipes = _load_val_swipes(data_path, model_config["validation_split"], seed=seed, limit=limit)
    modes = {
        "real": None,
        "raw": GenerateOptions(no_backtrack=False, seed=seed),
        "noback": GenerateOptions(no_backtrack=True, seed=seed),
    }

    buckets = {name: {"H": [], "D": [], "V": [], "all": []} for name in modes}

    for i, (start, end, real_path) in enumerate(swipes):
        angle = swipe_angle_bucket(end[0] - start[0], end[1] - start[1])
        for name, opts in modes.items():
            if opts is None:
                row = path_metrics(real_path, start, end, drop_forced_end=False)
            else:
                rng = np.random.default_rng(seed + i)
                gen = generate_touch_path(session, start, end, opts, rng)
                row = path_metrics(gen, start, end)
            buckets[name]["all"].append(row)
            buckets[name][angle].append(row)

    return {name: {key: _summarize(rows) for key, rows in by_angle.items()} for name, by_angle in buckets.items()}


def _print_mode(name: str, stats: dict) -> None:
    overall = stats["all"]
    print(
        f"  {name:7} steps={overall['steps_median']:.0f}  "
        f"{overall['ms_median']:.0f}ms  "
        f"step={overall['step_px_median']:.0f}px  "
        f"bow={overall['bow_abs_median']:.0f}px "
        f"(signed {overall['bow_signed_median']:+.0f})  "
        f"end_err={overall['end_err_median']:.0f}px  "
        f"close@15={overall['close_rate']*100:.0f}%  "
        f"reach@3={overall['reach_rate']*100:.0f}%"
    )
    for key in ("H", "D", "V"):
        s = stats[key]
        if s["n"] == 0:
            continue
        print(
            f"          {key} n={s['n']:<4}  "
            f"steps={s['steps_median']:.0f}  "
            f"{s['ms_median']:.0f}ms  "
            f"step={s['step_px_median']:.0f}px  "
            f"bow={s['bow_abs_median']:.0f}px  "
            f"end_err={s['end_err_median']:.0f}px  "
            f"close@15={s['close_rate']*100:.0f}%"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate touch.onnx generation quality")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args(argv)

    data = args.data or default_data_path()

    model = args.model or default_onnx_path()
    stats = evaluate(model, data, limit=args.limit)
    print(f"Evaluated {stats['real']['all']['n']} validation swipes from {data}")
    print(f"model {model}")
    print("end_err / close@15 / reach@3 use the last free MDN point (forced target snap excluded)")
    for name in ("real", "raw", "noback"):
        _print_mode(name, stats[name])


if __name__ == "__main__":
    main()
