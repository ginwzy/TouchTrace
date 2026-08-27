"""Evaluate touch.onnx generation quality, split by swipe angle."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from config_touch import model_config
from features import swipe_angle_bucket
from generate import GenerateOptions, TouchStep, generate_touch_path, load_session


def _open_jsonl(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open()


def _load_val_swipes(data_path: Path, val_fraction: float, seed: int = 42, limit: int | None = 200):
    rows = []
    with _open_jsonl(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
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


def _perp_median(path: list[TouchStep], start: tuple[float, float], end: tuple[float, float]) -> float:
    sx, sy = start
    ex, ey = end
    length = math.hypot(ex - sx, ey - sy) or 1.0
    tx, ty = (ex - sx) / length, (ey - sy) / length
    pts = path[:-1] if len(path) > 1 else path
    perps = [abs((p.x - sx) * (-ty) + (p.y - sy) * tx) for p in pts[1:]]
    return float(np.median(perps)) if perps else 0.0


def _summarize(rows: list[tuple[int, float, float, bool]]) -> dict:
    if not rows:
        return {"n": 0, "steps_median": 0.0, "ms_median": 0.0, "perp_median": 0.0, "reach_rate": 0.0}
    return {
        "n": len(rows),
        "steps_median": float(np.median([r[0] for r in rows])),
        "ms_median": float(np.median([r[1] for r in rows])),
        "perp_median": float(np.median([r[2] for r in rows])),
        "reach_rate": sum(1 for r in rows if r[3]) / len(rows),
    }


def evaluate(model_path: Path, data_path: Path, limit: int | None = 200, seed: int = 42) -> dict:
    session = load_session(model_path)
    swipes = _load_val_swipes(data_path, model_config["validation_split"], seed=seed, limit=limit)
    modes = {
        "raw": GenerateOptions(guided=False, smooth=False, no_backtrack=False, seed=seed),
        "noback": GenerateOptions(guided=False, smooth=False, no_backtrack=True, seed=seed),
        "guided": GenerateOptions(guided=True, smooth=True, seed=seed),
    }

    buckets = {name: {"H": [], "D": [], "V": [], "all": []} for name in modes}

    for i, (start, end, _path) in enumerate(swipes):
        angle = swipe_angle_bucket(end[0] - start[0], end[1] - start[1])
        for name, opts in modes.items():
            rng = np.random.default_rng(seed + i)
            gen = generate_touch_path(session, start, end, opts, rng)
            last = gen[-2] if len(gen) > 1 else gen[-1]
            err = math.hypot(end[0] - last.x, end[1] - last.y)
            row = (len(gen), float(gen[-1].t), _perp_median(gen, start, end), err < 3.0)
            buckets[name]["all"].append(row)
            buckets[name][angle].append(row)

    return {name: {key: _summarize(rows) for key, rows in by_angle.items()} for name, by_angle in buckets.items()}


def _print_mode(name: str, stats: dict) -> None:
    overall = stats["all"]
    print(
        f"  {name:7} median steps={overall['steps_median']:.0f}  "
        f"duration={overall['ms_median']:.0f}ms  "
        f"reach={overall['reach_rate']*100:.0f}%  "
        f"perp={overall['perp_median']:.0f}px"
    )
    for key in ("H", "D", "V"):
        s = stats[key]
        if s["n"] == 0:
            continue
        print(
            f"          {key} n={s['n']:<4}  reach={s['reach_rate']*100:.0f}%  "
            f"steps={s['steps_median']:.0f}  perp={s['perp_median']:.0f}px"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate touch.onnx generation quality")
    parser.add_argument("--model", type=Path, default=HERE / "touch.onnx")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args(argv)

    data = args.data
    if data is None:
        gz = HERE / "touch_data.jsonl.gz"
        data = gz if gz.exists() else HERE / "touch_data.jsonl"

    stats = evaluate(args.model, data, limit=args.limit)
    print(f"Evaluated {stats['raw']['all']['n']} validation swipes from {data}")
    for name in ("raw", "noback", "guided"):
        _print_mode(name, stats[name])


if __name__ == "__main__":
    main()
