"""Evaluate raw (no guided) touch generation against validation swipes."""

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
from generate import GenerateOptions, generate_touch_path, load_session


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
        swipes.append((start, end, len(path), path[-1]["timestamp"] - path[0]["timestamp"]))
    return swipes


def evaluate(model_path: Path, data_path: Path, limit: int | None = 200, seed: int = 42) -> dict:
    session = load_session(model_path)
    swipes = _load_val_swipes(data_path, model_config["validation_split"], seed=seed, limit=limit)
    raw_opts = GenerateOptions(guided=False, smooth=False, seed=seed)
    guided_opts = GenerateOptions(guided=True, smooth=True, seed=seed)

    raw_steps, guided_steps = [], []
    raw_ms, guided_ms = [], []
    raw_ok = guided_ok = 0

    for i, (start, end, real_len, real_ms) in enumerate(swipes):
        rng = np.random.default_rng(seed + i)
        raw = generate_touch_path(session, start, end, raw_opts, rng)
        guided = generate_touch_path(session, start, end, guided_opts, rng)

        raw_steps.append(len(raw))
        guided_steps.append(len(guided))
        raw_ms.append(raw[-1].t)
        guided_ms.append(guided[-1].t)

        for path in (raw, guided):
            last = path[-2] if len(path) > 1 else path[-1]
            err = math.hypot(end[0] - last.x, end[1] - last.y)
            if path is raw and err < 3:
                raw_ok += 1
            if path is guided and err < 3:
                guided_ok += 1

    n = len(swipes)
    return {
        "n": n,
        "raw_steps_median": float(np.median(raw_steps)),
        "guided_steps_median": float(np.median(guided_steps)),
        "raw_ms_median": float(np.median(raw_ms)),
        "guided_ms_median": float(np.median(guided_ms)),
        "raw_reach_rate": raw_ok / n,
        "guided_reach_rate": guided_ok / n,
    }


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
    print(f"Evaluated {stats['n']} validation swipes from {data}")
    print(f"  raw    median steps={stats['raw_steps_median']:.0f}  duration={stats['raw_ms_median']:.0f}ms  reach={stats['raw_reach_rate']*100:.0f}%")
    print(f"  guided median steps={stats['guided_steps_median']:.0f}  duration={stats['guided_ms_median']:.0f}ms  reach={stats['guided_reach_rate']*100:.0f}%")


if __name__ == "__main__":
    main()
