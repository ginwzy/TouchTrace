"""Human-baseline IMU stats from fused jsonl, split by condition."""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
_TRAIN_ROOT = HERE.parent
if str(_TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAIN_ROOT))

from touch.features import iter_jsonl, resolve_jsonl


def default_sensor_data_path() -> Path:
    return resolve_jsonl(HERE, "sensor_data")


def load_fused(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    for row in iter_jsonl(path):
        rows.append(row)
        if limit is not None and len(rows) >= limit:
            break
    return rows


def _pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, int(round((p / 100.0) * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def summarize(rows: list[dict]) -> None:
    print(f"trajectories: {len(rows)}")
    if not rows:
        return
    cond_n = defaultdict(int)
    mag = defaultdict(lambda: {"acc": [], "gyro": []})
    first = defaultdict(list)
    mismatch = 0
    for row in rows:
        cond = str((row.get("meta") or {}).get("condition") or "unknown")
        path = row.get("path") or []
        sensors = row.get("sensors") or []
        cond_n[cond] += 1
        if len(path) != len(sensors):
            mismatch += 1
            continue
        if sensors:
            a0 = sensors[0]["accel"]
            first[cond].append(math.hypot(a0[0], a0[1], a0[2]))
        for s in sensors:
            a = s["accel"]
            g = s["gyro"]
            mag[cond]["acc"].append(math.hypot(a[0], a[1], a[2]))
            mag[cond]["gyro"].append(math.hypot(g[0], g[1], g[2]))
    print("condition:", dict(sorted(cond_n.items())))
    print(f"path/sensor length mismatch: {mismatch}")
    print(f"{'cond':<10} {'n':>7} {'|a| p50':>10} {'|a| mean':>10} {'|a| p90':>10} {'|g| p50':>10} {'|g| mean':>10} {'t0 |a|':>10}")
    for cond in sorted(mag):
        acc = sorted(mag[cond]["acc"])
        gyro = sorted(mag[cond]["gyro"])
        t0 = sorted(first[cond])
        print(
            f"{cond:<10} {cond_n[cond]:>7} {_pct(acc, 50):>10.3f} {sum(acc) / len(acc):>10.3f} "
            f"{_pct(acc, 90):>10.3f} {_pct(gyro, 50):>10.4f} {sum(gyro) / len(gyro):>10.4f} "
            f"{_pct(t0, 50):>10.3f}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Print fused IMU stats by condition")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    path = args.data if args.data else default_sensor_data_path()
    if not path.exists():
        raise SystemExit(f"Missing {path}")
    summarize(load_fused(path, limit=args.limit))


if __name__ == "__main__":
    main()
