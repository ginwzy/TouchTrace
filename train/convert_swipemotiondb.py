"""Convert CSD4CA touch (+ optional IMU) CSV into training jsonl."""

from __future__ import annotations

import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

CONDITION = {
    "Normal": "seated",
    "Walking": "walking",
    "Stressful": "stress",
}

MIN_POINTS = 3
MIN_DURATION_MS = 50.0
MAX_DT_MS = 500.0
DEVICE = "pixel-6a"
SENSOR_NS_TO_MS = 1e6
# Drop IMU streams whose native span is far from the touch duration.
MAX_SENSOR_SPAN_RATIO = 3.0
MIN_SENSOR_COVER_RATIO = 0.5
TRAIN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TRAIN_ROOT.parent
RAW_DIR = REPO_ROOT / "data" / "raw" / "CSD4CA"


def convert_touch_csv(path: str | Path) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    meta: dict[str, dict] = {}

    with Path(path).open(newline="") as f:
        for row in csv.DictReader(f):
            sid = row["id_swipe"].strip()
            t = _f(row.get("time"))
            x = _f(row.get("x"))
            y = _f(row.get("y"))
            if sid == "" or t is None or x is None or y is None:
                continue
            groups[sid].append(
                {
                    "t": t,
                    "x": x,
                    "y": y,
                    "pressure": _f(row.get("pressure")),
                    "area": _f(row.get("touch_major")),
                }
            )
            if sid not in meta:
                meta[sid] = {
                    "condition": CONDITION.get(row.get("scenario", "").strip(), None),
                    "session": int(row["session"]) if row.get("session", "").strip().isdigit() else row.get("session"),
                    "user_id": str(row.get("user_id", "")).strip(),
                    "device": DEVICE,
                    "id_swipe": sid,
                }

    out = []
    for sid, pts in groups.items():
        traj = _to_trajectory(pts, meta[sid])
        if traj is not None:
            out.append(traj)
    return out


def parse_sensor_csv(path: str | Path) -> dict[str, list[tuple[float, float, float, float]]]:
    """Group IMU rows by id_swipe. Each sample is (t_ns, x, y, z)."""
    groups: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    with Path(path).open(newline="") as f:
        for row in csv.DictReader(f):
            sid = (row.get("id_swipe") or "").strip()
            t = _f(row.get("time"))
            x = _f(row.get("x"))
            y = _f(row.get("y"))
            z = _f(row.get("z"))
            if sid == "" or t is None or x is None or y is None or z is None:
                continue
            groups[sid].append((t, x, y, z))
    for sid, samples in groups.items():
        groups[sid] = _collapse_sensor(samples)
    return groups


def interpolate_xyz(
    samples: list[tuple[float, float, float, float]],
    query_ms: list[float],
) -> list[list[float]] | None:
    """Lerp device-frame XYZ onto touch-relative milliseconds.

    Sensor clocks are converted with (t_ns - t0_ns) / 1e6. Do not mix touch
    absolute ms with sensor nanoseconds.
    """
    if len(samples) < 2:
        return None
    arr = np.asarray(samples, dtype=np.float64)
    times = (arr[:, 0] - arr[0, 0]) / SENSOR_NS_TO_MS
    q = np.asarray(query_ms, dtype=np.float64)
    xyz = np.stack([np.interp(q, times, arr[:, i]) for i in range(1, 4)], axis=1)
    return xyz.tolist()


def merge_sensors(
    trajectories: list[dict],
    acc_by_id: dict[str, list[tuple[float, float, float, float]]],
    gyro_by_id: dict[str, list[tuple[float, float, float, float]]],
    *,
    max_span_ratio: float = MAX_SENSOR_SPAN_RATIO,
    min_cover_ratio: float = MIN_SENSOR_COVER_RATIO,
) -> tuple[list[dict], dict]:
    stats = Counter()
    out: list[dict] = []
    for traj in trajectories:
        sid = str(traj.get("meta", {}).get("id_swipe", ""))
        acc = acc_by_id.get(sid)
        gyro = gyro_by_id.get(sid)
        if not acc or not gyro:
            stats["missing"] += 1
            continue
        touch_ms = float(traj["path"][-1]["timestamp"])
        if not _span_ok(acc, touch_ms, max_span_ratio, min_cover_ratio):
            stats["span_drop"] += 1
            continue
        if not _span_ok(gyro, touch_ms, max_span_ratio, min_cover_ratio):
            stats["span_drop"] += 1
            continue
        times = [float(p["timestamp"]) for p in traj["path"]]
        acc_i = interpolate_xyz(acc, times)
        gyro_i = interpolate_xyz(gyro, times)
        if acc_i is None or gyro_i is None:
            continue
        sensors = [
            {"timestamp": t, "accel": a, "gyro": g}
            for t, a, g in zip(times, acc_i, gyro_i)
        ]
        out.append(_compact_fused(traj, sensors))
        stats["kept"] += 1
    return out, dict(stats)


def write_jsonl(trajectories: Iterable[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as f:
        for traj in trajectories:
            f.write(json.dumps(traj, separators=(",", ":")) + "\n")


def print_stats(trajectories: list[dict]) -> None:
    n = len(trajectories)
    print(f"trajectories: {n}")
    if n == 0:
        return
    cond = Counter(t["meta"]["condition"] for t in trajectories)
    lengths = [t["length"] for t in trajectories]
    dts = []
    dt_le0 = 0
    target_mismatch = 0
    for t in trajectories:
        last = t["path"][-1]
        if last["x"] != t["target"]["x"] or last["y"] != t["target"]["y"]:
            target_mismatch += 1
        for a, b in zip(t["path"], t["path"][1:]):
            dt = b["timestamp"] - a["timestamp"]
            dts.append(dt)
            if dt <= 0:
                dt_le0 += 1
    print("condition:", dict(sorted(cond.items())))
    print(
        "points: min/mean/max",
        min(lengths),
        round(sum(lengths) / n, 1),
        max(lengths),
    )
    if dts:
        dts.sort()
        print(
            "dt_ms: min/p50/mean/max",
            dts[0],
            dts[len(dts) // 2],
            round(sum(dts) / len(dts), 2),
            dts[-1],
        )
    print(f"dt<=0 pairs: {dt_le0}")
    print(f"target mismatch: {target_mismatch}")


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Convert CSD4CA CSV to jsonl")
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_DIR / "touch_data.csv",
        help="Path to CSD4CA touch_data.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output jsonl path (default: train/touch/touch_data.jsonl, or train/sensor/sensor_data.jsonl.gz with --sensors)",
    )
    parser.add_argument("--sensors", action="store_true", help="Fuse accelerometer + gyroscope onto touch points")
    parser.add_argument("--acc", type=Path, default=RAW_DIR / "acc_data.csv")
    parser.add_argument("--gyro", type=Path, default=RAW_DIR / "gyro_data.csv")
    parser.add_argument("--limit", type=int, default=None, help="Keep at most N fused trajectories (debug)")
    args = parser.parse_args(argv)

    output = args.output
    if output is None:
        output = (
            TRAIN_ROOT / "sensor" / "sensor_data.jsonl.gz"
            if args.sensors
            else TRAIN_ROOT / "touch" / "touch_data.jsonl"
        )

    print(f"Reading {args.input}")
    trajectories = convert_touch_csv(args.input)
    if args.sensors:
        print(f"Reading {args.acc}")
        acc_by_id = parse_sensor_csv(args.acc)
        print(f"Reading {args.gyro}")
        gyro_by_id = parse_sensor_csv(args.gyro)
        trajectories, merge_stats = merge_sensors(trajectories, acc_by_id, gyro_by_id)
        print("merge:", merge_stats)
        if args.limit is not None:
            trajectories = trajectories[: args.limit]
    write_jsonl(trajectories, output)
    print(f"Wrote {output} ({output.stat().st_size / 1e6:.1f} MB)")
    print_stats(trajectories)


def _to_trajectory(pts: list[dict], meta: dict) -> dict | None:
    pts = sorted(pts, key=lambda p: p["t"])
    collapsed = [pts[0]]
    for p in pts[1:]:
        if p["t"] == collapsed[-1]["t"]:
            collapsed[-1] = p
        else:
            collapsed.append(p)

    t0 = collapsed[0]["t"]
    path = []
    for p in collapsed:
        rec = {
            "x": p["x"],
            "y": p["y"],
            "timestamp": p["t"] - t0,
        }
        if p["pressure"] is not None:
            rec["pressure"] = p["pressure"]
        if p["area"] is not None:
            rec["area"] = p["area"]
        path.append(rec)

    if len(path) < MIN_POINTS:
        return None
    duration = path[-1]["timestamp"]
    if duration < MIN_DURATION_MS:
        return None
    for i in range(1, len(path)):
        dt = path[i]["timestamp"] - path[i - 1]["timestamp"]
        if dt <= 0 or dt > MAX_DT_MS:
            return None

    last = path[-1]
    return {
        "length": len(path),
        "target": {"x": last["x"], "y": last["y"]},
        "meta": meta,
        "path": path,
    }


def _collapse_sensor(
    samples: list[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    samples = sorted(samples, key=lambda s: s[0])
    out = [samples[0]]
    for s in samples[1:]:
        if s[0] == out[-1][0]:
            out[-1] = s
        else:
            out.append(s)
    return out


def _span_ok(
    samples: list[tuple[float, float, float, float]],
    touch_ms: float,
    max_span_ratio: float,
    min_cover_ratio: float,
) -> bool:
    if len(samples) < 2:
        return False
    span_ms = (samples[-1][0] - samples[0][0]) / SENSOR_NS_TO_MS
    duration = max(touch_ms, 1.0)
    if span_ms > max_span_ratio * duration:
        return False
    if span_ms < min_cover_ratio * duration:
        return False
    return True


def _compact_fused(traj: dict, sensors: list[dict]) -> dict:
    path = [
        {
            "x": round(float(p["x"]), 2),
            "y": round(float(p["y"]), 2),
            "timestamp": round(float(p["timestamp"]), 3),
        }
        for p in traj["path"]
    ]
    sensors = [
        {
            "timestamp": round(float(s["timestamp"]), 3),
            "accel": [round(float(v), 5) for v in s["accel"]],
            "gyro": [round(float(v), 6) for v in s["gyro"]],
        }
        for s in sensors
    ]
    meta_in = traj.get("meta") or {}
    meta = {
        "condition": meta_in.get("condition"),
        "session": meta_in.get("session"),
        "user_id": meta_in.get("user_id"),
        "device": meta_in.get("device", DEVICE),
    }
    last = path[-1]
    return {
        "length": len(path),
        "target": {"x": last["x"], "y": last["y"]},
        "meta": meta,
        "path": path,
        "sensors": sensors,
    }


def _f(value: str | None) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


if __name__ == "__main__":
    main()
