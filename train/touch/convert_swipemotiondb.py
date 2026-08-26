"""Convert CSD4CA touch CSV into mousecrack-style training jsonl."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

CONDITION = {
    "Normal": "seated",
    "Walking": "walking",
    "Stressful": "stress",
}

MIN_POINTS = 3
MIN_DURATION_MS = 50.0
MAX_DT_MS = 500.0
DEVICE = "pixel-6a"


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
                }

    out = []
    for sid, pts in groups.items():
        traj = _to_trajectory(pts, meta[sid])
        if traj is not None:
            out.append(traj)
    return out


def write_jsonl(trajectories: Iterable[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
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

    parser = argparse.ArgumentParser(description="Convert CSD4CA touch CSV to jsonl")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "raw" / "CSD4CA" / "touch_data.csv",
        help="Path to CSD4CA touch_data.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "touch_data.jsonl",
        help="Output jsonl path",
    )
    args = parser.parse_args(argv)
    print(f"Reading {args.input}")
    trajectories = convert_touch_csv(args.input)
    write_jsonl(trajectories, args.output)
    print(f"Wrote {args.output}")
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
