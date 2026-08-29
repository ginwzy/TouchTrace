"""Human-baseline IMU stats, plus generated vs real |a|/|gyro| by condition."""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_TRAIN_ROOT = HERE.parent
if str(_TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAIN_ROOT))

from sensor.config import model_config
from sensor.features import CONDITIONS, subsample_paired
from sensor.generate import (
    default_data_path,
    default_norm_path,
    default_onnx_path,
    generate_sensor_along_path,
    load_norm,
    load_session,
    sensor_mags,
)
from touch.features import iter_jsonl


def default_sensor_data_path() -> Path:
    return default_data_path()


def load_fused(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    for row in iter_jsonl(path):
        rows.append(row)
        if limit is not None and len(rows) >= limit:
            break
    return rows


def val_rows(
    rows: list[dict],
    val_fraction: float,
    seed: int = 42,
    limit: int | None = None,
) -> list[dict]:
    if not rows:
        return []
    order = np.random.default_rng(seed).permutation(len(rows))
    n_val = max(1, int(len(rows) * val_fraction))
    picked = [rows[int(i)] for i in order[:n_val]]
    if limit is not None:
        picked = picked[:limit]
    return picked


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
            _, acc, gyro = sensor_mags(sensors)
            first[cond].append(acc[0])
            mag[cond]["acc"].extend(acc)
            mag[cond]["gyro"].extend(gyro)
    print("condition:", dict(sorted(cond_n.items())))
    print(f"path/sensor length mismatch: {mismatch}")
    _print_mag_table(cond_n, mag, first)


def _print_mag_table(
    cond_n: dict[str, int],
    mag: dict[str, dict[str, list[float]]],
    first: dict[str, list[float]] | None = None,
    extra: dict[str, dict[str, float]] | None = None,
) -> None:
    extra_hdr = ""
    if extra:
        extra_hdr = f" {'|a| last':>10} {'last/t0':>8}"
    first_hdr = f" {'t0 |a|':>10}" if first is not None else ""
    print(
        f"{'cond':<10} {'n':>7} {'|a| p50':>10} {'|a| mean':>10} {'|a| p90':>10} "
        f"{'|g| p50':>10} {'|g| mean':>10}{first_hdr}{extra_hdr}"
    )
    for cond in sorted(mag):
        acc = sorted(mag[cond]["acc"])
        gyro = sorted(mag[cond]["gyro"])
        if not acc:
            continue
        line = (
            f"{cond:<10} {cond_n[cond]:>7} {_pct(acc, 50):>10.3f} {sum(acc) / len(acc):>10.3f} "
            f"{_pct(acc, 90):>10.3f} {_pct(gyro, 50):>10.4f} {sum(gyro) / len(gyro):>10.4f}"
        )
        if first is not None:
            t0 = sorted(first.get(cond) or [])
            line += f" {_pct(t0, 50):>10.3f}"
        if extra and cond in extra:
            line += f" {extra[cond]['last_a']:>10.3f} {extra[cond]['drift']:>8.2f}"
        print(line)


def _vec_err(real: list[dict], gen: list[dict]) -> tuple[list[float], list[float]]:
    acc_err, gyro_err = [], []
    for r, g in zip(real[1:], gen[1:]):
        ra, ga = r["accel"], g["accel"]
        rg, gg = r["gyro"], g["gyro"]
        acc_err.append(math.hypot(ga[0] - ra[0], ga[1] - ra[1], ga[2] - ra[2]))
        gyro_err.append(math.hypot(gg[0] - rg[0], gg[1] - rg[1], gg[2] - rg[2]))
    return acc_err, gyro_err


def _mode_name(temp: float, teacher: bool) -> str:
    prefix = "tf" if teacher else "ar"
    if temp <= 0:
        return f"{prefix}-mean"
    if temp >= 1:
        return f"{prefix}-sample"
    return f"{prefix}-t{temp:g}"


def compare_generated(
    rows: list[dict],
    session,
    norm: dict,
    *,
    limit: int = 200,
    seed: int = 42,
    temps: tuple[float, ...] = (0.0, 0.2, 0.3, 0.5, 1.0),
    report: bool = False,
) -> None:
    val = val_rows(rows, model_config["validation_split"], seed=seed, limit=limit)
    step_px = float(model_config["min_step_px"])
    modes = [(_mode_name(t, False), t, False) for t in temps]
    modes.append((_mode_name(0.0, True), 0.0, True))
    labels = ("real",) + tuple(name for name, _, _ in modes)
    buckets = {label: {c: {"acc": [], "gyro": []} for c in CONDITIONS} for label in labels}
    n = {label: defaultdict(int) for label in labels}
    first = {label: defaultdict(list) for label in labels}
    last_a = {label: defaultdict(list) for label in labels}
    err = {name: {c: {"acc": [], "gyro": []} for c in CONDITIONS} for name, _, _ in modes}
    used = 0
    for i, row in enumerate(val):
        cond = str((row.get("meta") or {}).get("condition") or "")
        if cond not in CONDITIONS:
            continue
        path, sensors = subsample_paired(row.get("path") or [], row.get("sensors") or [], min_step_px=step_px)
        if len(path) < 2 or len(path) != len(sensors):
            continue
        imu0 = sensors[0]["accel"] + sensors[0]["gyro"]
        seqs = {"real": sensors}
        for name, temp, teacher in modes:
            seqs[name] = generate_sensor_along_path(
                session,
                path,
                imu0,
                cond,
                norm,
                np.random.default_rng(seed + i),
                temp=temp,
                teacher_sensors=sensors if teacher else None,
            )
        for label, seq in seqs.items():
            _, acc, gyro = sensor_mags(seq)
            buckets[label][cond]["acc"].extend(acc)
            buckets[label][cond]["gyro"].extend(gyro)
            n[label][cond] += 1
            first[label][cond].append(acc[0])
            last_a[label][cond].append(acc[-1])
        for name, _, _ in modes:
            ae, ge = _vec_err(sensors, seqs[name])
            err[name][cond]["acc"].extend(ae)
            err[name][cond]["gyro"].extend(ge)
        used += 1
    print(
        f"\nGenerated vs human on {used} validation swipes "
        f"(frozen touch path, min_step_px={step_px})"
    )
    print("ar = autoregressive; tf = teacher-forced; t* = MDN temp (0=mean, 1=full sample)")
    extras: dict[str, dict] = {}
    for label in labels:
        extra = {}
        for cond in CONDITIONS:
            t0 = first[label].get(cond) or []
            last = last_a[label].get(cond) or []
            if not t0 or not last:
                continue
            mean_last = sum(last) / len(last)
            mean_t0 = sum(t0) / len(t0)
            extra[cond] = {"last_a": mean_last, "drift": mean_last / max(mean_t0, 1e-6)}
        extras[label] = extra
        print(label)
        _print_mag_table(n[label], buckets[label], first[label], extra)
    print("\npointwise |gen-human| after t0 (median)")
    print(f"{'mode':<12} {'cond':<10} {'|da| p50':>10} {'|dg| p50':>10}")
    for name, _, _ in modes:
        for cond in CONDITIONS:
            ae = sorted(err[name][cond]["acc"])
            ge = sorted(err[name][cond]["gyro"])
            if not ae:
                continue
            print(f"{name:<12} {cond:<10} {_pct(ae, 50):>10.3f} {_pct(ge, 50):>10.4f}")
    if report:
        _print_paste_report(used, labels, buckets, err, extras)


def _print_paste_report(
    used: int,
    labels: tuple[str, ...],
    buckets: dict,
    err: dict,
    extras: dict,
) -> None:
    print("\n=== SENSOR EVAL REPORT (paste into chat) ===")
    print(
        f"target=delta  mdn_temp={model_config['mdn_temp']}  "
        f"ss_max={model_config['ss_max']}  ss_temp={model_config['ss_temp']}  "
        f"ss_unroll_hops={model_config.get('ss_unroll_hops', 1)}  n={used}"
    )
    print(f"{'mode':<12} {'cond':<10} {'|a| p50':>8} {'|a| p90':>8} {'|g| p50':>8} {'|da|':>8} {'|dg|':>8} {'last/t0':>8}")
    for label in labels:
        extra = extras.get(label) or {}
        for cond in CONDITIONS:
            acc = sorted(buckets[label][cond]["acc"])
            gyro = sorted(buckets[label][cond]["gyro"])
            if not acc:
                continue
            da = dg = float("nan")
            if label in err:
                ae = sorted(err[label][cond]["acc"])
                ge = sorted(err[label][cond]["gyro"])
                if ae:
                    da, dg = _pct(ae, 50), _pct(ge, 50)
            drift = extra.get(cond, {}).get("drift", float("nan"))
            print(
                f"{label:<12} {cond:<10} {_pct(acc, 50):>8.3f} {_pct(acc, 90):>8.3f} "
                f"{_pct(gyro, 50):>8.4f} {da:>8.3f} {dg:>8.4f} {drift:>8.2f}"
            )
    print("Watch: AR |dg| vs tf-mean |dg| (should close); walking |a| p90 vs real.")
    print("=== END REPORT ===")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Print fused IMU stats by condition")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Cap jsonl rows loaded (debug)")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--norm", type=Path, default=None)
    parser.add_argument("--gen-limit", type=int, default=200, help="Validation swipes to roll out")
    parser.add_argument(
        "--temps",
        default=None,
        help="Comma-separated MDN temps for autoregressive compare (0=mean, 1=full sample)",
    )
    parser.add_argument("--skip-gen", action="store_true", help="Human baseline only")
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print a compact paste-into-chat summary after the generation compare",
    )
    args = parser.parse_args(argv)
    path = args.data if args.data else default_sensor_data_path()
    if not path.exists():
        raise SystemExit(f"Missing {path}")
    rows = load_fused(path, limit=args.limit)
    summarize(rows)
    if args.skip_gen:
        return
    model = args.model or default_onnx_path()
    if not model.exists():
        print(f"Skip generation compare (missing {model})")
        return
    if args.temps:
        temps = tuple(float(x) for x in args.temps.split(",") if x.strip() != "")
    else:
        t = float(model_config["mdn_temp"])
        temps = (0.0, t, 0.5, 1.0)
    compare_generated(
        rows,
        load_session(model),
        load_norm(args.norm or default_norm_path()),
        limit=args.gen_limit,
        temps=temps,
        report=args.report,
    )


if __name__ == "__main__":
    main()
