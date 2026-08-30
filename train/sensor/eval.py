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
from sensor.split import load_user_split, rows_for_partition
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
    stratified: bool = True,
) -> list[dict]:
    if not rows:
        return []
    order = np.random.default_rng(seed).permutation(len(rows))
    n_val = max(1, int(len(rows) * val_fraction))
    pool = [rows[int(i)] for i in order[:n_val]]
    if limit is None or limit >= len(pool):
        return pool
    if not stratified:
        return pool[:limit]

    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in pool:
        condition = str((row.get("meta") or {}).get("condition") or "unknown")
        buckets[condition].append(row)
    keys = [condition for condition in CONDITIONS if buckets.get(condition)]
    keys.extend(sorted(condition for condition in buckets if condition not in CONDITIONS))
    picked: list[dict] = []
    cursor = 0
    while len(picked) < limit:
        added = False
        for condition in keys:
            bucket = buckets[condition]
            if cursor < len(bucket):
                picked.append(bucket[cursor])
                added = True
                if len(picked) == limit:
                    break
        if not added:
            break
        cursor += 1
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


def _step_change_summary(sensors: list[dict]) -> tuple[float, float]:
    """Swipe-level median vector change, so long swipes do not dominate."""
    dacc: list[float] = []
    dgyro: list[float] = []
    for prev, curr in zip(sensors, sensors[1:]):
        pa, ca = prev["accel"], curr["accel"]
        pg, cg = prev["gyro"], curr["gyro"]
        dacc.append(math.hypot(ca[0] - pa[0], ca[1] - pa[1], ca[2] - pa[2]))
        dgyro.append(math.hypot(cg[0] - pg[0], cg[1] - pg[1], cg[2] - pg[2]))
    if not dacc:
        return float("nan"), float("nan")
    return float(np.median(dacc)), float(np.median(dgyro))


def _print_step_change_table(labels, activity) -> None:
    print("\ntemporal vector change (swipe-weighted median)")
    print(f"{'mode':<12} {'cond':<10} {'|delta a|':>10} {'|delta g|':>10}")
    for label in labels:
        for condition in CONDITIONS:
            dacc = sorted(activity[label][condition]["acc"])
            dgyro = sorted(activity[label][condition]["gyro"])
            if not dacc:
                continue
            print(
                f"{label:<12} {condition:<10} "
                f"{_pct(dacc, 50):>10.4f} {_pct(dgyro, 50):>10.5f}"
            )


def _mode_name(temp: float, teacher: bool, innovation_rho: float = 0.0) -> str:
    prefix = "tf" if teacher else "ar"
    if temp <= 0:
        base = f"{prefix}-mean"
    elif temp >= 1:
        base = f"{prefix}-sample"
    else:
        base = f"{prefix}-t{temp:g}"
    if not teacher and temp > 0 and innovation_rho > 0:
        base += f"-r{innovation_rho:g}"
    return base


def _lag_correlation(values: np.ndarray, lag: int) -> float:
    if len(values) <= lag:
        return float("nan")
    centered = values - values.mean(axis=0, keepdims=True)
    left = centered[:-lag]
    right = centered[lag:]
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(left * right) / denom)


def _gyro_persistence_summary(sensors: list[dict]) -> dict[str, float]:
    gyro = np.asarray([sample["gyro"] for sample in sensors], dtype=np.float64)
    if len(gyro) == 0:
        return {"mean": float("nan"), "residual": float("nan"), "lag1": float("nan"), "lag4": float("nan")}
    center = gyro.mean(axis=0)
    residual = np.linalg.norm(gyro - center, axis=1)
    return {
        "mean": float(np.linalg.norm(center)),
        "residual": float(np.median(residual)),
        "lag1": _lag_correlation(gyro, 1),
        "lag4": _lag_correlation(gyro, 4),
    }


def collect_generated(
    rows: list[dict],
    session,
    norm: dict,
    *,
    limit: int = 200,
    seed: int = 42,
    temps: tuple[float, ...] = (0.0, 0.2, 0.3, 0.4, 0.5, 1.0),
    innovation_rhos: tuple[float, ...] = (0.0,),
    selected_rows: list[dict] | None = None,
) -> dict:
    source = rows if selected_rows is None else selected_rows
    fraction = model_config["validation_split"] if selected_rows is None else 1.0
    val = val_rows(source, fraction, seed=seed, limit=limit)
    step_px = float(model_config["min_step_px"])
    modes = [
        (_mode_name(t, False, rho), t, False, rho)
        for t in temps
        for rho in innovation_rhos
        if t > 0 or rho == 0
    ]
    modes.append((_mode_name(0.0, True), 0.0, True, 0.0))
    labels = ("real",) + tuple(name for name, *_ in modes)
    buckets = {label: {c: {"acc": [], "gyro": []} for c in CONDITIONS} for label in labels}
    n = {label: defaultdict(int) for label in labels}
    first = {label: defaultdict(list) for label in labels}
    last_a = {label: defaultdict(list) for label in labels}
    err = {name: {c: {"acc": [], "gyro": []} for c in CONDITIONS} for name, *_ in modes}
    activity = {label: {c: {"acc": [], "gyro": []} for c in CONDITIONS} for label in labels}
    persistence = {
        label: {c: {"mean": [], "residual": [], "lag1": [], "lag4": []} for c in CONDITIONS}
        for label in labels
    }
    used = 0
    for i, row in enumerate(val):
        cond = str((row.get("meta") or {}).get("condition") or "")
        if cond not in CONDITIONS:
            continue
        path, sensors = subsample_paired(
            row.get("path") or [], row.get("sensors") or [], min_step_px=step_px
        )
        if len(path) < 2 or len(path) != len(sensors):
            continue
        imu0 = sensors[0]["accel"] + sensors[0]["gyro"]
        seqs = {"real": sensors}
        for name, temp, teacher, rho in modes:
            seqs[name] = generate_sensor_along_path(
                session,
                path,
                imu0,
                cond,
                norm,
                np.random.default_rng(seed + i),
                temp=temp,
                teacher_sensors=sensors if teacher else None,
                innovation_rho=rho,
            )
        for label, seq in seqs.items():
            _, acc, gyro = sensor_mags(seq)
            buckets[label][cond]["acc"].extend(acc)
            buckets[label][cond]["gyro"].extend(gyro)
            n[label][cond] += 1
            first[label][cond].append(acc[0])
            last_a[label][cond].append(acc[-1])
            dacc, dgyro = _step_change_summary(seq)
            if math.isfinite(dacc):
                activity[label][cond]["acc"].append(dacc)
                activity[label][cond]["gyro"].append(dgyro)
            for key, value in _gyro_persistence_summary(seq).items():
                if math.isfinite(value):
                    persistence[label][cond][key].append(value)
        for name, *_ in modes:
            ae, ge = _vec_err(sensors, seqs[name])
            err[name][cond]["acc"].extend(ae)
            err[name][cond]["gyro"].extend(ge)
        used += 1

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
    return {
        "used": used,
        "labels": labels,
        "modes": modes,
        "n": n,
        "buckets": buckets,
        "first": first,
        "err": err,
        "activity": activity,
        "persistence": persistence,
        "extras": extras,
        "step_px": step_px,
    }


def rollout_summary(stats: dict) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for label in stats["labels"]:
        by_condition: dict[str, dict[str, float]] = {}
        for cond in CONDITIONS:
            acc = sorted(stats["buckets"][label][cond]["acc"])
            gyro = sorted(stats["buckets"][label][cond]["gyro"])
            if not acc:
                continue
            activity = stats["activity"][label][cond]
            persistence = stats["persistence"][label][cond]
            errors = stats["err"].get(label, {}).get(cond, {})
            acc_err = sorted(errors.get("acc") or [])
            gyro_err = sorted(errors.get("gyro") or [])
            by_condition[cond] = {
                "n": int(stats["n"][label][cond]),
                "acc_p50": _pct(acc, 50),
                "acc_p90": _pct(acc, 90),
                "gyro_p50": _pct(gyro, 50),
                "gyro_mean": sum(gyro) / len(gyro),
                "delta_acc": _pct(sorted(activity["acc"]), 50),
                "delta_gyro": _pct(sorted(activity["gyro"]), 50),
                "gyro_swipe_mean": _pct(sorted(persistence["mean"]), 50),
                "gyro_residual": _pct(sorted(persistence["residual"]), 50),
                "gyro_lag1": _pct(sorted(persistence["lag1"]), 50),
                "gyro_lag4": _pct(sorted(persistence["lag4"]), 50),
                "point_acc": _pct(acc_err, 50) if acc_err else float("nan"),
                "point_gyro": _pct(gyro_err, 50) if gyro_err else float("nan"),
                "drift": stats["extras"].get(label, {}).get(cond, {}).get("drift", float("nan")),
            }
        out[label] = by_condition
    return out


def distribution_score(
    summary: dict[str, dict[str, dict[str, float]]],
    label: str,
    reference: str = "real",
) -> float:
    ratio_weights = {
        "acc_p90": 1.0,
        "gyro_p50": 2.0,
        "delta_acc": 1.0,
        "delta_gyro": 1.0,
        "gyro_swipe_mean": 2.0,
        "gyro_residual": 1.0,
        "drift": 1.0,
    }
    correlation_weights = {"gyro_lag1": 0.5, "gyro_lag4": 0.5}
    weighted_error = 0.0
    total_weight = 0.0
    for cond in CONDITIONS:
        actual = summary.get(label, {}).get(cond)
        expected = summary.get(reference, {}).get(cond)
        if not actual or not expected:
            continue
        for key, weight in ratio_weights.items():
            got = float(actual.get(key, float("nan")))
            want = float(expected.get(key, float("nan")))
            if math.isfinite(got) and math.isfinite(want) and got > 0 and want > 0:
                weighted_error += weight * abs(math.log(got / want))
                total_weight += weight
        for key, weight in correlation_weights.items():
            got = float(actual.get(key, float("nan")))
            want = float(expected.get(key, float("nan")))
            if math.isfinite(got) and math.isfinite(want):
                weighted_error += weight * abs(got - want)
                total_weight += weight
    return weighted_error / total_weight if total_weight else float("inf")


def _print_persistence_table(labels, persistence) -> None:
    print("\ngyro persistence (swipe-weighted median)")
    print(f"{'mode':<12} {'cond':<10} {'|mean g|':>10} {'|g-mean|':>10} {'lag1':>8} {'lag4':>8}")
    for label in labels:
        for cond in CONDITIONS:
            values = persistence[label][cond]
            means = sorted(values["mean"])
            if not means:
                continue
            print(
                f"{label:<12} {cond:<10} {_pct(means, 50):>10.4f} "
                f"{_pct(sorted(values['residual']), 50):>10.4f} "
                f"{_pct(sorted(values['lag1']), 50):>8.3f} "
                f"{_pct(sorted(values['lag4']), 50):>8.3f}"
            )


def compare_generated(
    rows: list[dict],
    session,
    norm: dict,
    *,
    limit: int = 200,
    seed: int = 42,
    temps: tuple[float, ...] = (0.0, 0.2, 0.3, 0.4, 0.5, 1.0),
    innovation_rhos: tuple[float, ...] = (0.0,),
    report: bool = False,
    selected_rows: list[dict] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    stats = collect_generated(
        rows,
        session,
        norm,
        limit=limit,
        seed=seed,
        temps=temps,
        innovation_rhos=innovation_rhos,
        selected_rows=selected_rows,
    )
    print(
        f"\nGenerated vs human on {stats['used']} stratified validation swipes "
        f"(frozen touch path, min_step_px={stats['step_px']})"
    )
    print("ar = autoregressive; tf = teacher-forced; t* = MDN temp; r* = innovation rho")
    for label in stats["labels"]:
        print(label)
        _print_mag_table(
            stats["n"][label],
            stats["buckets"][label],
            stats["first"][label],
            stats["extras"][label],
        )
    print("\npointwise |gen-human| after t0 (median)")
    print(f"{'mode':<12} {'cond':<10} {'|da| p50':>10} {'|dg| p50':>10}")
    for name, *_ in stats["modes"]:
        for cond in CONDITIONS:
            ae = sorted(stats["err"][name][cond]["acc"])
            ge = sorted(stats["err"][name][cond]["gyro"])
            if ae:
                print(f"{name:<12} {cond:<10} {_pct(ae, 50):>10.3f} {_pct(ge, 50):>10.4f}")
    _print_step_change_table(stats["labels"], stats["activity"])
    _print_persistence_table(stats["labels"], stats["persistence"])
    if stats["used"] < 300:
        print(
            f"\nWARNING: n={stats['used']} is diagnostic only; "
            "use --gen-limit 500 or more for model selection."
        )
    if report:
        _print_paste_report(
            stats["used"],
            stats["labels"],
            stats["buckets"],
            stats["err"],
            stats["extras"],
            stats["activity"],
        )
    return rollout_summary(stats)


def _print_paste_report(
    used: int,
    labels: tuple[str, ...],
    buckets: dict,
    err: dict,
    extras: dict,
    activity: dict,
) -> None:
    print("\n=== SENSOR EVAL REPORT (paste into chat) ===")
    print(
        f"target=delta  mdn_temp={model_config['mdn_temp']}  "
        f"ss_max={model_config['ss_max']}  ss_temp={model_config['ss_temp']}  "
        f"ss_unroll_hops={model_config.get('ss_unroll_hops', 1)}  "
        f"ss_target_clip_z={model_config.get('ss_target_clip_z', 0)}  n={used}"
    )
    print(
        f"{'mode':<12} {'cond':<10} {'|a| p50':>8} {'|a| p90':>8} "
        f"{'|g| p50':>8} {'|da|':>8} {'|dg|':>8} {'last/t0':>8}"
    )
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
    print("swipe-weighted temporal vector change")
    print(f"{'mode':<12} {'cond':<10} {'|delta a|':>10} {'|delta g|':>10}")
    for label in labels:
        for cond in CONDITIONS:
            dacc = sorted(activity[label][cond]["acc"])
            dgyro = sorted(activity[label][cond]["gyro"])
            if dacc:
                print(
                    f"{label:<12} {cond:<10} "
                    f"{_pct(dacc, 50):>10.4f} {_pct(dgyro, 50):>10.5f}"
                )
    print("=== END REPORT ===")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Print fused IMU stats by condition")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Cap jsonl rows loaded (debug)")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--norm", type=Path, default=None)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument(
        "--partition",
        choices=("train", "validation", "test"),
        default=None,
        help="Evaluate one user-grouped partition from the split manifest",
    )
    parser.add_argument("--gen-limit", type=int, default=200, help="Validation swipes to roll out")
    parser.add_argument(
        "--temps",
        default=None,
        help="Comma-separated MDN temps for autoregressive compare (0=mean, 1=full sample)",
    )
    parser.add_argument(
        "--rhos",
        default="0",
        help="Comma-separated innovation correlations in [0,1); 0 preserves independent draws",
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
    selected_rows = None
    if args.partition:
        split_path = args.split_manifest or HERE / model_config["split_manifest"]
        if not split_path.exists():
            raise SystemExit(f"Missing split manifest {split_path}")
        selected_rows = rows_for_partition(rows, load_user_split(split_path), args.partition)
        print(f"User-grouped {args.partition} partition: {len(selected_rows)} rows")
    summarize(selected_rows if selected_rows is not None else rows)
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
        temps = tuple(dict.fromkeys((0.0, t, 0.3, 0.4, 0.5, 1.0)))
    innovation_rhos = tuple(float(x) for x in args.rhos.split(",") if x.strip() != "")
    compare_generated(
        rows,
        load_session(model),
        load_norm(args.norm or default_norm_path()),
        limit=args.gen_limit,
        temps=temps,
        innovation_rhos=innovation_rhos,
        report=args.report,
        selected_rows=selected_rows,
    )


if __name__ == "__main__":
    main()
