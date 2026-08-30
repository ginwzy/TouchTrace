"""Compare saved sensor checkpoints with fixed-seed autoregressive rollout metrics."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import tempfile
from pathlib import Path

from sensor.convert import export_onnx
from sensor.eval import (
    _mode_name,
    collect_generated,
    distribution_score,
    load_fused,
    rollout_summary,
)
from sensor.generate import default_data_path, default_norm_path, load_norm, load_session
from sensor.split import load_user_split, rows_for_partition


def checkpoint_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    path = Path(raw_path).expanduser()
    if not label or not raw_path.strip():
        raise argparse.ArgumentTypeError("checkpoint must be LABEL=PATH")
    return label, path


def _ratio(actual: dict, expected: dict, key: str) -> float:
    got = float(actual.get(key, float("nan")))
    want = float(expected.get(key, float("nan")))
    if not math.isfinite(got) or not math.isfinite(want) or want <= 0:
        return float("nan")
    return got / want


def aggregate_diagnostics(summary: dict, label: str) -> dict:
    gyro_ratio: dict[str, float] = {}
    swipe_mean_ratio: dict[str, float] = {}
    residual_ratio: dict[str, float] = {}
    ar_tf_gyro_ratio: dict[str, float] = {}
    temporal_errors: list[float] = []
    accel_p90_errors: list[float] = []
    drift_errors: list[float] = []
    for condition, real in summary.get("real", {}).items():
        generated = summary.get(label, {}).get(condition)
        if not generated:
            continue
        gyro_ratio[condition] = _ratio(generated, real, "gyro_p50")
        swipe_mean_ratio[condition] = _ratio(generated, real, "gyro_swipe_mean")
        residual_ratio[condition] = _ratio(generated, real, "gyro_residual")
        teacher_forced = summary.get("tf-mean", {}).get(condition, {})
        ar_tf_gyro_ratio[condition] = _ratio(generated, teacher_forced, "gyro_p50")
        for key in ("delta_acc", "delta_gyro"):
            ratio = _ratio(generated, real, key)
            if math.isfinite(ratio):
                temporal_errors.append(abs(ratio - 1.0))
        ratio = _ratio(generated, real, "acc_p90")
        if math.isfinite(ratio):
            accel_p90_errors.append(abs(ratio - 1.0))
        ratio = _ratio(generated, real, "drift")
        if math.isfinite(ratio):
            drift_errors.append(abs(ratio - 1.0))
    return {
        "distribution_score": distribution_score(summary, label),
        "gyro_p50_ratio": gyro_ratio,
        "gyro_swipe_mean_ratio": swipe_mean_ratio,
        "gyro_residual_ratio": residual_ratio,
        "ar_tf_gyro_ratio": ar_tf_gyro_ratio,
        "temporal_mare": sum(temporal_errors) / len(temporal_errors) if temporal_errors else float("inf"),
        "accel_p90_mare": sum(accel_p90_errors) / len(accel_p90_errors) if accel_p90_errors else float("inf"),
        "drift_mare": sum(drift_errors) / len(drift_errors) if drift_errors else float("inf"),
    }


def _minimum_ratio(candidate: dict, key: str) -> float:
    values = [
        float(value)
        for value in candidate.get(key, {}).values()
        if math.isfinite(float(value))
    ]
    return min(values) if values else float("-inf")


def _aggregate_ratio_maps(diagnostics: list[dict], key: str, *, conservative: bool) -> dict:
    conditions = sorted(
        set().union(*(set(item.get(key, {})) for item in diagnostics))
    )
    out = {}
    for condition in conditions:
        values = [
            float(item[key][condition])
            for item in diagnostics
            if condition in item.get(key, {}) and math.isfinite(float(item[key][condition]))
        ]
        if values:
            out[condition] = min(values) if conservative else sum(values) / len(values)
    return out


def aggregate_seed_diagnostics(diagnostics: list[dict]) -> dict:
    if not diagnostics:
        raise ValueError("at least one seed diagnostic is required")
    return {
        "distribution_score": _mean_finite(item["distribution_score"] for item in diagnostics),
        "gyro_p50_ratio": _aggregate_ratio_maps(
            diagnostics, "gyro_p50_ratio", conservative=True
        ),
        "ar_tf_gyro_ratio": _aggregate_ratio_maps(
            diagnostics, "ar_tf_gyro_ratio", conservative=True
        ),
        "gyro_swipe_mean_ratio": _aggregate_ratio_maps(
            diagnostics, "gyro_swipe_mean_ratio", conservative=False
        ),
        "gyro_residual_ratio": _aggregate_ratio_maps(
            diagnostics, "gyro_residual_ratio", conservative=False
        ),
        "temporal_mare": max(float(item["temporal_mare"]) for item in diagnostics),
        "accel_p90_mare": max(float(item["accel_p90_mare"]) for item in diagnostics),
        "drift_mare": max(float(item["drift_mare"]) for item in diagnostics),
    }


def candidate_is_feasible(
    candidate: dict,
    *,
    gyro_p50_min: float = 0.80,
    temporal_limit: float = 0.10,
    accel_p90_limit: float = 0.10,
    drift_limit: float = 0.10,
) -> bool:
    return (
        _minimum_ratio(candidate, "gyro_p50_ratio") >= gyro_p50_min
        and _minimum_ratio(candidate, "ar_tf_gyro_ratio") >= gyro_p50_min
        and float(candidate["temporal_mare"]) <= temporal_limit
        and float(candidate["accel_p90_mare"]) <= accel_p90_limit
        and float(candidate["drift_mare"]) <= drift_limit
    )


def _constraint_violation(candidate: dict) -> float:
    limits = {
        "temporal_mare": 0.10,
        "accel_p90_mare": 0.10,
        "drift_mare": 0.10,
    }
    violation = sum(max(0.0, float(candidate[key]) / limit - 1.0) for key, limit in limits.items())
    for ratio_key in ("gyro_p50_ratio", "ar_tf_gyro_ratio"):
        minimum_ratio = _minimum_ratio(candidate, ratio_key)
        if not math.isfinite(minimum_ratio):
            return float("inf")
        violation += max(0.0, (0.80 - minimum_ratio) / 0.80)
    return violation


def select_candidate(candidates: list[dict]) -> dict:
    feasible = [candidate for candidate in candidates if candidate_is_feasible(candidate)]
    if feasible:
        return min(feasible, key=lambda item: item["distribution_score"])
    return min(candidates, key=lambda item: (_constraint_violation(item), item["distribution_score"]))


def _finite_or_none(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _finite_or_none(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_or_none(item) for item in value]
    return value


def _mean_finite(values) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else float("nan")


def _markdown(results: list[dict]) -> str:
    lines = [
        "# Sensor checkpoint sweep",
        "",
        "| Checkpoint | Temp | Rho | Feasible | Score | Gyro p50 ratio | AR/TF gyro | Swipe-mean ratio | Temporal MARE | Accel p90 MARE | Drift MARE |",
        "| --- | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        best = result["best"]
        lines.append(
            "| {checkpoint} | {temp:.2f} | {rho:.2f} | {feasible} | {score:.4f} | {gyro:.1%} | "
            "{ar_tf:.1%} | {swipe:.1%} | {temporal:.1%} | {accel:.1%} | {drift:.1%} |".format(
                checkpoint=result["label"],
                temp=best["temp"],
                rho=best["rho"],
                feasible="yes" if candidate_is_feasible(best) else "no",
                score=best["distribution_score"],
                gyro=_mean_finite(best["gyro_p50_ratio"].values()),
                ar_tf=_mean_finite(best["ar_tf_gyro_ratio"].values()),
                swipe=_mean_finite(best["gyro_swipe_mean_ratio"].values()),
                temporal=best["temporal_mare"],
                accel=best["accel_p90_mare"],
                drift=best["drift_mare"],
            )
        )
    lines.extend([""])
    if any(min(result["used_by_seed"].values()) < 500 for result in results):
        lines.append("**Diagnostic only:** fewer than 500 swipes were evaluated for at least one seed.")
        lines.append("")
    lines.extend(
        [
            "Candidates require gyro p50 >= 80% of real and AR/TF gyro >= 80% in every "
            "condition for every seed, plus temporal, accel-p90, and drift MARE <= 10%.",
            "Among feasible candidates, lower mean distribution score is better; ratios should approach 100%.",
            "Every checkpoint uses the same validation rows and RNG seeds; hard gates retain the worst seed.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=checkpoint_spec,
        required=True,
        metavar="LABEL=PATH",
        help="Repeat for every H5 weights file to compare",
    )
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--norm", type=Path, default=None)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument(
        "--partition",
        choices=("train", "validation", "test"),
        default=None,
        help="Evaluate one user-grouped partition from a split manifest",
    )
    parser.add_argument("--temps", default="0.2,0.27,0.3,0.4")
    parser.add_argument(
        "--rhos",
        default="0",
        help="Comma-separated innovation correlations; 0 preserves independent MDN sampling",
    )
    parser.add_argument("--gen-limit", type=int, default=500)
    parser.add_argument(
        "--seeds",
        default="42,43,44",
        help="Comma-separated rollout seeds; hard gates use the worst seed",
    )
    parser.add_argument("--output", type=Path, required=True, help="JSON output path; Markdown is written alongside it")
    args = parser.parse_args(argv)

    checkpoints = args.checkpoint
    labels = [label for label, _ in checkpoints]
    if len(set(labels)) != len(labels):
        raise SystemExit("Checkpoint labels must be unique")
    for label, path in checkpoints:
        if not path.is_file():
            raise SystemExit(f"Missing checkpoint {label}: {path}")

    data_path = args.data or default_data_path()
    norm_path = args.norm or default_norm_path()
    rows = load_fused(data_path)
    selected_rows = None
    split_path = args.split_manifest
    if args.partition:
        if split_path is None or not split_path.exists():
            raise SystemExit("--partition requires an existing --split-manifest")
        selected_rows = rows_for_partition(rows, load_user_split(split_path), args.partition)
        print(f"User-grouped {args.partition} partition: {len(selected_rows)} rows")
    norm = load_norm(norm_path)
    temps = tuple(float(value) for value in args.temps.split(",") if value.strip())
    rhos = tuple(float(value) for value in args.rhos.split(",") if value.strip())
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    if not temps:
        raise SystemExit("At least one temperature is required")
    if not rhos:
        raise SystemExit("At least one innovation correlation is required")
    if not seeds:
        raise SystemExit("At least one rollout seed is required")
    if args.gen_limit < 500:
        print("WARNING: gen-limit < 500 is diagnostic only and must not select a release candidate")

    results: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="touchtrace-sensor-sweep-") as raw_dir:
        out_dir = Path(raw_dir)
        for index, (checkpoint_label, weights) in enumerate(checkpoints, start=1):
            print(f"[{index}/{len(checkpoints)}] {checkpoint_label}: {weights}", flush=True)
            onnx_path = out_dir / f"{checkpoint_label}.onnx"
            export_onnx(weights=weights, output=onnx_path, publish=False)
            session = load_session(onnx_path)
            summaries = {}
            used_by_seed = {}
            seed_diagnostics: dict[str, list[dict]] = {}
            for seed in seeds:
                stats = collect_generated(
                    rows,
                    session,
                    norm,
                    limit=args.gen_limit,
                    seed=seed,
                    temps=temps,
                    innovation_rhos=rhos,
                    selected_rows=selected_rows,
                )
                summary = rollout_summary(stats)
                summaries[str(seed)] = summary
                used_by_seed[str(seed)] = stats["used"]
                for temp in temps:
                    for rho in rhos:
                        if temp <= 0 and rho != 0:
                            continue
                        mode = _mode_name(temp, False, rho)
                        seed_diagnostics.setdefault(mode, []).append(
                            {"seed": seed, **aggregate_diagnostics(summary, mode)}
                        )
                del stats, summary
                gc.collect()

            candidates = []
            for temp in temps:
                for rho in rhos:
                    if temp <= 0 and rho != 0:
                        continue
                    mode = _mode_name(temp, False, rho)
                    per_seed = seed_diagnostics[mode]
                    diagnostics = aggregate_seed_diagnostics(per_seed)
                    candidates.append(
                        {
                            "temp": temp,
                            "rho": rho,
                            "mode": mode,
                            "seed_metrics": per_seed,
                            **diagnostics,
                        }
                    )
            best = select_candidate(candidates)
            result = {
                "label": checkpoint_label,
                "weights": str(weights.resolve()),
                "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                "used_by_seed": used_by_seed,
                "candidates": candidates,
                "best": best,
                "summaries": summaries,
            }
            results.append(result)
            print(
                f"  best temp={best['temp']:.2f} rho={best['rho']:.2f} "
                f"score={best['distribution_score']:.4f} temporal_mare={best['temporal_mare']:.2%}",
                flush=True,
            )
            del summaries, seed_diagnostics
            gc.collect()

    payload = {
        "data": str(data_path.resolve()),
        "norm": str(norm_path.resolve()),
        "seeds": seeds,
        "gen_limit": args.gen_limit,
        "temps": temps,
        "rhos": rhos,
        "split_manifest": str(split_path.resolve()) if split_path else None,
        "partition": args.partition,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_finite_or_none(payload), indent=2) + "\n", encoding="utf-8")
    markdown_path = args.output.with_suffix(".md")
    markdown = _markdown(results)
    markdown_path.write_text(markdown, encoding="utf-8")
    print("\n" + markdown)
    print(f"JSON: {args.output}")
    print(f"Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
