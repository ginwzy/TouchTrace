"""Plot real vs generated IMU along frozen validation touch paths."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
_TRAIN_ROOT = HERE.parent
if str(_TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAIN_ROOT))

from sensor.config import model_config
from sensor.eval import default_sensor_data_path, load_fused, val_rows
from sensor.features import CONDITIONS, subsample_paired
from sensor.generate import (
    default_norm_path,
    default_onnx_path,
    generate_sensor_along_path,
    load_norm,
    load_session,
    sensor_mags,
)


def _pick_examples(rows: list[dict], min_len: int = 12) -> dict[str, dict]:
    picked: dict[str, dict] = {}
    for row in rows:
        cond = str((row.get("meta") or {}).get("condition") or "")
        if cond not in CONDITIONS or cond in picked:
            continue
        path, sensors = subsample_paired(
            row.get("path") or [],
            row.get("sensors") or [],
            min_step_px=float(model_config["min_step_px"]),
        )
        if len(path) < min_len or len(path) != len(sensors):
            continue
        picked[cond] = {**row, "path": path, "sensors": sensors}
        if len(picked) == len(CONDITIONS):
            break
    return picked


def _plot_mags(
    examples: dict[str, dict],
    generated: dict[str, list[list[dict]]],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(len(CONDITIONS), 2, figsize=(12, 9), sharex=False)
    for row, cond in enumerate(CONDITIONS):
        if cond not in examples:
            axes[row, 0].set_visible(False)
            axes[row, 1].set_visible(False)
            continue
        real = examples[cond]["sensors"]
        t_r, a_r, g_r = sensor_mags(real)
        ax_a, ax_g = axes[row]
        ax_a.plot(t_r, a_r, color="black", lw=1.6, label="human")
        ax_g.plot(t_r, g_r, color="black", lw=1.6, label="human")
        for i, gen in enumerate(generated[cond]):
            t_g, a_g, g_g = sensor_mags(gen)
            ax_a.plot(t_g, a_g, alpha=0.75, lw=1.2, label=f"gen {i + 1}")
            ax_g.plot(t_g, g_g, alpha=0.75, lw=1.2, label=f"gen {i + 1}")
        ax_a.set_title(f"{cond}  |a|")
        ax_g.set_title(f"{cond}  |gyro|")
        ax_a.set_ylabel("m/s²")
        ax_g.set_ylabel("rad/s")
        ax_a.grid(True, alpha=0.3)
        ax_g.grid(True, alpha=0.3)
        if row == 0:
            ax_a.legend(fontsize=8)
    axes[-1, 0].set_xlabel("t (ms)")
    axes[-1, 1].set_xlabel("t (ms)")
    fig.suptitle("IMU magnitude along frozen touch paths", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_diag(seqs: dict[str, list[dict]], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    styles = {
        "human": ("black", 1.8, "-"),
        "ar-mean": ("C0", 1.4, "--"),
        "ar-sample": ("C3", 1.2, ":"),
        "tf-mean": ("C1", 1.3, "--"),
    }
    default_style = ("C2", 1.4, "-")
    for ax, key, ylabel in (
        (axes[0], "acc", "|a| (m/s²)"),
        (axes[1], "gyro", "|gyro| (rad/s)"),
    ):
        for name, seq in seqs.items():
            ts, acc, gyro = sensor_mags(seq)
            ys = acc if key == "acc" else gyro
            color, lw, ls = styles.get(name, default_style)
            ax.plot(ts, ys, color=color, lw=lw, ls=ls, label=name)
        ax.set_xlabel("t (ms)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Seated diagnostic: AR mean / temp / full sample vs teacher-forced mean", y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_channels(example: dict, generated: list[dict], out_path: Path, title: str) -> None:
    labels = ["ax", "ay", "az", "gx", "gy", "gz"]
    real = example["sensors"]
    t_r = [float(s["timestamp"]) for s in real]
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    for i, ax in enumerate(axes.flatten()):
        key = "accel" if i < 3 else "gyro"
        idx = i if i < 3 else i - 3
        ax.plot(t_r, [s[key][idx] for s in real], color="black", lw=1.6, label="human")
        ax.plot(
            [float(s["timestamp"]) for s in generated],
            [s[key][idx] for s in generated],
            alpha=0.85,
            lw=1.2,
            label="generated",
        )
        ax.set_title(labels[i])
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    axes[-1, 0].set_xlabel("t (ms)")
    axes[-1, 1].set_xlabel("t (ms)")
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_hists(
    real_by_cond: dict[str, dict[str, list[float]]],
    gen_by_cond: dict[str, dict[str, list[float]]],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(len(CONDITIONS), 2, figsize=(12, 9), sharex=False)
    for row, cond in enumerate(CONDITIONS):
        for col, key in enumerate(("acc", "gyro")):
            ax = axes[row, col]
            real = real_by_cond.get(cond, {}).get(key) or []
            gen = gen_by_cond.get(cond, {}).get(key) or []
            if real:
                ax.hist(real, bins=40, density=True, alpha=0.55, color="black", label="human")
            if gen:
                ax.hist(gen, bins=40, density=True, alpha=0.55, color="C0", label="generated")
            ax.set_title(f"{cond}  {'|a|' if key == 'acc' else '|gyro|'}")
            ax.grid(True, alpha=0.3)
            if row == 0 and col == 1:
                ax.legend(fontsize=8)
    fig.suptitle("IMU magnitude distributions (validation swipes)", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot real vs generated IMU")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--norm", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--temp",
        type=float,
        default=None,
        help="MDN temp (0=mean, 1=full sample). Default: model_config mdn_temp",
    )
    args = parser.parse_args(argv)

    model = args.model or default_onnx_path()
    data = args.data or default_sensor_data_path()
    if not model.exists():
        raise SystemExit(f"Missing {model}")
    out_dir = args.out_dir or HERE / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    temp = float(model_config["mdn_temp"] if args.temp is None else args.temp)

    rows = val_rows(load_fused(data), model_config["validation_split"], seed=args.seed, limit=args.limit)
    examples = _pick_examples(rows)
    if not examples:
        raise SystemExit("No validation swipes long enough to plot")

    session = load_session(model)
    norm = load_norm(args.norm or default_norm_path())

    def rollout(path, sensors, cond, rng, *, t=temp, teacher=False):
        imu0 = sensors[0]["accel"] + sensors[0]["gyro"]
        return generate_sensor_along_path(
            session,
            path,
            imu0,
            cond,
            norm,
            rng,
            temp=t,
            teacher_sensors=sensors if teacher else None,
        )

    generated: dict[str, list[list[dict]]] = {}
    for i, (cond, row) in enumerate(examples.items()):
        gens = [
            rollout(row["path"], row["sensors"], cond, np.random.default_rng(args.seed + i * 100 + s))
            for s in range(args.samples)
        ]
        generated[cond] = gens
        print(f"{cond}: {len(row['path'])} points, {row['path'][-1]['timestamp']:.0f} ms, temp={temp:g}")

    mag_path = out_dir / "imu_mags.png"
    _plot_mags(examples, generated, mag_path)
    print(f"Saved {mag_path}")

    if "seated" in examples:
        ch_path = out_dir / "imu_channels_seated.png"
        _plot_channels(
            examples["seated"],
            generated["seated"][0],
            ch_path,
            f"Seated IMU channels (human vs AR temp={temp:g})",
        )
        print(f"Saved {ch_path}")
        row = examples["seated"]
        rng = np.random.default_rng(args.seed)
        diag = {"human": row["sensors"]}
        temp_name = f"ar-t{temp:g}"
        for name, t, teacher in (
            ("ar-mean", 0.0, False),
            (temp_name, temp, False),
            ("ar-sample", 1.0, False),
            ("tf-mean", 0.0, True),
        ):
            diag[name] = rollout(row["path"], row["sensors"], "seated", rng, t=t, teacher=teacher)
        diag_path = out_dir / "imu_diag_seated.png"
        _plot_diag(diag, diag_path)
        print(f"Saved {diag_path}")

    real_by: dict[str, dict[str, list[float]]] = {c: {"acc": [], "gyro": []} for c in CONDITIONS}
    gen_by: dict[str, dict[str, list[float]]] = {c: {"acc": [], "gyro": []} for c in CONDITIONS}
    for j, row in enumerate(rows):
        cond = str((row.get("meta") or {}).get("condition") or "")
        if cond not in CONDITIONS:
            continue
        path, sensors = subsample_paired(
            row.get("path") or [],
            row.get("sensors") or [],
            min_step_px=float(model_config["min_step_px"]),
        )
        if len(path) < 2 or len(path) != len(sensors):
            continue
        gen = rollout(path, sensors, cond, np.random.default_rng(args.seed + 1000 + j))
        _, a_r, g_r = sensor_mags(sensors)
        _, a_g, g_g = sensor_mags(gen)
        real_by[cond]["acc"].extend(a_r)
        real_by[cond]["gyro"].extend(g_r)
        gen_by[cond]["acc"].extend(a_g)
        gen_by[cond]["gyro"].extend(g_g)

    hist_path = out_dir / "imu_hist.png"
    _plot_hists(real_by, gen_by, hist_path)
    print(f"Saved {hist_path}")


if __name__ == "__main__":
    main()
