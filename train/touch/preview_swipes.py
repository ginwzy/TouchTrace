"""Generate touch swipe trajectories from touch.onnx and plot them."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config_touch import model_config
from features import swipe_angle_bucket
from generate import GenerateOptions, TouchStep, generate_touch_path, load_session

SCREEN_W, SCREEN_H = 1080, 2142

DEFAULT_SWIPES = [
    ("~400px diagonal (typical)", (200, 900), (580, 1250)),
    ("~400px horizontal", (300, 1100), (700, 1100)),
    ("~400px vertical", (540, 800), (540, 1200)),
    ("short diagonal", (100, 800), (540, 1200)),
    ("long diagonal", (80, 1900), (950, 350)),
]


def _resolve_data_path() -> Path:
    gz = Path(__file__).resolve().parent / "touch_data.jsonl.gz"
    jsonl = Path(__file__).resolve().parent / "touch_data.jsonl"
    return gz if gz.exists() else jsonl


def _swipes_from_val(data_path: Path, seed: int = 42) -> list[tuple[str, tuple[float, float], tuple[float, float]]]:
    from eval_generate import _load_val_swipes

    swipes = _load_val_swipes(data_path, model_config["validation_split"], seed=seed, limit=None)
    scored = []
    for start, end, _path in swipes:
        dx, dy = end[0] - start[0], end[1] - start[1]
        scored.append((swipe_angle_bucket(dx, dy), math.hypot(dx, dy), start, end))

    def closest(bucket: str, target_len: float):
        cands = [s for s in scored if s[0] == bucket] or scored
        return min(cands, key=lambda s: abs(s[1] - target_len))

    picks = [
        ("val vertical ~400px", closest("V", 400)),
        ("val diagonal ~400px", closest("D", 400)),
        ("val horizontal ~400px", closest("H", 400)),
        ("val short", closest("V", 150)),
        ("val long diagonal", closest("D", 900)),
    ]
    return [(label, item[2], item[3]) for label, item in picks]


def plot_swipes(
    groups: list[tuple[str, tuple[float, float], tuple[float, float], list[list[TouchStep]]]],
    out_path: Path,
    title: str,
) -> None:
    n = len(groups)
    cols = 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    axes_flat = np.atleast_1d(axes).flatten()
    colors = plt.cm.tab10.colors

    for ax, (label, start, end, paths) in zip(axes_flat, groups):
        for i, path in enumerate(paths):
            xs = [p.x for p in path]
            ys = [p.y for p in path]
            ax.plot(xs, ys, "-", color=colors[i % len(colors)], alpha=0.85, linewidth=1.5, label=f"sample {i + 1}")

        ax.scatter([start[0]], [start[1]], c="limegreen", s=80, marker="o", edgecolors="black", zorder=6)
        ax.scatter([end[0]], [end[1]], c="red", s=100, marker="*", edgecolors="black", zorder=6)
        ax.set_xlim(0, SCREEN_W)
        ax.set_ylim(SCREEN_H, 0)
        ax.set_aspect("equal")
        ax.set_title(label)
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    for ax in axes_flat[len(groups) :]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and plot touch swipes from touch.onnx")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/home/users/nate/Downloads/train/new/touch.onnx"),
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--raw", action="store_true", help="Disable guided steering and smoothing")
    parser.add_argument(
        "--noback",
        action="store_true",
        help="Raw MDN but redirect reverse steps toward the target",
    )
    parser.add_argument(
        "--from-data",
        action="store_true",
        help="Plot real validation start/end pairs instead of the fixed OOD swipes",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or args.model.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.noback:
        opts = GenerateOptions(guided=False, smooth=False, no_backtrack=True, seed=args.seed)
        mode = "raw + no-backtrack"
        suffix = "swipes_noback.png"
    elif args.raw:
        opts = GenerateOptions(guided=False, smooth=False, seed=args.seed)
        mode = "raw MDN"
        suffix = "swipes_raw.png"
    else:
        opts = GenerateOptions(smooth=True, seed=args.seed)
        mode = "guided + smooth"
        suffix = "swipes_guided.png"

    swipes = _swipes_from_val(_resolve_data_path()) if args.from_data else DEFAULT_SWIPES
    if args.from_data:
        suffix = suffix.replace(".png", "_val.png")

    session = load_session(args.model)
    groups = []

    for idx, (label, start, end) in enumerate(swipes):
        paths = []
        for s in range(args.samples):
            rng = np.random.default_rng(args.seed + idx * 100 + s)
            paths.append(generate_touch_path(session, start, end, opts, rng))
        groups.append((label, start, end, paths))
        p0 = paths[0]
        print(f"{label}: {len(p0)} points, {p0[-1].t:.0f} ms")

    grid_path = out_dir / suffix
    plot_swipes(groups, grid_path, f"TouchTrace generated swipes ({mode})")
    print(f"\nSaved: {grid_path}")


if __name__ == "__main__":
    main()
