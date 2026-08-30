"""Export trained sensor Keras weights to ONNX (MDN params, no TFP sampling layer)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_TRAIN_ROOT = HERE.parent
if str(_TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAIN_ROOT))

from sensor.config import model_config
from sensor.train import build_sensor_model
from touch.convert import export_mdn_onnx


def resolve_sensor_weights(here: Path, config: dict, lite: bool = False, weights: Path | None = None) -> Path:
    """Prefer the final saved run, then last, then hold-best."""
    if weights is not None:
        return Path(weights)
    fallback = here / config["weights_lite" if lite else "weights"]
    last = here / f"{fallback.stem}_last.h5"
    best = here / f"{fallback.stem}_best.h5"
    for cand in (fallback, last, best):
        if cand.exists():
            return cand
    return fallback


def export_onnx(
    lite: bool = False,
    weights: Path | None = None,
    output: Path | None = None,
    norm: Path | None = None,
    *,
    publish: bool = True,
) -> Path:
    norm_path = Path(norm) if norm is not None else HERE / model_config["norm"]
    return export_mdn_onnx(
        build_model=build_sensor_model,
        config=model_config,
        here=HERE,
        lite=lite,
        weights=resolve_sensor_weights(HERE, model_config, lite=lite, weights=weights),
        output=output,
        extra_files=(norm_path,),
        publish=publish,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export sensor model weights to ONNX")
    parser.add_argument("--lite", action="store_true")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--norm",
        type=Path,
        default=None,
        help="Normalization file paired with the selected weights",
    )
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="Keep the ONNX output isolated instead of copying release assets",
    )
    args = parser.parse_args(argv)
    export_onnx(
        lite=args.lite,
        weights=args.weights,
        output=args.output,
        norm=args.norm,
        publish=not args.no_publish,
    )


if __name__ == "__main__":
    main()
