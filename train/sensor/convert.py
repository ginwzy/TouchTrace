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


def export_onnx(lite: bool = False, weights: Path | None = None, output: Path | None = None) -> Path:
    return export_mdn_onnx(
        build_model=build_sensor_model,
        config=model_config,
        here=HERE,
        lite=lite,
        weights=weights,
        output=output,
        extra_files=(HERE / model_config["norm"],),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export sensor model weights to ONNX")
    parser.add_argument("--lite", action="store_true")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    export_onnx(lite=args.lite, weights=args.weights, output=args.output)


if __name__ == "__main__":
    main()
