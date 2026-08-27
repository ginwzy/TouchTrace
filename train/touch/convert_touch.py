"""Export trained touch Keras weights to ONNX (MDN params, no TFP sampling layer)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from config_touch import model_config
from train_touch import build_touch_model


def export_onnx(lite: bool = False, weights: Path | None = None, output: Path | None = None) -> Path:
    import tensorflow as tf
    import tf2onnx
    import tf_keras as keras

    units = model_config["lstm_units_lite"] if lite else model_config["lstm_units"]
    if weights is not None:
        weights_path = Path(weights)
    else:
        fallback = HERE / model_config["weights_lite" if lite else "weights"]
        best = HERE / f"{fallback.stem}_best.h5"
        weights_path = best if best.exists() else fallback
    out_path = Path(output) if output else HERE / (
        "touch_lite.onnx" if lite else model_config["onnx_model"]
    )

    original_model = build_touch_model(units)
    original_model.load_weights(str(weights_path))
    export_model = keras.Model(inputs=original_model.inputs, outputs=original_model.layers[-2].output)

    print(f"Converting {weights_path} → {out_path}")
    spec = (tf.TensorSpec((1, None, model_config["input_dims"]), tf.float32, name="input"),)
    tf2onnx.convert.from_keras(export_model, input_signature=spec, output_path=str(out_path))
    print("Conversion complete.")
    return out_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export touch model weights to ONNX")
    parser.add_argument("--lite", action="store_true")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    export_onnx(lite=args.lite, weights=args.weights, output=args.output)


if __name__ == "__main__":
    main()
