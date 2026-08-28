"""Export trained touch Keras weights to ONNX (MDN params, no TFP sampling layer)."""

from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

HERE = Path(__file__).resolve().parent
_TRAIN_ROOT = HERE.parent
if str(_TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAIN_ROOT))

from touch.config import model_config
from touch.generate import inference_public_onnx
from touch.train import build_touch_model


def export_mdn_onnx(
    *,
    build_model: Callable[[int], object],
    config: dict,
    here: Path,
    lite: bool = False,
    weights: Path | None = None,
    output: Path | None = None,
    extra_files: Sequence[Path] = (),
) -> Path:
    import tensorflow as tf
    import tf2onnx
    import tf_keras as keras

    units = config["lstm_units_lite"] if lite else config["lstm_units"]
    if weights is not None:
        weights_path = Path(weights)
    else:
        fallback = here / config["weights_lite" if lite else "weights"]
        best = here / f"{fallback.stem}_best.h5"
        weights_path = best if best.exists() else fallback
    default_name = f"{Path(config['onnx_model']).stem}_lite.onnx" if lite else config["onnx_model"]
    out_path = Path(output) if output else here / default_name

    original_model = build_model(units)
    original_model.load_weights(str(weights_path))
    export_model = keras.Model(inputs=original_model.inputs, outputs=original_model.layers[-2].output)

    print(f"Converting {weights_path} → {out_path}")
    spec = (tf.TensorSpec((1, None, config["input_dims"]), tf.float32, name="input"),)
    tf2onnx.convert.from_keras(export_model, input_signature=spec, output_path=str(out_path))
    print("Conversion complete.")
    public = inference_public_onnx(out_path.name)
    if public.parent.is_dir() and public.resolve() != out_path.resolve():
        shutil.copy2(out_path, public)
        print(f"Copied to {public}")
        for src in extra_files:
            if src.exists():
                shutil.copy2(src, public.parent / src.name)
                print(f"Copied {src.name} to {public.parent}")
    return out_path


def export_onnx(lite: bool = False, weights: Path | None = None, output: Path | None = None) -> Path:
    return export_mdn_onnx(
        build_model=build_touch_model,
        config=model_config,
        here=HERE,
        lite=lite,
        weights=weights,
        output=output,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export touch model weights to ONNX")
    parser.add_argument("--lite", action="store_true")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    export_onnx(lite=args.lite, weights=args.weights, output=args.output)


if __name__ == "__main__":
    main()
