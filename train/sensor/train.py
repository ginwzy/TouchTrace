"""Train a frozen-touch IMU LSTM+MDN on fused CSD4CA jsonl.

Input: previous accel/gyro (z-scored) + remaining-frame touch step + condition one-hot.
Output: current accel/gyro (z-scored). Magnetometer is not trained.

    python -m sensor.train
    python -m sensor.train --lite
    python -m sensor.train --epochs 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

HERE = Path(__file__).resolve().parent
_TRAIN_ROOT = HERE.parent
if str(_TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAIN_ROOT))

import numpy as np

from sensor.config import model_config
from sensor.features import (
    apply_sensor_norm,
    collect_init_by_condition,
    fit_sensor_norm,
    load_sensor_jsonl,
    load_sensor_sequences,
)
from touch.features import resolve_jsonl
from touch.train import (
    _configure_onednn,
    _log_first_batch,
    _resolve_loader_mode,
    _split_arrays,
    _split_sequences,
    build_mdn_lstm,
    log_train_devices,
    make_batch_sequence,
    make_prepad_sequence,
)


def build_sensor_model(lstm_units: int):
    return build_mdn_lstm(model_config, lstm_units)


def _resolve_data_path() -> Path:
    return resolve_jsonl(HERE, "sensor_data")


def _mask_weights(Y: np.ndarray, pad: float) -> np.ndarray:
    return (Y[:, :, 0] != pad).astype(np.float32)


def _zscore_list(xs, ys, mean, std, pad: float):
    out_x, out_y = [], []
    for x, y in zip(xs, ys):
        xn, yn = apply_sensor_norm(x[None], y[None], mean, std, pad)
        out_x.append(xn[0])
        out_y.append(yn[0])
    return out_x, out_y


def _write_norm(path: Path, mean, std, init_by_condition) -> None:
    payload = {
        "mean": [float(v) for v in mean],
        "std": [float(v) for v in std],
        "axes": ["ax", "ay", "az", "gx", "gy", "gz"],
        "init_by_condition": init_by_condition,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {path}")


def train(
    lite: bool = False,
    epochs: int | None = None,
    data: Path | None = None,
    prepad: bool = False,
    sequence: bool = False,
    min_step_px: float | None = None,
    no_augment: bool = False,
) -> Path:
    from tf_keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TerminateOnNaN

    loader_mode = _resolve_loader_mode(prepad, sequence)
    _configure_onednn(loader_mode)

    units = model_config["lstm_units_lite"] if lite else model_config["lstm_units"]
    weights_name = model_config["weights_lite" if lite else "weights"]
    prefix = Path(weights_name).stem
    data_path = Path(data) if data else _resolve_data_path()
    n_epochs = epochs if epochs is not None else model_config["epochs"]
    pad = model_config["pad"]
    batch_size = model_config["batch_size"]
    max_steps = model_config["max_steps"]
    step_px = model_config["min_step_px"] if min_step_px is None else min_step_px
    noise_std = None if no_augment else model_config["input_noise_std"]
    remaining_frame = bool(model_config.get("remaining_frame", True))
    rng = np.random.default_rng(42)

    log_train_devices(loader_mode)
    print(
        f"Parsing sensor data from {data_path} (max_steps={max_steps}, min_step_px={step_px}, "
        f"remaining_frame={remaining_frame}) ..."
    )
    if not data_path.exists():
        raise SystemExit(f"Missing {data_path}. Run: python convert_swipemotiondb.py --sensors")

    model = build_sensor_model(units)
    model.summary()

    callbacks = [
        TerminateOnNaN(),
        ModelCheckpoint(
            filepath=str(HERE / f"{prefix}_last.h5"),
            save_weights_only=True,
            save_freq="epoch",
            verbose=0,
        ),
        ModelCheckpoint(
            filepath=str(HERE / f"{prefix}_best.h5"),
            save_weights_only=True,
            save_best_only=True,
            monitor="val_loss",
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=model_config["reduce_lr_factor"],
            patience=model_config["reduce_lr_patience"],
            min_lr=model_config["min_lr"],
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=model_config["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    init_by_condition = collect_init_by_condition(data_path)

    if loader_mode == "prepad":
        X, Y = load_sensor_jsonl(
            data_path,
            pad=pad,
            max_steps=max_steps,
            min_step_px=step_px,
            remaining_frame=remaining_frame,
        )
        X_train, Y_train, X_val, Y_val = _split_arrays(X, Y, model_config["validation_split"])
        norm = fit_sensor_norm(Y_train, pad)
        X_train, Y_train = apply_sensor_norm(X_train, Y_train, norm["mean"], norm["std"], pad)
        X_val, Y_val = apply_sensor_norm(X_val, Y_val, norm["mean"], norm["std"], pad)
        W_train = _mask_weights(Y_train, pad)
        W_val = _mask_weights(Y_val, pad)
        valid_steps = np.sum(Y_train[:, :, 0] != pad, axis=1)
        print(
            f"Loaded {len(X)} paths (prepad). "
            f"train len min/mean/max={valid_steps.min()}/{valid_steps.mean():.1f}/{valid_steps.max()}"
        )
        print(f"IMU z-score mean={np.round(norm['mean'], 4).tolist()} std={np.round(norm['std'], 4).tolist()}")
        train_seq = make_prepad_sequence(
            X_train, Y_train, W_train, batch_size, pad, shuffle=True, noise_std=noise_std, rng=rng
        )
        val_seq = make_prepad_sequence(X_val, Y_val, W_val, batch_size, pad, shuffle=False, noise_std=None, rng=rng)
        _log_first_batch(train_seq, noise_std, False, "off", remaining_frame)
        model.fit(train_seq, epochs=n_epochs, validation_data=val_seq, callbacks=callbacks)
    else:
        xs, ys = load_sensor_sequences(
            data_path, max_steps=max_steps, min_step_px=step_px, remaining_frame=remaining_frame
        )
        x_train, y_train, x_val, y_val = _split_sequences(xs, ys, model_config["validation_split"])
        stacked = np.concatenate(y_train, axis=0)[None, ...]
        norm = fit_sensor_norm(stacked, pad)
        x_train, y_train = _zscore_list(x_train, y_train, norm["mean"], norm["std"], pad)
        x_val, y_val = _zscore_list(x_val, y_val, norm["mean"], norm["std"], pad)
        w_train = [np.ones(len(y), dtype=np.float32) for y in y_train]
        w_val = [np.ones(len(y), dtype=np.float32) for y in y_val]
        lengths = [len(x) for x in xs]
        print(
            f"Loaded {len(xs)} paths (sequence). "
            f"len min/mean/max={min(lengths)}/{sum(lengths) / len(lengths):.1f}/{max(lengths)}"
        )
        print(f"IMU z-score mean={np.round(norm['mean'], 4).tolist()} std={np.round(norm['std'], 4).tolist()}")
        train_seq = make_batch_sequence(
            x_train, y_train, w_train, batch_size, pad, True, noise_std, rng
        )
        val_seq = make_batch_sequence(x_val, y_val, w_val, batch_size, pad, False, None, rng)
        _log_first_batch(train_seq, noise_std, False, "off", remaining_frame)
        model.fit(train_seq, epochs=n_epochs, validation_data=val_seq, callbacks=callbacks)

    _write_norm(HERE / model_config["norm"], norm["mean"], norm["std"], init_by_condition)
    out = HERE / weights_name
    model.save_weights(str(out))
    print(f"\nTraining complete. Weights: {out}")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the touch-conditioned IMU LSTM+MDN")
    parser.add_argument("--lite", action="store_true", help="Train 2×64 LSTM instead of 2×128")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--data", type=Path, default=None, help="Path to sensor_data.jsonl[.gz]")
    loader = parser.add_mutually_exclusive_group()
    loader.add_argument("--prepad", action="store_true", help="Prepad to max_steps (default)")
    loader.add_argument("--sequence", action="store_true", help="Dynamic batch padding (Mac Metal fallback)")
    parser.add_argument("--min-step-px", type=float, default=None, help="Path subsample threshold")
    parser.add_argument("--no-augment", action="store_true", help="Disable IMU input noise")
    args = parser.parse_args(argv)
    train(
        lite=args.lite,
        epochs=args.epochs,
        data=args.data,
        prepad=args.prepad,
        sequence=args.sequence,
        min_step_px=args.min_step_px,
        no_augment=args.no_augment,
    )


if __name__ == "__main__":
    main()
