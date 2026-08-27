"""Train the Phase 1 touch LSTM + MDN on CSD4CA jsonl.

Same architecture as the mouse pipeline: 5-d input, 3-d MDN output, 2×128 LSTM.
Sequences are truncated to max_steps (default 128).

Default: prepad once to max_steps and model.fit on fixed arrays (fast on CPU and CUDA).
Use --sequence only if training hangs on Mac Metal.

    python train_touch.py
    python train_touch.py --lite
    python train_touch.py --epochs 1
    python train_touch.py --sequence   # Mac Metal fallback only
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np

from config_touch import model_config
from features import load_trajectory_jsonl, load_trajectory_sequences

LoaderMode = str  # "prepad" | "sequence"


def mdn_loss(y_true, y_pred):
    import tensorflow as tf

    pad = model_config["pad"]
    mask = tf.cast(tf.math.not_equal(y_true[:, :, 0], pad), tf.float32)
    safe_y_true = tf.where(tf.expand_dims(mask, -1) == 1.0, y_true, tf.zeros_like(y_true))
    loss = -y_pred.log_prob(safe_y_true) * mask
    return tf.reduce_sum(loss) / (tf.reduce_sum(mask) + 1e-8)


def _configure_onednn(loader_mode: LoaderMode) -> None:
    if loader_mode == "prepad":
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")
    else:
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


def _resolve_loader_mode(prepad: bool, sequence: bool) -> LoaderMode:
    if prepad and sequence:
        raise SystemExit("Use at most one of --prepad and --sequence.")
    if sequence:
        return "sequence"
    return "prepad"


def log_train_devices(loader_mode: LoaderMode) -> None:
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    device = "GPU" if gpus else "CPU"
    print(f"TensorFlow {tf.__version__} device={device} GPUs={gpus} loader={loader_mode}")


def make_batch_sequence(xs, ys, batch_size: int, pad: float, shuffle: bool):
    import tf_keras

    class TouchBatchSequence(tf_keras.utils.Sequence):
        """Pads each batch to that batch's max length. Avoids tf.data + Metal hangs."""

        def __init__(self):
            self.xs = xs
            self.ys = ys
            self.batch_size = batch_size
            self.pad = pad
            self.shuffle = shuffle
            self.indices = np.arange(len(xs))
            self.on_epoch_end()

        def __len__(self) -> int:
            return math.ceil(len(self.xs) / self.batch_size)

        def on_epoch_end(self) -> None:
            if self.shuffle:
                np.random.shuffle(self.indices)

        def __getitem__(self, idx: int):
            batch_idx = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
            max_len = max(len(self.xs[i]) for i in batch_idx)
            x = np.full((len(batch_idx), max_len, self.xs[0].shape[-1]), self.pad, dtype=np.float32)
            y = np.full((len(batch_idx), max_len, self.ys[0].shape[-1]), self.pad, dtype=np.float32)
            for row, i in enumerate(batch_idx):
                n = len(self.xs[i])
                x[row, :n] = self.xs[i]
                y[row, :n] = self.ys[i]
            return x, y

    return TouchBatchSequence()


def build_touch_model(lstm_units: int):
    import tensorflow_probability as tfp
    import tf_keras
    from tf_keras.layers import LSTM, Dense, Input
    from tf_keras.models import Sequential

    # No Masking: Metal LSTM+Masking can hang. Post-padding is loss-masked instead.
    tfpl = tfp.layers
    model = Sequential(
        [
            Input(shape=(None, model_config["input_dims"])),
            LSTM(lstm_units, return_sequences=True),
            LSTM(lstm_units, return_sequences=True),
            Dense(
                int(
                    tfpl.MixtureNormal.params_size(
                        num_components=model_config["components"],
                        event_shape=model_config["output_dims"],
                    )
                )
            ),
            tfpl.MixtureNormal(
                num_components=model_config["components"],
                event_shape=model_config["output_dims"],
            ),
        ]
    )
    try:
        adam = tf_keras.optimizers.legacy.Adam
    except AttributeError:
        adam = tf_keras.optimizers.Adam
    model.compile(optimizer=adam(learning_rate=model_config["learning_rate"]), loss=mdn_loss)
    return model


def _train_val_indices(n: int, val_fraction: float, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    n_val = max(1, int(n * val_fraction))
    order = np.random.default_rng(seed).permutation(n)
    return order[n_val:], order[:n_val]


def _split_sequences(xs, ys, val_fraction: float, seed: int = 42):
    train_idx, val_idx = _train_val_indices(len(xs), val_fraction, seed)
    x_train = [xs[i] for i in train_idx]
    y_train = [ys[i] for i in train_idx]
    x_val = [xs[i] for i in val_idx]
    y_val = [ys[i] for i in val_idx]
    return x_train, y_train, x_val, y_val


def _split_arrays(X: np.ndarray, Y: np.ndarray, val_fraction: float, seed: int = 42):
    train_idx, val_idx = _train_val_indices(len(X), val_fraction, seed)
    return X[train_idx], Y[train_idx], X[val_idx], Y[val_idx]


def _resolve_data_path() -> Path:
    jsonl = HERE / "touch_data.jsonl"
    gz = HERE / "touch_data.jsonl.gz"
    if jsonl.exists():
        return jsonl
    return gz


def train(
    lite: bool = False,
    epochs: int | None = None,
    data: Path | None = None,
    prepad: bool = False,
    sequence: bool = False,
) -> Path:
    from tf_keras.callbacks import EarlyStopping, ModelCheckpoint

    loader_mode = _resolve_loader_mode(prepad, sequence)
    _configure_onednn(loader_mode)

    units = model_config["lstm_units_lite"] if lite else model_config["lstm_units"]
    weights_name = model_config["weights_lite"] if lite else model_config["weights"]
    data_path = Path(data) if data else _resolve_data_path()
    n_epochs = epochs if epochs is not None else model_config["epochs"]
    pad = model_config["pad"]
    batch_size = model_config["batch_size"]
    max_steps = model_config["max_steps"]

    log_train_devices(loader_mode)
    print(f"Parsing data from {data_path} (max_steps={max_steps}) ...")

    model = build_touch_model(units)
    model.summary()

    prefix = "model_lite" if lite else "model"
    callbacks = [
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
        EarlyStopping(
            monitor="val_loss",
            patience=model_config["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    if loader_mode == "prepad":
        X, Y = load_trajectory_jsonl(data_path, pad=pad, max_steps=max_steps)
        valid_steps = np.sum(Y[:, :, 0] != pad, axis=1)
        print(
            f"Loaded {len(X)} paths (prepad). "
            f"len min/mean/max={valid_steps.min()}/{valid_steps.mean():.1f}/{valid_steps.max()}"
        )
        X_train, Y_train, X_val, Y_val = _split_arrays(X, Y, model_config["validation_split"])
        print(f"train arrays {X_train.shape}, val arrays {X_val.shape}, batch_size={batch_size}")
        model.fit(
            X_train,
            Y_train,
            epochs=n_epochs,
            batch_size=batch_size,
            validation_data=(X_val, Y_val),
            callbacks=callbacks,
            shuffle=True,
        )
    else:
        xs, ys = load_trajectory_sequences(data_path, max_steps=max_steps)
        lengths = [len(x) for x in xs]
        print(
            f"Loaded {len(xs)} paths (sequence). "
            f"len min/mean/max={min(lengths)}/{sum(lengths) / len(lengths):.1f}/{max(lengths)}"
        )
        x_train, y_train, x_val, y_val = _split_sequences(xs, ys, model_config["validation_split"])
        train_seq = make_batch_sequence(x_train, y_train, batch_size, pad, shuffle=True)
        val_seq = make_batch_sequence(x_val, y_val, batch_size, pad, shuffle=False)
        xb, yb = train_seq[0]
        print(f"example batch shape={xb.shape} (dynamic time dim, cap {max_steps})")
        model.fit(
            train_seq,
            epochs=n_epochs,
            validation_data=val_seq,
            callbacks=callbacks,
        )

    out = HERE / weights_name
    model.save_weights(str(out))
    print(f"\nTraining complete. Weights: {out}")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the touch LSTM+MDN model")
    parser.add_argument("--lite", action="store_true", help="Train 2×64 LSTM instead of 2×128")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--data", type=Path, default=None, help="Path to touch_data.jsonl")
    loader = parser.add_mutually_exclusive_group()
    loader.add_argument(
        "--prepad",
        action="store_true",
        help="Prepad to max_steps (default; same as omitting both loader flags)",
    )
    loader.add_argument(
        "--sequence",
        action="store_true",
        help="Dynamic batch padding via Sequence (Mac Metal only if prepad hangs)",
    )
    args = parser.parse_args(argv)
    train(
        lite=args.lite,
        epochs=args.epochs,
        data=args.data,
        prepad=args.prepad,
        sequence=args.sequence,
    )


if __name__ == "__main__":
    main()
