"""Train a frozen-touch IMU LSTM+MDN on fused CSD4CA jsonl.

Input: previous accel/gyro (z-scored) + remaining-frame touch step + condition one-hot.
Output: current − previous accel/gyro (z-scored ΔIMU). Magnetometer is not trained.

    python -m sensor.train
    python -m sensor.train --lite
    python -m sensor.train --epochs 1
    python -m sensor.train --no-ss
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
    abs_z_from_delta_z,
    apply_sensor_norm,
    collect_init_by_condition,
    delta_z_from_abs_z,
    fit_sensor_norm,
    load_sensor_jsonl,
    load_sensor_jsonl_grouped,
    load_sensor_sequences,
    load_sensor_sequences_grouped,
)
from sensor.generate import mix_mdn_draw
from sensor.split import (
    build_user_split,
    load_user_split,
    partition_indices,
    write_user_split,
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


def make_sensor_mdn_loss(pad: float, nll_clip: float, hybrid_weight: float):
    import tensorflow as tf

    def sensor_mdn_loss(y_true, distribution):
        natural = y_true[..., :6]
        mask = tf.cast(tf.math.not_equal(natural[:, :, 0], pad), tf.float32)
        safe_natural = tf.where(mask[..., None] == 1.0, natural, tf.zeros_like(natural))
        nll = -distribution.log_prob(safe_natural)
        nll = tf.where(tf.math.is_finite(nll), nll, tf.ones_like(nll) * nll_clip)
        nll = tf.clip_by_value(nll, 0.0, nll_clip)
        loss = tf.reduce_sum(nll * mask) / (tf.reduce_sum(mask) + 1e-8)

        width = y_true.shape[-1]
        if width is not None and int(width) >= 13 and hybrid_weight > 0:
            correction = y_true[..., 6:12]
            aux_mask = tf.cast(y_true[..., 12], tf.float32) * mask
            safe_correction = tf.where(
                aux_mask[..., None] > 0,
                correction,
                tf.zeros_like(correction),
            )
            error = distribution.mean() - safe_correction
            absolute = tf.abs(error)
            quadratic = tf.minimum(absolute, 1.0)
            huber = tf.reduce_mean(0.5 * tf.square(quadratic) + (absolute - quadratic), axis=-1)
            loss += float(hybrid_weight) * tf.reduce_sum(huber * aux_mask) / (
                tf.reduce_sum(mask) + 1e-8
            )
        return loss

    return sensor_mdn_loss


def build_sensor_model(lstm_units: int):
    model = build_mdn_lstm(model_config, lstm_units)
    model.compile(
        optimizer=model.optimizer,
        loss=make_sensor_mdn_loss(
            float(model_config["pad"]),
            float(model_config["nll_clip"]),
            float(model_config["ss_hybrid_weight"]),
        ),
    )
    return model


def _resolve_data_path() -> Path:
    return resolve_jsonl(HERE, "sensor_data")


def _mask_weights(Y: np.ndarray, pad: float) -> np.ndarray:
    return (Y[:, :, 0] != pad).astype(np.float32)


def load_or_build_user_split(
    user_ids,
    manifest_path: Path,
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict:
    observed = {str(user_id) for user_id in user_ids if str(user_id).strip()}
    if manifest_path.exists():
        split = load_user_split(manifest_path)
        assigned = set().union(
            *(set(split[f"{partition}_users"]) for partition in ("train", "validation", "test"))
        )
        if assigned != observed:
            missing = sorted(observed - assigned)
            stale = sorted(assigned - observed)
            raise ValueError(
                f"split manifest user set differs from data; missing={missing}, stale={stale}"
            )
        return split
    return build_user_split(
        user_ids,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )


def scheduled_sampling_prob(epoch: int, warmup: int, ramp: int, ss_max: float) -> float:
    if ss_max <= 0 or epoch < warmup:
        return 0.0
    if ramp <= 0:
        return float(ss_max)
    return float(min(ss_max, ss_max * (epoch - warmup + 1) / ramp))


def ss_schedule_epochs(warmup: int, ramp: int, total: int) -> tuple[int, int, int]:
    """Exclusive Keras `epochs=` ends for TF warmup, SS ramp, and max-p hold."""
    total = max(total, 0)
    warmup_end = max(0, min(warmup, total))
    ramp_end = max(warmup_end, min(warmup_end + max(ramp, 0), total))
    return warmup_end, ramp_end, total


def ss_checkpoint_tag(prob: float) -> str:
    return f"p{int(round(float(prob) * 100)):03d}"


def fixed_ar_validation_arrays(
    xs,
    ys,
    pad: float,
    max_paths: int,
    max_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Small fixed validation batch for deterministic closed-loop scoring."""
    n = min(len(xs), max(0, int(max_paths)))
    if n == 0:
        return (
            np.zeros((0, 0, model_config["input_dims"]), dtype=np.float32),
            np.zeros((0, 0, model_config["output_dims"]), dtype=np.float32),
        )
    steps = max(1, int(max_steps))
    if isinstance(xs, np.ndarray):
        return (
            np.array(xs[:n, :steps], dtype=np.float32, copy=True),
            np.array(ys[:n, :steps], dtype=np.float32, copy=True),
        )
    width = min(steps, max(len(xs[i]) for i in range(n)))
    x = np.full((n, width, xs[0].shape[-1]), pad, dtype=np.float32)
    y = np.full((n, width, ys[0].shape[-1]), pad, dtype=np.float32)
    for i in range(n):
        size = min(width, len(xs[i]))
        x[i, :size] = xs[i][:size]
        y[i, :size] = ys[i][:size]
    return x, y


def _as_numpy(value) -> np.ndarray:
    return np.asarray(value.numpy() if hasattr(value, "numpy") else value, dtype=np.float32)


def autoregressive_mean_mae_z(
    model,
    x: np.ndarray,
    y: np.ndarray,
    norm: dict,
    pad: float,
) -> float:
    """Mean absolute state error from a deterministic mixture-mean rollout."""
    if len(x) == 0 or x.shape[1] == 0:
        return float("inf")
    source_x = np.asarray(x, dtype=np.float32)
    source_y = np.asarray(y, dtype=np.float32)
    rolled = np.array(source_x, copy=True)
    valid = source_y[:, :, 0] != pad
    human_next_z = abs_z_from_delta_z(source_x[..., :6], source_y, norm)
    errors: list[np.ndarray] = []
    for step in range(source_x.shape[1]):
        active = valid[:, step]
        if not active.any():
            break
        distribution = model(rolled[:, : step + 1], training=False)
        delta_z = _as_numpy(distribution.mean())[:, -1, :6]
        predicted_next_z = abs_z_from_delta_z(rolled[:, step, :6], delta_z, norm)
        errors.append(np.abs(predicted_next_z[active] - human_next_z[active, step]))
        if step + 1 < source_x.shape[1]:
            carry = active & valid[:, step + 1]
            rolled[carry, step + 1, :6] = predicted_next_z[carry]
    if not errors:
        return float("inf")
    return float(np.concatenate([err.reshape(-1) for err in errors]).mean())


def mix_scheduled_imu(
    x: np.ndarray,
    y: np.ndarray,
    pred_imu: np.ndarray,
    pad: float,
    prob: float,
    rng: np.random.Generator,
    use: np.ndarray | None = None,
) -> np.ndarray:
    """Replace some next-step imu_prev channels with the model's previous prediction.

    X[t, :6] conditions Y[t]. Predictions at t-1 therefore land on X[t, :6].
    """
    mask = y[:, :, 0] != pad
    if use is None:
        if prob <= 0:
            return x
        use = rng.random(mask.shape) < prob
    use = np.array(use, dtype=bool, copy=True)
    use[:, 0] = False
    use[:, 1:] &= mask[:, :-1] & mask[:, 1:]
    if not use.any():
        return x
    x_mix = np.array(x, dtype=np.float32, copy=True)
    prev = np.zeros_like(x_mix[..., :6])
    prev[:, 1:] = pred_imu[:, :-1, :6]
    x_mix[..., :6] = np.where(use[..., None], prev, x_mix[..., :6])
    return x_mix


def unroll_scheduled_imu(
    x: np.ndarray,
    y: np.ndarray,
    pad: float,
    prob: float,
    rng: np.random.Generator,
    predict_delta_z,
    norm: dict,
    hops: int,
    target_clip_z: float | None = None,
    diagnostics: dict | None = None,
    target_mode: str = "correction",
) -> tuple[np.ndarray, np.ndarray]:
    """Mix closed-loop inputs and construct the selected scheduled-sampling target."""
    if target_mode not in {"correction", "natural", "hybrid"}:
        raise ValueError(f"unsupported scheduled-sampling target mode: {target_mode}")
    hops = int(hops)
    if prob <= 0 or hops <= 0:
        return x, y
    mask = y[:, :, 0] != pad
    use = rng.random(mask.shape) < prob
    use[:, 0] = False
    use[:, 1:] &= mask[:, :-1] & mask[:, 1:]
    if not use.any():
        return x, y
    human_next_z = abs_z_from_delta_z(x[..., :6], y, norm)
    x_work = x
    for _ in range(hops):
        delta_z = predict_delta_z(x_work)
        pred_abs_z = abs_z_from_delta_z(x_work[..., :6], delta_z, norm)
        x_work = mix_scheduled_imu(x_work, y, pred_abs_z, pad, prob, rng, use=use)

    correction_z = delta_z_from_abs_z(x_work[..., :6], human_next_z, norm)
    selected = correction_z[use]
    clip = float(target_clip_z) if target_clip_z is not None else 0.0
    clipped_values = 0
    if clip > 0:
        clipped_values = int(np.count_nonzero(np.abs(selected) > clip))
        correction_z = np.clip(correction_z, -clip, clip)
    if diagnostics is not None:
        diagnostics["target_mode"] = target_mode
        diagnostics["target_values"] = diagnostics.get("target_values", 0) + int(selected.size)
        diagnostics["clipped_values"] = diagnostics.get("clipped_values", 0) + clipped_values
        max_abs = float(np.max(np.abs(selected))) if selected.size else 0.0
        diagnostics["max_abs_before_clip"] = max(
            float(diagnostics.get("max_abs_before_clip", 0.0)), max_abs
        )
    if target_mode == "correction":
        y_work = np.where(use[..., None], correction_z, y)
    elif target_mode == "natural":
        y_work = y
    else:
        auxiliary = np.where(use[..., None], correction_z, y)
        y_work = np.concatenate((y, auxiliary, use[..., None].astype(np.float32)), axis=-1)
    return x_work, y_work.astype(np.float32, copy=False)


def wrap_scheduled_sampling(
    inner,
    model,
    pad: float,
    state: dict,
    rng: np.random.Generator,
    norm: dict,
    ss_temp: float,
    hops: int = 1,
    target_clip_z: float | None = None,
    target_mode: str = "correction",
):
    import tf_keras

    diagnostics: dict[str, float] = {}

    def predict_delta_z(x_in):
        dist = model(x_in, training=False)
        mean = _as_numpy(dist.mean())
        if ss_temp <= 0:
            return mean
        return mix_mdn_draw(mean, _as_numpy(dist.sample()), ss_temp)

    class ScheduledSamplingSequence(tf_keras.utils.Sequence):
        def __len__(self):
            return len(inner)

        def on_epoch_end(self):
            inner.on_epoch_end()
            total = int(diagnostics.get("target_values", 0))
            if total:
                clipped = int(diagnostics.get("clipped_values", 0))
                print(
                    f"ss {diagnostics.get('target_mode', target_mode)} p={float(state['p']):.3f}: "
                    f"clipped={clipped / total:.2%} "
                    f"max|z|={float(diagnostics.get('max_abs_before_clip', 0.0)):.2f} "
                    f"limit={target_clip_z}"
                )
            diagnostics.clear()

        def __getitem__(self, idx: int):
            x, y, w = inner[idx]
            x_mix, y_mix = unroll_scheduled_imu(
                x,
                y,
                pad,
                float(state["p"]),
                rng,
                predict_delta_z,
                norm,
                hops,
                target_clip_z=target_clip_z,
                diagnostics=diagnostics,
                target_mode=target_mode,
            )
            return x_mix, y_mix, w

    return ScheduledSamplingSequence()


def _zscore_list(xs, ys, norm, pad: float):
    out_x, out_y = [], []
    for x, y in zip(xs, ys):
        xn, yn = apply_sensor_norm(x[None], y[None], norm, pad)
        out_x.append(xn[0])
        out_y.append(yn[0])
    return out_x, out_y


def _write_norm(path: Path, norm: dict, init_by_condition) -> None:
    payload = {
        **norm,
        "init_by_condition": init_by_condition,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {path}")


def make_ar_validation_callback(
    x,
    y,
    norm: dict,
    pad: float,
    abort_threshold: float | None = None,
):
    import tf_keras

    class ARValidation(tf_keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            score = autoregressive_mean_mae_z(self.model, x, y, norm, pad)
            if logs is not None:
                logs["val_ar_mae_z"] = score
            print(f"val_ar_mae_z: {score:.5f}")
            if abort_threshold is not None and (
                not np.isfinite(score) or score > float(abort_threshold)
            ):
                print(
                    f"Stopping: val_ar_mae_z={score:.5f} exceeds stability guard "
                    f"{float(abort_threshold):.5f}"
                )
                self.model.stop_training = True

    return ARValidation()


def make_ss_milestone_callback(prefix: str, state: dict, probabilities):
    import tf_keras

    class SSMilestones(tf_keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.pending = sorted({float(p) for p in probabilities if 0 < float(p) <= 1.0})

        def on_epoch_end(self, epoch, logs=None):
            current = float(state["p"])
            reached = [prob for prob in self.pending if current + 1e-9 >= prob]
            for prob in reached:
                path = HERE / f"{prefix}_ss_{ss_checkpoint_tag(prob)}.h5"
                self.model.save_weights(str(path))
                print(f"Saved SS milestone p>={prob:g}: {path}")
                self.pending.remove(prob)

    return SSMilestones()


def make_periodic_checkpoint_callback(prefix: str, interval: int):
    import tf_keras

    class PeriodicCheckpoint(tf_keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            completed = int(epoch) + 1
            if completed % int(interval) != 0:
                return
            path = HERE / f"{prefix}_candidate_e{completed:03d}.h5"
            self.model.save_weights(str(path))
            print(f"Saved rollout-selection candidate: {path}")

    return PeriodicCheckpoint()


def _make_callbacks(
    prefix: Path,
    *,
    monitor: str = "val_loss",
    save_best: bool = True,
    early_stop: bool = False,
    reduce_lr: bool = True,
    ss_state: dict | None = None,
    ar_validation_data: tuple | None = None,
    ss_milestone_probs=(),
    initial_best: float | None = None,
    restore_best_weights: bool = True,
    periodic_interval: int | None = None,
):
    import tf_keras
    from tf_keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TerminateOnNaN

    class _SSProb(tf_keras.callbacks.Callback):
        def __init__(self, state: dict, warmup: int, ramp: int, ss_max: float):
            super().__init__()
            self.state = state
            self.warmup = warmup
            self.ramp = ramp
            self.ss_max = ss_max

        def on_epoch_begin(self, epoch, logs=None):
            self.state["p"] = scheduled_sampling_prob(epoch, self.warmup, self.ramp, self.ss_max)
            print(f"scheduled sampling p={self.state['p']:.3f}")

    callbacks: list = [TerminateOnNaN()]
    if ss_state is not None:
        callbacks.append(
            _SSProb(
                ss_state,
                int(model_config["ss_warmup_epochs"]),
                int(model_config["ss_ramp_epochs"]),
                float(model_config["ss_max"]),
            )
        )
    if ar_validation_data is not None:
        callbacks.append(
            make_ar_validation_callback(
                *ar_validation_data,
                abort_threshold=float(model_config["ar_stability_max_mae_z"]),
            )
        )
    if ss_state is not None and ss_milestone_probs:
        callbacks.append(make_ss_milestone_callback(str(prefix), ss_state, ss_milestone_probs))
    interval = int(
        model_config.get("rollout_checkpoint_interval", 0)
        if periodic_interval is None
        else periodic_interval
    )
    if interval > 0:
        callbacks.append(make_periodic_checkpoint_callback(str(prefix), interval))
    callbacks.append(
        ModelCheckpoint(
            filepath=str(HERE / f"{prefix}_last.h5"),
            save_weights_only=True,
            save_freq="epoch",
            verbose=0,
        )
    )
    if save_best:
        callbacks.append(
            ModelCheckpoint(
                filepath=str(HERE / f"{prefix}_best.h5"),
                save_weights_only=True,
                save_best_only=True,
                monitor=monitor,
                initial_value_threshold=initial_best,
                verbose=1,
            )
        )
    if reduce_lr:
        callbacks.append(
            ReduceLROnPlateau(
                monitor=monitor,
                factor=model_config["reduce_lr_factor"],
                patience=model_config["reduce_lr_patience"],
                min_lr=model_config["min_lr"],
                verbose=1,
            )
        )
    if early_stop:
        callbacks.append(
            EarlyStopping(
                monitor=monitor,
                patience=model_config["early_stopping_patience"],
                restore_best_weights=restore_best_weights,
                verbose=1,
            )
        )
    return callbacks


def train(
    lite: bool = False,
    epochs: int | None = None,
    data: Path | None = None,
    prepad: bool = False,
    sequence: bool = False,
    min_step_px: float | None = None,
    no_augment: bool = False,
    no_ss: bool = False,
    trajectory_split: bool = False,
    ss_target_mode: str | None = None,
    initial_weights: Path | None = None,
    initial_epoch: int = 0,
    run_name: str | None = None,
) -> Path:
    loader_mode = _resolve_loader_mode(prepad, sequence)
    _configure_onednn(loader_mode)

    units = model_config["lstm_units_lite"] if lite else model_config["lstm_units"]
    configured_weights = Path(model_config["weights_lite" if lite else "weights"])
    prefix = Path(run_name).stem if run_name else configured_weights.stem
    weights_name = f"{prefix}{configured_weights.suffix}"
    norm_name = f"{prefix}_norm.json" if run_name else str(model_config["norm"])
    data_path = Path(data) if data else _resolve_data_path()
    n_epochs = epochs if epochs is not None else model_config["epochs"]
    pad = model_config["pad"]
    batch_size = model_config["batch_size"]
    max_steps = model_config["max_steps"]
    step_px = model_config["min_step_px"] if min_step_px is None else min_step_px
    noise_std = None if no_augment else model_config["input_noise_std"]
    remaining_frame = bool(model_config.get("remaining_frame", True))
    split_by_user = bool(model_config.get("split_by_user", False)) and not trajectory_split
    target_mode = ss_target_mode or str(model_config.get("ss_target_mode", "correction"))
    if target_mode not in {"correction", "natural", "hybrid"}:
        raise ValueError(f"unsupported scheduled-sampling target mode: {target_mode}")
    initial_epoch = int(initial_epoch)
    if initial_epoch < 0:
        raise ValueError("initial_epoch must be non-negative")
    if initial_epoch and initial_weights is None:
        raise ValueError("initial_epoch requires initial_weights")
    rng = np.random.default_rng(42)
    ss_max = 0.0 if no_ss else float(model_config["ss_max"])
    warmup_end, ramp_end, total_epochs = ss_schedule_epochs(
        int(model_config["ss_warmup_epochs"]),
        int(model_config["ss_ramp_epochs"]),
        n_epochs,
    )

    log_train_devices(loader_mode)
    print(
        f"Parsing sensor data from {data_path} (max_steps={max_steps}, min_step_px={step_px}, "
        f"remaining_frame={remaining_frame}) ..."
    )
    if ss_max > 0:
        print(
            f"scheduled sampling: warmup={warmup_end} ramp->{ramp_end} hold->{total_epochs} "
            f"max={ss_max} temp={model_config['ss_temp']} hops={int(model_config['ss_unroll_hops'])} "
            f"target={target_mode} target_clip_z={float(model_config['ss_target_clip_z'])}"
        )
    if not data_path.exists():
        raise SystemExit(f"Missing {data_path}. Run: python convert_swipemotiondb.py --sensors")

    model = build_sensor_model(units)
    if initial_weights is not None:
        model.load_weights(str(initial_weights))
        print(f"Loaded initial weights from {initial_weights} at epoch {initial_epoch}")
    model.summary()

    split = None
    split_path = HERE / str(model_config["split_manifest"])
    train_users: set[str] | None = None
    if loader_mode == "prepad":
        if split_by_user:
            X, Y, user_ids = load_sensor_jsonl_grouped(
                data_path,
                pad=pad,
                max_steps=max_steps,
                min_step_px=step_px,
                remaining_frame=remaining_frame,
            )
            split = load_or_build_user_split(
                user_ids,
                split_path,
                validation_fraction=float(model_config["validation_split"]),
                test_fraction=float(model_config["test_split"]),
                seed=int(model_config["split_seed"]),
            )
            partitions = partition_indices(user_ids, split)
            train_idx = partitions["train"]
            val_idx = partitions["validation"]
            X_train, Y_train = X[train_idx], Y[train_idx]
            X_val, Y_val = X[val_idx], Y[val_idx]
            split["row_counts"] = {name: int(len(indices)) for name, indices in partitions.items()}
            train_users = set(split["train_users"])
        else:
            X, Y = load_sensor_jsonl(
                data_path,
                pad=pad,
                max_steps=max_steps,
                min_step_px=step_px,
                remaining_frame=remaining_frame,
            )
            X_train, Y_train, X_val, Y_val = _split_arrays(
                X, Y, model_config["validation_split"]
            )
        norm = fit_sensor_norm(X_train, Y_train, pad)
        X_train, Y_train = apply_sensor_norm(X_train, Y_train, norm, pad)
        X_val, Y_val = apply_sensor_norm(X_val, Y_val, norm, pad)
        W_train = _mask_weights(Y_train, pad)
        W_val = _mask_weights(Y_val, pad)
        valid_steps = np.sum(Y_train[:, :, 0] != pad, axis=1)
        print(
            f"Loaded {len(X)} paths (prepad). "
            f"train len min/mean/max={valid_steps.min()}/{valid_steps.mean():.1f}/{valid_steps.max()}"
        )
        train_seq = make_prepad_sequence(
            X_train, Y_train, W_train, batch_size, pad, shuffle=True, noise_std=noise_std, rng=rng
        )
        val_seq = make_prepad_sequence(
            X_val, Y_val, W_val, batch_size, pad, shuffle=False, noise_std=None, rng=rng
        )
        ar_source_x, ar_source_y = X_val, Y_val
    else:
        if split_by_user:
            xs, ys, user_ids = load_sensor_sequences_grouped(
                data_path,
                max_steps=max_steps,
                min_step_px=step_px,
                remaining_frame=remaining_frame,
            )
            split = load_or_build_user_split(
                user_ids,
                split_path,
                validation_fraction=float(model_config["validation_split"]),
                test_fraction=float(model_config["test_split"]),
                seed=int(model_config["split_seed"]),
            )
            partitions = partition_indices(user_ids, split)
            x_train = [xs[int(index)] for index in partitions["train"]]
            y_train = [ys[int(index)] for index in partitions["train"]]
            x_val = [xs[int(index)] for index in partitions["validation"]]
            y_val = [ys[int(index)] for index in partitions["validation"]]
            split["row_counts"] = {name: int(len(indices)) for name, indices in partitions.items()}
            train_users = set(split["train_users"])
        else:
            xs, ys = load_sensor_sequences(
                data_path,
                max_steps=max_steps,
                min_step_px=step_px,
                remaining_frame=remaining_frame,
            )
            x_train, y_train, x_val, y_val = _split_sequences(
                xs, ys, model_config["validation_split"]
            )
        stacked_x = np.concatenate(x_train, axis=0)[None, ...]
        stacked_y = np.concatenate(y_train, axis=0)[None, ...]
        norm = fit_sensor_norm(stacked_x, stacked_y, pad)
        x_train, y_train = _zscore_list(x_train, y_train, norm, pad)
        x_val, y_val = _zscore_list(x_val, y_val, norm, pad)
        w_train = [np.ones(len(y), dtype=np.float32) for y in y_train]
        w_val = [np.ones(len(y), dtype=np.float32) for y in y_val]
        lengths = [len(x) for x in xs]
        print(
            f"Loaded {len(xs)} paths (sequence). "
            f"len min/mean/max={min(lengths)}/{sum(lengths) / len(lengths):.1f}/{max(lengths)}"
        )
        train_seq = make_batch_sequence(
            x_train, y_train, w_train, batch_size, pad, True, noise_std, rng
        )
        val_seq = make_batch_sequence(x_val, y_val, w_val, batch_size, pad, False, None, rng)
        ar_source_x, ar_source_y = x_val, y_val

    if split is not None:
        write_user_split(split_path, split)
        print(
            "User-grouped split: "
            f"train={len(split['train_users'])} users/{split['row_counts']['train']} rows, "
            f"validation={len(split['validation_users'])} users/{split['row_counts']['validation']} rows, "
            f"test={len(split['test_users'])} users/{split['row_counts']['test']} rows"
        )
        print(f"Wrote {split_path}")
    else:
        print("WARNING: trajectory-level split enabled; users may overlap between train and validation")
    init_by_condition = collect_init_by_condition(data_path, allowed_users=train_users)

    ar_x, ar_y = fixed_ar_validation_arrays(
        ar_source_x,
        ar_source_y,
        pad,
        max_paths=int(model_config["ar_eval_paths"]),
        max_steps=int(model_config["ar_eval_steps"]),
    )
    ar_validation_data = (ar_x, ar_y, norm, pad)
    print(f"Fixed AR validation batch: {ar_x.shape}")

    print(
        f"IMU z-score mean={np.round(norm['mean'], 4).tolist()} std={np.round(norm['std'], 4).tolist()}"
    )
    print(
        f"ΔIMU z-score mean={np.round(norm['delta_mean'], 4).tolist()} "
        f"std={np.round(norm['delta_std'], 4).tolist()}"
    )

    _log_first_batch(train_seq, noise_std, False, "off", remaining_frame)
    if ss_max <= 0:
        if total_epochs > initial_epoch:
            print(f"Teacher forcing only: epochs {initial_epoch}->{total_epochs}")
            model.fit(
                train_seq,
                epochs=total_epochs,
                initial_epoch=initial_epoch,
                validation_data=val_seq,
                callbacks=_make_callbacks(
                    prefix,
                    early_stop=True,
                    ar_validation_data=ar_validation_data,
                ),
            )
    else:
        ramp_initial = max(warmup_end, initial_epoch)
        ss_state = {
            "p": scheduled_sampling_prob(
                ramp_initial,
                int(model_config["ss_warmup_epochs"]),
                int(model_config["ss_ramp_epochs"]),
                ss_max,
            )
        }
        if warmup_end > initial_epoch:
            print(f"Phase 1 teacher forcing: epochs {initial_epoch}->{warmup_end}")
            model.fit(
                train_seq,
                epochs=warmup_end,
                initial_epoch=initial_epoch,
                validation_data=val_seq,
                callbacks=_make_callbacks(
                    prefix,
                    save_best=False,
                    ar_validation_data=ar_validation_data,
                ),
            )
        if ramp_initial < total_epochs:
            ss_train = wrap_scheduled_sampling(
                train_seq,
                model,
                pad,
                ss_state,
                rng,
                norm,
                float(model_config["ss_temp"]),
                hops=int(model_config["ss_unroll_hops"]),
                target_clip_z=float(model_config["ss_target_clip_z"]),
                target_mode=target_mode,
            )
            if ramp_end > ramp_initial:
                print(f"Phase 2 SS ramp: epochs {ramp_initial}->{ramp_end}")
                model.fit(
                    ss_train,
                    epochs=ramp_end,
                    initial_epoch=ramp_initial,
                    validation_data=val_seq,
                    callbacks=_make_callbacks(
                        prefix,
                        monitor="loss",
                        save_best=False,
                        reduce_lr=False,
                        ss_state=ss_state,
                        ar_validation_data=ar_validation_data,
                        ss_milestone_probs=tuple(model_config["ss_checkpoint_probs"]),
                    ),
                )
            hold_initial = max(ramp_end, initial_epoch)
            if total_epochs > hold_initial:
                model.optimizer.learning_rate.assign(float(model_config["learning_rate"]))
                ss_state["p"] = ss_max
                pre_hold_ar = autoregressive_mean_mae_z(model, ar_x, ar_y, norm, pad)
                print(
                    f"Phase 3 p={ss_max:g} hold: epochs {hold_initial}->{total_epochs} "
                    f"(AR stability baseline={pre_hold_ar:.5f}; "
                    f"lr reset to {model_config['learning_rate']})"
                )
                model.fit(
                    ss_train,
                    epochs=total_epochs,
                    initial_epoch=hold_initial,
                    validation_data=val_seq,
                    callbacks=_make_callbacks(
                        prefix,
                        monitor="val_ar_mae_z",
                        save_best=False,
                        reduce_lr=False,
                        early_stop=False,
                        ss_state=None,
                        ar_validation_data=ar_validation_data,
                    ),
                )
                final_ar = autoregressive_mean_mae_z(model, ar_x, ar_y, norm, pad)
                print(f"Final hold AR stability metric: val_ar_mae_z={final_ar:.5f}")

    _write_norm(HERE / norm_name, norm, init_by_condition)
    out = HERE / weights_name
    model.save_weights(str(out))
    print(f"\nTraining complete. Last weights: {out}")
    print("Select a periodic candidate with sensor.sweep before publishing release assets.")
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
    parser.add_argument("--no-ss", action="store_true", help="Disable scheduled sampling (teacher forcing only)")
    parser.add_argument(
        "--ss-target-mode",
        choices=("correction", "natural", "hybrid"),
        default=None,
        help="Scheduled-sampling label strategy",
    )
    parser.add_argument("--initial-weights", type=Path, default=None, help="Shared warmup checkpoint")
    parser.add_argument("--initial-epoch", type=int, default=0, help="Epoch represented by initial weights")
    parser.add_argument("--run-name", default=None, help="Isolated output/checkpoint prefix")
    parser.add_argument(
        "--trajectory-split",
        action="store_true",
        help="Use the legacy trajectory-random split instead of disjoint users",
    )
    args = parser.parse_args(argv)
    train(
        lite=args.lite,
        epochs=args.epochs,
        data=args.data,
        prepad=args.prepad,
        sequence=args.sequence,
        min_step_px=args.min_step_px,
        no_augment=args.no_augment,
        no_ss=args.no_ss,
        trajectory_split=args.trajectory_split,
        ss_target_mode=args.ss_target_mode,
        initial_weights=args.initial_weights,
        initial_epoch=args.initial_epoch,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
