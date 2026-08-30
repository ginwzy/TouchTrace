import numpy as np
import pytest

from sensor.convert import export_onnx as export_sensor_onnx
from sensor.convert import resolve_sensor_weights
from sensor.generate import mix_mdn_draw
from sensor.split import write_user_split
from sensor.train import (
    autoregressive_mean_mae_z,
    fixed_ar_validation_arrays,
    load_or_build_user_split,
    make_ar_validation_callback,
    make_periodic_checkpoint_callback,
    make_sensor_mdn_loss,
    mix_scheduled_imu,
    scheduled_sampling_prob,
    ss_checkpoint_tag,
    ss_schedule_epochs,
    unroll_scheduled_imu,
)


def test_export_sensor_onnx_uses_explicit_norm(tmp_path, monkeypatch):
    import sensor.convert as sensor_convert

    captured = {}

    def fake_export(**kwargs):
        captured.update(kwargs)
        return kwargs["output"]

    monkeypatch.setattr(sensor_convert, "export_mdn_onnx", fake_export)
    weights = tmp_path / "candidate.h5"
    norm = tmp_path / "candidate_norm.json"
    output = tmp_path / "candidate.onnx"
    assert export_sensor_onnx(
        weights=weights,
        output=output,
        norm=norm,
        publish=False,
    ) == output
    assert captured["extra_files"] == (norm,)
    assert captured["publish"] is False


def test_ss_prob_zero_during_warmup():
    assert scheduled_sampling_prob(0, warmup=20, ramp=50, ss_max=1.0) == 0.0
    assert scheduled_sampling_prob(19, warmup=20, ramp=50, ss_max=1.0) == 0.0


def test_ss_prob_ramps_then_caps():
    p0 = scheduled_sampling_prob(20, warmup=20, ramp=50, ss_max=1.0)
    p_mid = scheduled_sampling_prob(44, warmup=20, ramp=50, ss_max=1.0)
    p_end = scheduled_sampling_prob(69, warmup=20, ramp=50, ss_max=1.0)
    p_late = scheduled_sampling_prob(200, warmup=20, ramp=50, ss_max=1.0)
    assert abs(p0 - 1.0 / 50) < 1e-9
    assert p_mid > p0
    assert abs(p_end - 1.0) < 1e-9
    assert p_late == 1.0


def test_ss_schedule_splits_warmup_ramp_hold():
    assert ss_schedule_epochs(20, 50, 250) == (20, 70, 250)
    assert ss_schedule_epochs(20, 50, 30) == (20, 30, 30)
    assert ss_schedule_epochs(20, 50, 20) == (20, 20, 20)
    assert ss_schedule_epochs(20, 50, 10) == (10, 10, 10)


def test_resolve_sensor_weights_prefers_final_over_best(tmp_path):
    cfg = {"weights": "sensor_model.h5"}
    best = tmp_path / "sensor_model_best.h5"
    last = tmp_path / "sensor_model_last.h5"
    final = tmp_path / "sensor_model.h5"
    best.write_bytes(b"best")
    assert resolve_sensor_weights(tmp_path, cfg).name == "sensor_model_best.h5"
    last.write_bytes(b"last")
    assert resolve_sensor_weights(tmp_path, cfg).name == "sensor_model_last.h5"
    final.write_bytes(b"final")
    assert resolve_sensor_weights(tmp_path, cfg).name == "sensor_model.h5"
    explicit = tmp_path / "custom.h5"
    explicit.write_bytes(b"custom")
    assert resolve_sensor_weights(tmp_path, cfg, weights=explicit) == explicit


def test_mix_mdn_draw_interpolates():
    mean = np.zeros(6)
    sample = np.ones(6)
    assert np.allclose(mix_mdn_draw(mean, sample, 0.0), 0.0)
    assert np.allclose(mix_mdn_draw(mean, sample, 1.0), 1.0)
    assert np.allclose(mix_mdn_draw(mean, sample, 0.2), 0.2)


def test_mix_replaces_next_prev_with_previous_prediction():
    pad = -999.0
    x = np.zeros((1, 3, 13), dtype=np.float32)
    x[0, :, :6] = np.array([[1.0] * 6, [2.0] * 6, [3.0] * 6], dtype=np.float32)
    y = np.ones((1, 3, 6), dtype=np.float32)
    pred = np.array([[[10.0] * 6, [20.0] * 6, [30.0] * 6]], dtype=np.float32)
    mixed = mix_scheduled_imu(x, y, pred, pad, prob=1.0, rng=np.random.default_rng(0))
    assert np.allclose(mixed[0, 0, :6], 1.0)
    assert np.allclose(mixed[0, 1, :6], 10.0)
    assert np.allclose(mixed[0, 2, :6], 20.0)
    assert np.allclose(mixed[..., 6:], 0.0)


def test_mix_skips_when_prob_zero():
    x = np.ones((2, 4, 13), dtype=np.float32)
    y = np.ones((2, 4, 6), dtype=np.float32)
    pred = np.zeros((2, 4, 6), dtype=np.float32)
    out = mix_scheduled_imu(x, y, pred, pad=-1.0, prob=0.0, rng=np.random.default_rng(0))
    assert out is x


def _unit_norm():
    return {
        "mean": [0.0] * 6,
        "std": [1.0] * 6,
        "delta_mean": [0.0] * 6,
        "delta_std": [1.0] * 6,
        "target": "delta",
    }


def test_unroll_zero_hops_is_noop():
    x = np.ones((1, 3, 13), dtype=np.float32)
    y = np.zeros((1, 3, 6), dtype=np.float32)
    out_x, out_y = unroll_scheduled_imu(
        x, y, pad=-999.0, prob=1.0, rng=np.random.default_rng(0),
        predict_delta_z=lambda _x: np.zeros_like(y),
        norm=_unit_norm(), hops=0,
    )
    assert out_x is x
    assert out_y is y


def test_unroll_compounds_zero_delta_across_hops():
    x = np.zeros((1, 3, 13), dtype=np.float32)
    x[0, :, :6] = np.array([[1.0] * 6, [2.0] * 6, [3.0] * 6], dtype=np.float32)
    y = np.ones((1, 3, 6), dtype=np.float32)
    rng = np.random.default_rng(0)

    def predict_zero(_x):
        return np.zeros_like(y)

    x1, _ = unroll_scheduled_imu(
        x, y, pad=-999.0, prob=1.0, rng=rng, predict_delta_z=predict_zero, norm=_unit_norm(), hops=1,
    )
    assert np.allclose(x1[0, 0, :6], 1.0)
    assert np.allclose(x1[0, 1, :6], 1.0)
    assert np.allclose(x1[0, 2, :6], 2.0)

    x2, y2 = unroll_scheduled_imu(
        x, y, pad=-999.0, prob=1.0, rng=np.random.default_rng(0),
        predict_delta_z=predict_zero, norm=_unit_norm(), hops=2,
    )
    assert np.allclose(x2[0, :, :6], 1.0)
    # orig next = prev + 1; after two hops imu_prev is 1 everywhere except it is 1 at t=0 too.
    assert np.allclose(y2[0, 0], 1.0)
    assert np.allclose(y2[0, 1], 2.0)
    assert np.allclose(y2[0, 2], 3.0)


def test_unroll_clips_only_retargeted_delta_channels():
    x = np.zeros((1, 3, 13), dtype=np.float32)
    x[0, :, :6] = np.array([[0.0] * 6, [10.0] * 6, [20.0] * 6], dtype=np.float32)
    y = np.ones((1, 3, 6), dtype=np.float32)
    diagnostics = {}

    x_mix, y_mix = unroll_scheduled_imu(
        x,
        y,
        pad=-999.0,
        prob=1.0,
        rng=np.random.default_rng(0),
        predict_delta_z=lambda _x: np.zeros_like(y),
        norm=_unit_norm(),
        hops=1,
        target_clip_z=2.0,
        diagnostics=diagnostics,
    )

    assert np.allclose(x_mix[0, 0, :6], 0.0)
    assert np.allclose(y_mix[0, 0], 1.0)  # Natural first-step target is untouched.
    assert np.allclose(y_mix[0, 1:], 2.0)
    assert diagnostics["target_values"] == 12
    assert diagnostics["clipped_values"] == 12
    assert diagnostics["max_abs_before_clip"] == 11.0


def test_unroll_natural_target_keeps_human_deltas():
    x = np.zeros((1, 3, 13), dtype=np.float32)
    x[0, :, :6] = np.array([[0.0] * 6, [10.0] * 6, [20.0] * 6], dtype=np.float32)
    y = np.ones((1, 3, 6), dtype=np.float32)

    _, y_mix = unroll_scheduled_imu(
        x,
        y,
        pad=-999.0,
        prob=1.0,
        rng=np.random.default_rng(0),
        predict_delta_z=lambda _x: np.zeros_like(y),
        norm=_unit_norm(),
        hops=1,
        target_clip_z=2.0,
        target_mode="natural",
    )

    assert y_mix.shape == y.shape
    assert np.array_equal(y_mix, y)


def test_unroll_hybrid_separates_natural_nll_and_mean_correction():
    x = np.zeros((1, 3, 13), dtype=np.float32)
    x[0, :, :6] = np.array([[0.0] * 6, [10.0] * 6, [20.0] * 6], dtype=np.float32)
    y = np.ones((1, 3, 6), dtype=np.float32)

    _, packed = unroll_scheduled_imu(
        x,
        y,
        pad=-999.0,
        prob=1.0,
        rng=np.random.default_rng(0),
        predict_delta_z=lambda _x: np.zeros_like(y),
        norm=_unit_norm(),
        hops=1,
        target_clip_z=2.0,
        target_mode="hybrid",
    )

    assert packed.shape == (1, 3, 13)
    assert np.array_equal(packed[..., :6], y)
    assert np.allclose(packed[0, :, 6:12], [[1.0] * 6, [2.0] * 6, [2.0] * 6])
    assert np.array_equal(packed[0, :, 12], [0.0, 1.0, 1.0])


def test_sensor_mdn_loss_applies_hybrid_penalty_only_when_masked():
    import tensorflow as tf
    import tensorflow_probability as tfp

    distribution = tfp.distributions.Independent(
        tfp.distributions.Normal(
            loc=tf.zeros((1, 2, 6), dtype=tf.float32),
            scale=tf.ones((1, 2, 6), dtype=tf.float32),
        ),
        reinterpreted_batch_ndims=1,
    )
    loss = make_sensor_mdn_loss(pad=-999.0, nll_clip=100.0, hybrid_weight=0.1)
    natural = tf.zeros((1, 2, 6), dtype=tf.float32)
    correction = tf.ones((1, 2, 6), dtype=tf.float32)
    inactive = tf.concat((natural, correction, tf.zeros((1, 2, 1))), axis=-1)
    active = tf.concat((natural, correction, tf.ones((1, 2, 1))), axis=-1)

    baseline = float(loss(natural, distribution).numpy())
    assert float(loss(inactive, distribution).numpy()) == pytest.approx(baseline)
    assert float(loss(active, distribution).numpy()) > baseline


class _MeanDistribution:
    def __init__(self, value):
        self.value = value

    def mean(self):
        return self.value


class _ConstantDeltaModel:
    def __init__(self, delta: float):
        self.delta = delta

    def __call__(self, x, training=False):
        assert training is False
        shape = (*np.asarray(x).shape[:-1], 6)
        return _MeanDistribution(np.full(shape, self.delta, dtype=np.float32))


def test_autoregressive_metric_rolls_predictions_forward_and_ignores_pad():
    pad = -999.0
    x = np.full((2, 3, 13), pad, dtype=np.float32)
    y = np.full((2, 3, 6), pad, dtype=np.float32)
    x[0, :, :6] = np.array([[0.0] * 6, [1.0] * 6, [2.0] * 6])
    y[0, :, :] = 1.0
    x[1, 0, :6] = 5.0
    y[1, 0, :] = 1.0

    assert autoregressive_mean_mae_z(_ConstantDeltaModel(1.0), x, y, _unit_norm(), pad) == 0.0
    assert autoregressive_mean_mae_z(_ConstantDeltaModel(0.0), x, y, _unit_norm(), pad) == 1.75


def test_fixed_ar_validation_arrays_truncates_and_pads_sequences():
    xs = [np.ones((4, 13), dtype=np.float32), np.ones((2, 13), dtype=np.float32) * 2]
    ys = [np.ones((4, 6), dtype=np.float32), np.ones((2, 6), dtype=np.float32) * 2]
    x, y = fixed_ar_validation_arrays(xs, ys, pad=-999.0, max_paths=2, max_steps=3)
    assert x.shape == (2, 3, 13)
    assert y.shape == (2, 3, 6)
    assert np.all(y[0] == 1.0)
    assert np.all(y[1, :2] == 2.0)
    assert np.all(y[1, 2] == -999.0)


def test_ss_checkpoint_tag_is_stable():
    assert ss_checkpoint_tag(0.2) == "p020"
    assert ss_checkpoint_tag(0.8) == "p080"
    assert ss_checkpoint_tag(1.0) == "p100"


def test_periodic_checkpoint_uses_completed_epoch_number(tmp_path, monkeypatch):
    import sensor.train as sensor_train

    saved = []

    class Model:
        def save_weights(self, path):
            saved.append(path)

    monkeypatch.setattr(sensor_train, "HERE", tmp_path)
    callback = make_periodic_checkpoint_callback("sensor_model", interval=5)
    callback.set_model(Model())
    callback.on_epoch_end(3)
    callback.on_epoch_end(4)
    assert saved == [str(tmp_path / "sensor_model_candidate_e005.h5")]


def test_hold_ar_callback_stops_only_on_severe_divergence(monkeypatch):
    import sensor.train as sensor_train

    class Model:
        stop_training = False

    monkeypatch.setattr(sensor_train, "autoregressive_mean_mae_z", lambda *_args: 2.1)
    callback = make_ar_validation_callback(
        np.zeros((1, 1, 13)),
        np.zeros((1, 1, 6)),
        _unit_norm(),
        -999.0,
        abort_threshold=2.0,
    )
    model = Model()
    callback.set_model(model)
    logs = {}
    callback.on_epoch_end(0, logs)
    assert logs["val_ar_mae_z"] == 2.1
    assert model.stop_training is True


def test_user_split_manifest_is_reused_and_dataset_drift_fails(tmp_path):
    users = [str(index) for index in range(10)]
    manifest_path = tmp_path / "split.json"
    split = load_or_build_user_split(
        users,
        manifest_path,
        validation_fraction=0.2,
        test_fraction=0.2,
        seed=42,
    )
    split["row_counts"] = {"train": 6, "validation": 2, "test": 2}
    write_user_split(manifest_path, split)

    reused = load_or_build_user_split(
        list(reversed(users)),
        manifest_path,
        validation_fraction=0.2,
        test_fraction=0.2,
        seed=7,
    )
    assert reused == split

    with pytest.raises(ValueError, match="user set differs"):
        load_or_build_user_split(
            users + ["new-user"],
            manifest_path,
            validation_fraction=0.2,
            test_fraction=0.2,
            seed=42,
        )
