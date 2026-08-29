import numpy as np

from sensor.convert import resolve_sensor_weights
from sensor.generate import mix_mdn_draw
from sensor.train import mix_scheduled_imu, scheduled_sampling_prob, ss_schedule_epochs


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


def test_resolve_sensor_weights_prefers_final_over_tf_best(tmp_path):
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
