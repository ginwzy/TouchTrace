import math

import numpy as np

from sensor.features import encode_sensor_trajectory, pack_sensor_step
from sensor.generate import PARAMS_SIZE, decode_imu_mdn, generate_sensor_along_path, mixture_mean, sample_imu_mdn
from sensor.test_features import _fused


class _FakeTensor:
    def __init__(self, name: str):
        self.name = name


class _DeltaSession:
    """MDN that always emits a tiny z-space step (component 0, near-zero scale)."""

    def get_inputs(self):
        return [_FakeTensor("input")]

    def get_outputs(self):
        return [_FakeTensor("dense")]

    def run(self, _names, _feeds):
        params = np.full((1, 1, PARAMS_SIZE), -20.0, dtype=np.float32)
        params[0, 0, 0] = 40.0
        params[0, 0, 1:5] = -40.0
        params[0, 0, 5:11] = 0.0
        return [params]


def test_pack_matches_encode_first_step():
    path, sensors, cond = _fused()
    X, _Y = encode_sensor_trajectory(path, sensors, cond, remaining_frame=True)
    packed = pack_sensor_step(
        sensors[0]["accel"] + sensors[0]["gyro"],
        path[0],
        path[1],
        path[-1],
        cond,
        remaining_frame=True,
    )
    assert packed == X[0]


def test_sample_imu_mdn_is_six_dim():
    params = np.zeros(PARAMS_SIZE, dtype=np.float64)
    params[0] = 10.0
    sample = sample_imu_mdn(params, np.random.default_rng(0))
    assert sample.shape == (6,)
    assert np.isfinite(sample).all()


def test_mixture_mean_picks_dominant_component():
    params = np.zeros(PARAMS_SIZE, dtype=np.float64)
    params[0] = 20.0
    params[1:5] = -20.0
    params[5:11] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    mean = mixture_mean(params)
    assert np.allclose(mean, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], atol=1e-6)


class _RecordSession(_DeltaSession):
    def __init__(self):
        self.prev_z = []

    def run(self, _names, feeds):
        x = next(iter(feeds.values()))
        self.prev_z.append(x[0, -1, :6].copy())
        return super().run(_names, feeds)


def test_teacher_forcing_uses_real_previous_imu():
    path, sensors, cond = _fused()
    imu0 = sensors[0]["accel"] + sensors[0]["gyro"]
    norm = {"mean": [0.0] * 6, "std": [1.0] * 6}
    ar = _RecordSession()
    tf = _RecordSession()
    generate_sensor_along_path(ar, path, imu0, cond, norm, np.random.default_rng(0), temp=0)
    generate_sensor_along_path(
        tf, path, imu0, cond, norm, np.random.default_rng(0), temp=0, teacher_sensors=sensors
    )
    assert np.allclose(ar.prev_z[0], imu0)
    assert np.allclose(tf.prev_z[0], imu0)
    assert np.allclose(tf.prev_z[1], sensors[1]["accel"] + sensors[1]["gyro"])
    assert not np.allclose(ar.prev_z[1], tf.prev_z[1])


def test_generate_keeps_timestamps_and_init():
    path, sensors, cond = _fused()
    imu0 = sensors[0]["accel"] + sensors[0]["gyro"]
    norm = {"mean": [0.0] * 6, "std": [1.0] * 6}
    out = generate_sensor_along_path(_DeltaSession(), path, imu0, cond, norm, np.random.default_rng(0))
    assert len(out) == len(path)
    assert [s["timestamp"] for s in out] == [p["timestamp"] for p in path]
    assert out[0]["accel"] == sensors[0]["accel"]
    assert out[0]["gyro"] == sensors[0]["gyro"]
    assert math.hypot(*out[1]["accel"]) < 1.0


def test_decode_temp_interpolates_between_mean_and_sample():
    params = np.zeros(PARAMS_SIZE, dtype=np.float64)
    params[0] = 8.0
    params[5:11] = 1.0
    center = mixture_mean(params)
    drawn = decode_imu_mdn(params, np.random.default_rng(1), temp=1.0)
    mixed = decode_imu_mdn(params, np.random.default_rng(1), temp=0.4)
    assert np.allclose(decode_imu_mdn(params, np.random.default_rng(1), temp=0.0), center)
    assert np.allclose(mixed, center + 0.4 * (drawn - center))
