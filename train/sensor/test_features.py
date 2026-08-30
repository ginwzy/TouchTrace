"""Sensor encoding: fused path+IMU → (imu_prev, kinematics, condition) sequences."""

import json
from pathlib import Path

import numpy as np

from sensor.features import (
    abs_z_from_delta_z,
    apply_sensor_norm,
    delta_z_from_abs_z,
    collect_init_by_condition,
    condition_onehot,
    encode_sensor_trajectory,
    fit_sensor_norm,
    load_sensor_jsonl,
    load_sensor_jsonl_grouped,
    require_delta_norm,
    subsample_paired,
)


def _fused(condition="walking"):
    path = [
        {"x": 0.0, "y": 0.0, "timestamp": 0.0},
        {"x": 0.0, "y": 10.0, "timestamp": 16.0},
        {"x": 0.0, "y": 40.0, "timestamp": 48.0},
    ]
    sensors = [
        {"timestamp": 0.0, "accel": [0.0, 0.0, 9.8], "gyro": [0.0, 0.0, 0.0]},
        {"timestamp": 16.0, "accel": [0.2, 0.0, 9.8], "gyro": [0.01, 0.0, 0.0]},
        {"timestamp": 48.0, "accel": [0.4, 0.0, 9.8], "gyro": [0.02, 0.0, 0.0]},
    ]
    return path, sensors, condition


def test_condition_onehot_order():
    assert condition_onehot("seated") == [1.0, 0.0, 0.0]
    assert condition_onehot("walking") == [0.0, 1.0, 0.0]
    assert condition_onehot("stress") == [0.0, 0.0, 1.0]
    assert condition_onehot("nope") == [0.0, 0.0, 0.0]


def test_subsample_paired_keeps_aligned_sensors():
    path, sensors, _ = _fused()
    path.insert(1, {"x": 0.0, "y": 1.0, "timestamp": 8.0})
    sensors.insert(1, {"timestamp": 8.0, "accel": [0.1, 0.0, 9.8], "gyro": [0.0, 0.0, 0.0]})
    p2, s2 = subsample_paired(path, sensors, min_step_px=3.0)
    assert [p["y"] for p in p2] == [0.0, 10.0, 40.0]
    assert [s["accel"][0] for s in s2] == [0.0, 0.2, 0.4]


def test_encode_sensor_uses_prev_imu_current_step_and_condition():
    path, sensors, cond = _fused()
    X, Y = encode_sensor_trajectory(path, sensors, cond, remaining_frame=True)
    assert len(X) == 2
    assert len(Y) == 2
    assert X[0][:6] == [0.0, 0.0, 9.8, 0.0, 0.0, 0.0]
    assert Y[0] == [0.2, 0.0, 0.0, 0.01, 0.0, 0.0]
    assert Y[1] == [0.2, 0.0, 0.0, 0.01, 0.0, 0.0]
    assert X[0][-3:] == [0.0, 1.0, 0.0]
    # remaining-frame: vertical step is pure tangent, rem is remaining length at prev
    assert abs(X[0][6] - 10.0) < 1e-6
    assert abs(X[0][7]) < 1e-6
    assert abs(X[0][8] - 16.0) < 1e-6
    assert abs(X[0][9] - 40.0) < 1e-6


def test_load_sensor_jsonl_and_zscore(tmp_path: Path):
    path, sensors, cond = _fused()
    rec = {
        "target": {"x": 0.0, "y": 40.0},
        "meta": {"condition": cond, "user_id": "user-a"},
        "path": path,
        "sensors": sensors,
    }
    jsonl = tmp_path / "touch_sensor_data.jsonl"
    jsonl.write_text(json.dumps(rec) + "\n")
    X, Y = load_sensor_jsonl(jsonl, pad=-999.0, min_step_px=0.0)
    grouped_x, grouped_y, users = load_sensor_jsonl_grouped(jsonl, pad=-999.0, min_step_px=0.0)
    assert np.array_equal(grouped_x, X)
    assert np.array_equal(grouped_y, Y)
    assert users == ["user-a"]
    assert X.shape[-1] == 13
    assert Y.shape[-1] == 6
    assert Y.shape[0] == 1
    assert np.allclose(Y[0, 0], [0.2, 0.0, 0.0, 0.01, 0.0, 0.0])
    norm = fit_sensor_norm(X, Y, pad=-999.0)
    assert norm["target"] == "delta"
    Xn, Yn = apply_sensor_norm(X, Y, norm, pad=-999.0)
    assert abs(float(Yn.mean())) < 1e-5
    inits = collect_init_by_condition(jsonl)
    assert "walking" in inits
    assert abs(inits["walking"]["mean"][2] - 9.8) < 1e-6
    assert collect_init_by_condition(jsonl, allowed_users={"other"}) == {}


def test_abs_z_from_delta_z_integrates_in_device_frame():
    norm = {
        "mean": [0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
        "std": [2.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "delta_mean": [0.0] * 6,
        "delta_std": [0.5] * 6,
        "target": "delta",
    }
    # prev abs = (0, 0, 10) → z = (0, 0, 0); delta = 1.0 on ax → z_delta ax = 2.0
    prev_z = np.zeros(6, dtype=np.float32)
    delta_z = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    got = abs_z_from_delta_z(prev_z, delta_z, norm)
    # next abs ax = 0 + 1 = 1 → z = (1-0)/2 = 0.5
    assert np.allclose(got, [0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert np.allclose(delta_z_from_abs_z(prev_z, got, norm), delta_z)


def test_require_delta_norm_rejects_absolute_stats():
    import pytest

    with pytest.raises(ValueError, match="delta stats"):
        require_delta_norm({"mean": [0.0] * 6, "std": [1.0] * 6})
