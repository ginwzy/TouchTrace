from collections import Counter

from sensor.eval import (
    _gyro_persistence_summary,
    _mode_name,
    _step_change_summary,
    distribution_score,
    val_rows,
)


def _row(condition: str, value: int) -> dict:
    return {"meta": {"condition": condition}, "value": value}


def test_val_rows_stratifies_a_limited_validation_sample():
    rows = (
        [_row("seated", i) for i in range(20)]
        + [_row("walking", i) for i in range(20)]
        + [_row("stress", i) for i in range(20)]
    )
    picked = val_rows(rows, val_fraction=0.5, seed=42, limit=9, stratified=True)
    counts = Counter(row["meta"]["condition"] for row in picked)
    assert counts == {"seated": 3, "walking": 3, "stress": 3}


def test_mode_name_includes_innovation_correlation():
    assert _mode_name(0.3, False, 0.0) == "ar-t0.3"
    assert _mode_name(0.3, False, 0.7) == "ar-t0.3-r0.7"
    assert _mode_name(0.0, False, 0.7) == "ar-mean"


def test_step_change_summary_uses_vector_deltas():
    sensors = [
        {"accel": [0.0, 0.0, 9.0], "gyro": [0.0, 0.0, 0.0]},
        {"accel": [3.0, 4.0, 9.0], "gyro": [0.0, 0.0, 0.2]},
        {"accel": [3.0, 4.0, 10.0], "gyro": [0.0, 0.3, 0.2]},
    ]
    dacc, dgyro = _step_change_summary(sensors)
    assert dacc == 3.0  # median of [1, 5]
    assert dgyro == 0.25  # median of [0.2, 0.3]


def test_gyro_persistence_separates_swipe_mean_and_residual():
    sensors = [
        {"gyro": [1.0, 0.0, 0.0]},
        {"gyro": [2.0, 0.0, 0.0]},
        {"gyro": [3.0, 0.0, 0.0]},
    ]
    result = _gyro_persistence_summary(sensors)
    assert result["mean"] == 2.0
    assert result["residual"] == 1.0
    assert result["lag1"] == 0.0
    assert result["lag4"] != result["lag4"]  # NaN for a sequence shorter than the lag.


def test_distribution_score_is_zero_for_reference_and_penalizes_collapse():
    real = {
        condition: {
            "acc_p90": 10.0,
            "gyro_p50": 0.2,
            "delta_acc": 0.1,
            "delta_gyro": 0.04,
            "gyro_swipe_mean": 0.15,
            "gyro_residual": 0.08,
            "gyro_lag1": 0.7,
            "gyro_lag4": 0.3,
            "drift": 1.0,
        }
        for condition in ("seated", "walking", "stress")
    }
    collapsed = {
        condition: {**metrics, "gyro_p50": 0.1, "gyro_swipe_mean": 0.05}
        for condition, metrics in real.items()
    }
    summary = {"real": real, "same": real, "collapsed": collapsed}
    assert distribution_score(summary, "same") == 0.0
    assert distribution_score(summary, "collapsed") > 0.1
