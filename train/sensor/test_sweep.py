from pathlib import Path

import pytest

from sensor.sweep import (
    aggregate_diagnostics,
    aggregate_seed_diagnostics,
    candidate_is_feasible,
    checkpoint_spec,
    select_candidate,
)


def _condition_metrics(scale: float = 1.0) -> dict:
    return {
        "acc_p90": 10.0 * scale,
        "gyro_p50": 0.2 * scale,
        "delta_acc": 0.1 * scale,
        "delta_gyro": 0.04 * scale,
        "gyro_swipe_mean": 0.15 * scale,
        "gyro_residual": 0.08 * scale,
        "gyro_lag1": 0.7,
        "gyro_lag4": 0.3,
        "drift": 1.0 * scale,
    }


def test_checkpoint_spec_parses_label_and_expands_home():
    label, path = checkpoint_spec("warmup=~/weights.h5")
    assert label == "warmup"
    assert path == Path.home() / "weights.h5"


def test_checkpoint_spec_rejects_missing_label_or_separator():
    with pytest.raises(Exception):
        checkpoint_spec("weights.h5")
    with pytest.raises(Exception):
        checkpoint_spec("=weights.h5")


def test_aggregate_diagnostics_reports_ratios_and_errors():
    real = {condition: _condition_metrics() for condition in ("seated", "walking", "stress")}
    candidate = {condition: _condition_metrics(0.5) for condition in real}
    summary = {"real": real, "ar-t0.2": candidate}
    result = aggregate_diagnostics(summary, "ar-t0.2")
    assert result["gyro_p50_ratio"] == {
        "seated": 0.5,
        "walking": 0.5,
        "stress": 0.5,
    }
    assert result["temporal_mare"] == 0.5
    assert result["accel_p90_mare"] == 0.5
    assert result["drift_mare"] == 0.5
    assert result["distribution_score"] > 0.0


def test_seed_aggregation_uses_mean_score_and_worst_hard_gates():
    first = {
        "distribution_score": 0.1,
        "gyro_p50_ratio": {"seated": 0.9, "walking": 0.85, "stress": 1.0},
        "ar_tf_gyro_ratio": {"seated": 1.0, "walking": 0.9, "stress": 0.95},
        "gyro_swipe_mean_ratio": {"seated": 0.8},
        "gyro_residual_ratio": {"seated": 0.9},
        "temporal_mare": 0.08,
        "accel_p90_mare": 0.02,
        "drift_mare": 0.01,
    }
    second = {
        **first,
        "distribution_score": 0.3,
        "gyro_p50_ratio": {"seated": 0.79, "walking": 0.9, "stress": 1.0},
        "gyro_swipe_mean_ratio": {"seated": 1.0},
        "temporal_mare": 0.12,
    }
    aggregate = aggregate_seed_diagnostics([first, second])
    assert aggregate["distribution_score"] == pytest.approx(0.2)
    assert aggregate["gyro_p50_ratio"]["seated"] == 0.79
    assert aggregate["gyro_swipe_mean_ratio"]["seated"] == pytest.approx(0.9)
    assert aggregate["temporal_mare"] == 0.12
    assert not candidate_is_feasible(aggregate)


def test_candidate_selection_applies_stability_gates_before_score():
    noisy = {
        "temp": 0.4,
        "distribution_score": 0.1,
        "gyro_p50_ratio": {"seated": 1.0, "walking": 1.0, "stress": 1.0},
        "ar_tf_gyro_ratio": {"seated": 1.0, "walking": 1.0, "stress": 1.0},
        "temporal_mare": 0.6,
        "accel_p90_mare": 0.01,
        "drift_mare": 0.01,
    }
    stable = {
        "temp": 0.27,
        "distribution_score": 0.3,
        "gyro_p50_ratio": {"seated": 0.9, "walking": 0.8, "stress": 1.1},
        "ar_tf_gyro_ratio": {"seated": 1.0, "walking": 0.9, "stress": 1.1},
        "temporal_mare": 0.08,
        "accel_p90_mare": 0.03,
        "drift_mare": 0.02,
    }
    assert not candidate_is_feasible(noisy)
    assert candidate_is_feasible(stable)
    assert select_candidate([noisy, stable]) is stable


def test_candidate_selection_rejects_any_collapsed_gyro_condition():
    candidate = {
        "distribution_score": 0.1,
        "gyro_p50_ratio": {"seated": 1.0, "walking": 0.79, "stress": 1.0},
        "ar_tf_gyro_ratio": {"seated": 1.0, "walking": 1.0, "stress": 1.0},
        "temporal_mare": 0.05,
        "accel_p90_mare": 0.02,
        "drift_mare": 0.02,
    }
    assert not candidate_is_feasible(candidate)


def test_candidate_selection_rejects_ar_to_teacher_forced_collapse():
    candidate = {
        "distribution_score": 0.1,
        "gyro_p50_ratio": {"seated": 1.0, "walking": 1.0, "stress": 1.0},
        "ar_tf_gyro_ratio": {"seated": 1.0, "walking": 0.79, "stress": 1.0},
        "temporal_mare": 0.05,
        "accel_p90_mare": 0.02,
        "drift_mare": 0.02,
    }
    assert not candidate_is_feasible(candidate)
