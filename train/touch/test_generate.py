import math

from touch.eval import path_metrics
from touch.generate import GenerateOptions, TouchStep, _clamp_step_mag, _no_backtrack_step


def test_no_backtrack_redirects_reverse_step():
    dx, dy = _no_backtrack_step(0.0, 0.0, (100.0, 0.0), -8.0, 1.0, min_step_px=3.0)
    assert dx > 0
    assert abs(dy) < 1e-6


def test_no_backtrack_keeps_forward_step():
    dx, dy = _no_backtrack_step(0.0, 0.0, (100.0, 0.0), 6.0, 2.0, min_step_px=3.0)
    assert (dx, dy) == (6.0, 2.0)


def test_default_options_use_noback():
    opts = GenerateOptions()
    assert opts.no_backtrack is True


def test_clamp_step_mag_caps_length_keeps_direction():
    dx, dy = _clamp_step_mag(30.0, 40.0, max_step_px=10.0)
    assert abs(math.hypot(dx, dy) - 10.0) < 1e-6
    assert abs(dx / 30.0 - dy / 40.0) < 1e-6


def test_clamp_step_mag_leaves_short_steps():
    assert _clamp_step_mag(3.0, 4.0, max_step_px=35.0) == (3.0, 4.0)


def test_path_metrics_signed_bow_excludes_forced_end():
    start, end = (0.0, 0.0), (100.0, 0.0)
    path = [
        TouchStep(x=0, y=0, t=0.0),
        TouchStep(x=50, y=-20, t=40.0),
        TouchStep(x=99, y=-1, t=78.0),
        TouchStep(x=100, y=0, t=80.0),
    ]
    m = path_metrics(path, start, end)
    assert m["bow_signed"] == -20.0
    assert m["bow_px"] == 20.0
    assert m["reach"] is True
    assert m["end_err"] < 3.0
    assert abs(m["step_px"] - math.hypot(50, 20)) < 1.0


def test_path_metrics_real_end_uses_last_point():
    start, end = (0.0, 0.0), (100.0, 0.0)
    path = [
        TouchStep(x=0, y=0, t=0.0),
        TouchStep(x=50, y=-20, t=40.0),
        TouchStep(x=100, y=0, t=80.0),
    ]
    m = path_metrics(path, start, end, drop_forced_end=False)
    assert m["reach"] is True
    assert m["end_err"] == 0.0
