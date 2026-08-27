from generate import _no_backtrack_step


def test_no_backtrack_redirects_reverse_step():
    dx, dy = _no_backtrack_step(0.0, 0.0, (100.0, 0.0), -8.0, 1.0, min_step_px=3.0)
    assert dx > 0
    assert abs(dy) < 1e-6


def test_no_backtrack_keeps_forward_step():
    dx, dy = _no_backtrack_step(0.0, 0.0, (100.0, 0.0), 6.0, 2.0, min_step_px=3.0)
    assert (dx, dy) == (6.0, 2.0)
