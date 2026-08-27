"""Feature encoding for touch training: path points → (dx, dy, dt, dist) sequences."""

from pathlib import Path
import gzip

from features import encode_trajectory, load_trajectory_jsonl, load_trajectory_sequences, subsample_path


def test_subsample_path_drops_micro_moves_keeps_endpoints():
    path = [
        {"x": 0.0, "y": 0.0, "timestamp": 0.0},
        {"x": 1.0, "y": 0.0, "timestamp": 5.0},
        {"x": 2.0, "y": 0.0, "timestamp": 10.0},
        {"x": 20.0, "y": 0.0, "timestamp": 30.0},
    ]
    out = subsample_path(path, min_step_px=3.0)
    assert [p["x"] for p in out] == [0.0, 20.0]


def test_encode_trajectory_uses_previous_step_and_remaining_distance():
    path = [
        {"x": 100.0, "y": 800.0, "timestamp": 0.0},
        {"x": 140.0, "y": 760.0, "timestamp": 16.0},
        {"x": 540.0, "y": 1200.0, "timestamp": 100.0},
    ]
    target = {"x": 540.0, "y": 1200.0}

    X, Y = encode_trajectory(path, target)

    assert X == [
        [0.0, 0.0, 0.0, 440.0, 400.0],
        [40.0, -40.0, 16.0, 400.0, 440.0],
    ]
    assert Y == [
        [40.0, -40.0, 16.0],
        [400.0, 440.0, 84.0],
    ]


def test_load_jsonl_pads_shorter_paths_and_ignores_pressure(tmp_path: Path):
    jsonl = tmp_path / "touch_data.jsonl"
    jsonl.write_text(
        '{"target":{"x":12.0,"y":8.0},"path":['
        '{"x":0.0,"y":0.0,"timestamp":0.0,"pressure":0.5},'
        '{"x":4.0,"y":0.0,"timestamp":10.0,"pressure":0.6},'
        '{"x":12.0,"y":8.0,"timestamp":30.0,"pressure":0.4}'
        "]}\n"
        '{"target":{"x":3.0,"y":0.0},"path":['
        '{"x":0.0,"y":0.0,"timestamp":0.0},'
        '{"x":3.0,"y":0.0,"timestamp":20.0}'
        "]}\n"
    )

    X, Y = load_trajectory_jsonl(jsonl, pad=-999.0)

    assert X.shape == (2, 2, 5)
    assert Y.shape == (2, 2, 3)
    assert X[0, 0].tolist() == [0.0, 0.0, 0.0, 12.0, 8.0]
    assert Y[0, 0].tolist() == [4.0, 0.0, 10.0]
    assert Y[1, 0].tolist() == [3.0, 0.0, 20.0]
    assert Y[1, 1].tolist() == [-999.0, -999.0, -999.0]


def _path_points(n: int) -> list[dict]:
    return [{"x": float(i), "y": 0.0, "timestamp": float(i * 10)} for i in range(n)]


def test_load_jsonl_truncates_steps_above_max(tmp_path: Path):
    jsonl = tmp_path / "touch_data.jsonl"
    jsonl.write_text('{"target":{"x":139.0,"y":0.0},"path":' + str(_path_points(140)).replace("'", '"') + "}\n")

    X, Y = load_trajectory_jsonl(jsonl, pad=-999.0, max_steps=8)

    assert X.shape == (1, 8, 5)
    assert Y.shape == (1, 8, 3)
    assert Y[0, 0].tolist() == [1.0, 0.0, 10.0]
    assert Y[0, 7].tolist() == [1.0, 0.0, 10.0]
    assert X[0, 7].tolist() == [1.0, 0.0, 10.0, 132.0, 0.0]


def test_load_sequences_keeps_variable_length(tmp_path: Path):
    jsonl = tmp_path / "touch_data.jsonl"
    jsonl.write_text(
        '{"target":{"x":12.0,"y":0.0},"path":'
        + str(_path_points(13)).replace("'", '"')
        + "}\n"
        '{"target":{"x":3.0,"y":0.0},"path":'
        + str(_path_points(2)).replace("'", '"')
        + "}\n"
    )

    xs, ys = load_trajectory_sequences(jsonl, max_steps=8)

    assert [x.shape for x in xs] == [(8, 5), (1, 5)]
    assert [y.shape for y in ys] == [(8, 3), (1, 3)]
    assert ys[1][0].tolist() == [1.0, 0.0, 10.0]


def test_load_gzipped_jsonl(tmp_path: Path):
    raw = (
        '{"target":{"x":3.0,"y":0.0},"path":['
        '{"x":0.0,"y":0.0,"timestamp":0.0},'
        '{"x":3.0,"y":0.0,"timestamp":20.0}'
        "]}\n"
    )
    gz = tmp_path / "touch_data.jsonl.gz"
    with gzip.open(gz, "wt", encoding="utf-8") as f:
        f.write(raw)

    X, Y = load_trajectory_jsonl(gz, pad=-999.0)

    assert X.shape == (1, 1, 5)
    assert Y[0, 0].tolist() == [3.0, 0.0, 20.0]
