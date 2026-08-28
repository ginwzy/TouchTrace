"""CSD4CA touch CSV → training jsonl. Tests the convert_touch_csv seam."""

import csv
import json
from pathlib import Path

from convert_swipemotiondb import (
    convert_touch_csv,
    interpolate_xyz,
    merge_sensors,
    parse_sensor_csv,
    write_jsonl,
)

V2_FIELDS = [
    "session",
    "scenario",
    "user_id",
    "used_hand",
    "age",
    "gender",
    "id_swipe",
    "time",
    "x",
    "y",
    "touch_major",
    "touch_minor",
    "pressure",
    "finger_size",
]


def write_csv(path: Path, rows: list[dict]) -> Path:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=V2_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return path


def point(
    swipe: str,
    t: float,
    x: float,
    y: float,
    *,
    scenario="Normal",
    session="1",
    user_id="10",
    pressure=0.4,
    touch_major=176.0,
    touch_minor=102.0,
    finger_size=0.07,
):
    return {
        "session": session,
        "scenario": scenario,
        "user_id": user_id,
        "used_hand": "r",
        "age": "21",
        "gender": "f",
        "id_swipe": swipe,
        "time": str(t),
        "x": str(x),
        "y": str(y),
        "touch_major": str(touch_major),
        "touch_minor": str(touch_minor),
        "pressure": str(pressure),
        "finger_size": str(finger_size),
    }


def test_valid_swipe_becomes_relative_path_with_target_at_last_point(tmp_path: Path):
    csv_path = write_csv(
        tmp_path / "touch.csv",
        [
            point("100", 1000, 100, 800, pressure=0.2, touch_major=120),
            point("100", 1016, 140, 760, pressure=0.5, touch_major=160),
            point("100", 1100, 540, 1200, pressure=0.3, touch_major=140),
        ],
    )

    out = convert_touch_csv(csv_path)

    assert len(out) == 1
    traj = out[0]
    assert traj["length"] == 3
    assert traj["target"] == {"x": 540.0, "y": 1200.0}
    assert traj["path"] == [
        {"x": 100.0, "y": 800.0, "timestamp": 0.0, "pressure": 0.2, "area": 120.0},
        {"x": 140.0, "y": 760.0, "timestamp": 16.0, "pressure": 0.5, "area": 160.0},
        {"x": 540.0, "y": 1200.0, "timestamp": 100.0, "pressure": 0.3, "area": 140.0},
    ]
    assert traj["meta"] == {
        "condition": "seated",
        "session": 1,
        "user_id": "10",
        "device": "pixel-6a",
        "id_swipe": "100",
    }


def test_same_millisecond_keeps_the_last_point(tmp_path: Path):
    csv_path = write_csv(
        tmp_path / "touch.csv",
        [
            point("1", 0, 10, 10),
            point("1", 20, 20, 20, pressure=0.1, touch_major=50),
            point("1", 20, 25, 30, pressure=0.9, touch_major=200),
            point("1", 80, 40, 40),
        ],
    )

    traj = convert_touch_csv(csv_path)[0]

    assert traj["length"] == 3
    assert traj["path"][1] == {
        "x": 25.0,
        "y": 30.0,
        "timestamp": 20.0,
        "pressure": 0.9,
        "area": 200.0,
    }


def test_drops_swipes_that_are_too_short_or_have_a_large_gap(tmp_path: Path):
    csv_path = write_csv(
        tmp_path / "touch.csv",
        [
            point("two", 0, 0, 0),
            point("two", 80, 10, 10),
            point("brief", 0, 0, 0),
            point("brief", 10, 1, 1),
            point("brief", 40, 2, 2),
            point("gap", 0, 0, 0),
            point("gap", 10, 1, 1),
            point("gap", 600, 2, 2),
            point("ok", 0, 0, 0),
            point("ok", 16, 8, 8),
            point("ok", 80, 16, 16),
        ],
    )

    out = convert_touch_csv(csv_path)
    assert len(out) == 1
    assert out[0]["path"][-1]["x"] == 16.0


def test_maps_walking_and_stressful_conditions(tmp_path: Path):
    csv_path = write_csv(
        tmp_path / "touch.csv",
        [
            point("w", 0, 0, 0, scenario="Walking", user_id="4"),
            point("w", 16, 1, 1, scenario="Walking", user_id="4"),
            point("w", 80, 2, 2, scenario="Walking", user_id="4"),
            point("s", 0, 0, 0, scenario="Stressful", user_id="5", session="2"),
            point("s", 16, 1, 1, scenario="Stressful", user_id="5", session="2"),
            point("s", 80, 2, 2, scenario="Stressful", user_id="5", session="2"),
        ],
    )

    out = {t["meta"]["user_id"]: t["meta"] for t in convert_touch_csv(csv_path)}
    assert out["4"]["condition"] == "walking"
    assert out["5"]["condition"] == "stress"
    assert out["5"]["session"] == 2


def test_skips_rows_with_empty_time(tmp_path: Path):
    csv_path = write_csv(
        tmp_path / "touch.csv",
        [
            point("1", 0, 0, 0),
            {**point("1", 16, 1, 1), "time": ""},
            point("1", 16, 2, 2),
            point("1", 80, 3, 3),
        ],
    )

    traj = convert_touch_csv(csv_path)[0]
    assert traj["length"] == 3
    assert [p["x"] for p in traj["path"]] == [0.0, 2.0, 3.0]


def test_write_jsonl_round_trips_a_trajectory(tmp_path: Path):
    csv_path = write_csv(
        tmp_path / "touch.csv",
        [
            point("1", 0, 100, 800),
            point("1", 16, 140, 760),
            point("1", 80, 200, 700),
        ],
    )
    out_path = tmp_path / "touch_data.jsonl"
    write_jsonl(convert_touch_csv(csv_path), out_path)

    lines = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert len(lines) == 1
    assert lines[0]["target"] == {"x": 200.0, "y": 700.0}
    assert lines[0]["path"][0]["timestamp"] == 0.0


SENSOR_FIELDS = [
    "session",
    "scenario",
    "user_id",
    "used_hand",
    "age",
    "gender",
    "id_swipe",
    "time",
    "accuracy",
    "x",
    "y",
    "z",
]


def write_sensor_csv(path: Path, rows: list[dict]) -> Path:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SENSOR_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return path


def sensor_row(swipe: str, t_ns: float, x: float, y: float, z: float) -> dict:
    return {
        "session": "1",
        "scenario": "Normal",
        "user_id": "10",
        "used_hand": "r",
        "age": "21",
        "gender": "f",
        "id_swipe": swipe,
        "time": str(t_ns),
        "accuracy": "1.0",
        "x": str(x),
        "y": str(y),
        "z": str(z),
    }


def test_interpolate_uses_relative_ms_not_absolute_clocks():
    # Sensor clock is ns starting at 1e15; queries are touch-relative ms.
    t0 = 1e15
    samples = [
        (t0, 0.0, 0.0, 9.8),
        (t0 + 20e6, 20.0, 0.0, 9.8),
    ]
    out = interpolate_xyz(samples, [0.0, 10.0, 20.0])
    assert out is not None
    assert abs(out[0][0] - 0.0) < 1e-9
    assert abs(out[1][0] - 10.0) < 1e-9
    assert abs(out[2][0] - 20.0) < 1e-9
    assert abs(out[1][2] - 9.8) < 1e-9


def test_merge_sensors_aligns_by_id_swipe_and_relative_time(tmp_path: Path):
    touch = write_csv(
        tmp_path / "touch.csv",
        [
            point("100", 5000, 0, 0),
            point("100", 5040, 10, 0),
            point("100", 5080, 20, 0),
        ],
    )
    t0 = 7.85e14
    acc = write_sensor_csv(
        tmp_path / "acc.csv",
        [
            sensor_row("100", t0, 0.0, 0.0, 9.8),
            sensor_row("100", t0 + 80e6, 2.0, 0.0, 9.8),
        ],
    )
    gyro = write_sensor_csv(
        tmp_path / "gyro.csv",
        [
            sensor_row("100", t0, 0.0, 0.0, 0.0),
            sensor_row("100", t0 + 80e6, 0.4, 0.0, 0.0),
        ],
    )

    fused, stats = merge_sensors(
        convert_touch_csv(touch),
        parse_sensor_csv(acc),
        parse_sensor_csv(gyro),
    )

    assert stats["kept"] == 1
    traj = fused[0]
    assert len(traj["sensors"]) == 3
    assert [s["timestamp"] for s in traj["sensors"]] == [p["timestamp"] for p in traj["path"]]
    assert abs(traj["sensors"][1]["accel"][0] - 1.0) < 1e-6
    assert abs(traj["sensors"][1]["gyro"][0] - 0.2) < 1e-6
    assert "pressure" not in traj["path"][0]


def test_merge_sensors_drops_missing_gyro_and_span_outliers(tmp_path: Path):
    touch = write_csv(
        tmp_path / "touch.csv",
        [
            point("ok", 0, 0, 0),
            point("ok", 16, 4, 0),
            point("ok", 80, 8, 0),
            point("nogyro", 0, 0, 0),
            point("nogyro", 16, 4, 0),
            point("nogyro", 80, 8, 0),
            point("longacc", 0, 0, 0),
            point("longacc", 16, 4, 0),
            point("longacc", 80, 8, 0),
        ],
    )
    t0 = 1e12
    acc = write_sensor_csv(
        tmp_path / "acc.csv",
        [
            sensor_row("ok", t0, 0, 0, 9.8),
            sensor_row("ok", t0 + 80e6, 0, 0, 9.8),
            sensor_row("nogyro", t0, 0, 0, 9.8),
            sensor_row("nogyro", t0 + 80e6, 0, 0, 9.8),
            sensor_row("longacc", t0, 0, 0, 9.8),
            sensor_row("longacc", t0 + 80e6 * 50, 0, 0, 9.8),
        ],
    )
    gyro = write_sensor_csv(
        tmp_path / "gyro.csv",
        [
            sensor_row("ok", t0, 0, 0, 0),
            sensor_row("ok", t0 + 80e6, 0, 0, 0),
            sensor_row("longacc", t0, 0, 0, 0),
            sensor_row("longacc", t0 + 80e6, 0, 0, 0),
        ],
    )

    fused, stats = merge_sensors(
        convert_touch_csv(touch),
        parse_sensor_csv(acc),
        parse_sensor_csv(gyro),
    )
    assert stats["kept"] == 1
    assert stats["missing"] == 1
    assert stats["span_drop"] == 1
    assert fused[0]["meta"]["condition"] == "seated"


def test_write_jsonl_gzip(tmp_path: Path):
    csv_path = write_csv(
        tmp_path / "touch.csv",
        [
            point("1", 0, 100, 800),
            point("1", 16, 140, 760),
            point("1", 80, 200, 700),
        ],
    )
    out_path = tmp_path / "touch_data.jsonl.gz"
    write_jsonl(convert_touch_csv(csv_path), out_path)
    import gzip

    lines = [json.loads(line) for line in gzip.open(out_path, "rt", encoding="utf-8")]
    assert len(lines) == 1
    assert lines[0]["meta"]["id_swipe"] == "1"
