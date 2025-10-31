import math

import pytest

from src.sim.planner import PlannerPreviewer


def build_arc_drive_line(radius: float = 50.0, points: int = 25):
    angles = [math.pi * 0.5 * i / (points - 1) for i in range(points)]
    return [
        (radius * math.cos(theta), radius * math.sin(theta))
        for theta in angles
    ]


def test_preview_uses_default_frequency():
    previewer = PlannerPreviewer()
    drive_line = build_arc_drive_line()
    total_length = radius_length(drive_line)
    speed_limits = [
        {
            "start_idx": 0,
            "end_idx": len(drive_line) - 1,
            "start_s": 0.0,
            "end_s": total_length,
            "speed_mph": 30,
            "sign_idx": 0,
        }
    ]
    previewer.set_plan(drive_line, speed_limits)
    position = (drive_line[0][0], 0.0, drive_line[0][1])
    preview = previewer.preview(position, speed=0.0)

    expected_steps = int(round(previewer.config.horizon_seconds * previewer.config.preview_hz))
    assert len(preview.speed) == expected_steps
    assert preview.dt == pytest.approx(0.1)
    assert all(pytest.approx(30 * 0.44704, rel=1e-6) == v for v in preview.speed)
    expected_lat = (30 * 0.44704) ** 2 / 50.0
    assert preview.lat_accel[0] == pytest.approx(expected_lat, rel=1e-3)

    heading = math.atan2(drive_line[1][1] - drive_line[0][1], drive_line[1][0] - drive_line[0][0])
    target = previewer.immediate_target(position, speed=0.0, heading=heading, preview=preview)
    assert target.speed == preview.speed[0]
    assert target.lat_accel == preview.lat_accel[0]
    assert target.lateral_error == pytest.approx(0.0)
    assert target.heading_error == pytest.approx(0.0)


def test_preview_respects_custom_rate():
    previewer = PlannerPreviewer()
    drive_line = build_arc_drive_line()
    total_length = radius_length(drive_line)
    speed_limits = [
        {
            "start_idx": 0,
            "end_idx": len(drive_line) - 1,
            "start_s": 0.0,
            "end_s": total_length,
            "speed_mph": 20,
            "sign_idx": 0,
        }
    ]
    previewer.set_plan(drive_line, speed_limits)
    position = (drive_line[0][0], 0.0, drive_line[0][1])

    preview = previewer.preview(position, speed=0.0, preview_hz=5.0)
    assert preview.dt == pytest.approx(0.2)
    assert len(preview.speed) == int(round(previewer.config.horizon_seconds * 5.0))


def test_projection_sign_convention():
    previewer = PlannerPreviewer()
    drive_line = build_arc_drive_line()
    total_length = radius_length(drive_line)
    speed_limits = [
        {
            "start_idx": 0,
            "end_idx": len(drive_line) - 1,
            "start_s": 0.0,
            "end_s": total_length,
            "speed_mph": 25,
            "sign_idx": 0,
        }
    ]
    previewer.set_plan(drive_line, speed_limits)
    start = (drive_line[0][0], 0.0, drive_line[0][1])
    left = (drive_line[0][0] - 1.0, 0.0, drive_line[0][1])
    right = (drive_line[0][0] + 1.0, 0.0, drive_line[0][1])

    base_target = previewer.immediate_target(start, speed=0.0, heading=math.pi / 2)
    left_target = previewer.immediate_target(left, speed=0.0, heading=math.pi / 2)
    right_target = previewer.immediate_target(right, speed=0.0, heading=math.pi / 2)

    assert base_target.lateral_error == pytest.approx(0.0)
    assert left_target.lateral_error > 0.0
    assert right_target.lateral_error < 0.0


def radius_length(drive_line):
    total = 0.0
    for a, b in zip(drive_line[:-1], drive_line[1:]):
        dx = b[0] - a[0]
        dz = b[1] - a[1]
        total += math.hypot(dx, dz)
    return total
