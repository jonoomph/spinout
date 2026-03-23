from src.sim.control_api import DriverCommand


def test_driver_command_from_action_accepts_legacy_accel_key():
    cmd = DriverCommand.from_action({"steer": 0.25, "accel": 0.5, "brake": 0.1})

    assert cmd.steer == 0.25
    assert cmd.throttle == 0.5
    assert cmd.brake == 0.1


def test_driver_command_from_action_accepts_throttle_key():
    cmd = DriverCommand.from_action({"steer": -0.25, "throttle": 0.75, "brake": 0.2})

    assert cmd.steer == -0.25
    assert cmd.throttle == 0.75
    assert cmd.brake == 0.2


def test_driver_command_from_action_prefers_throttle_when_both_keys_exist():
    cmd = DriverCommand.from_action({"accel": 0.1, "throttle": 0.8, "brake": 0.0})

    assert cmd.throttle == 0.8
