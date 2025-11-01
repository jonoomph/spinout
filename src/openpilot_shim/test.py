import math
import os

import pytest

from .controllers import LatControlStub, LongControlStub
from .datatypes import (
  SimCarInterface,
  SimCarParams,
  SimCarState,
  SimLiveParameters,
  SimVehicleModel,
)

USE_OPENPILOT = os.environ.get("CONTROL_SCAFFOLD_USE_OPENPILOT")
USE_OPENPILOT = USE_OPENPILOT is not None and USE_OPENPILOT not in ("0", "")


@pytest.fixture
def sample_inputs():
  if USE_OPENPILOT:
    from cereal import car, log
    from opendbc.car.vehicle_model import VehicleModel

    CP = car.CarParams.new_message()
    CP.steerLimitTimer = 0.5
    CP.steerControlType = car.CarParams.SteerControlType.torque
    CP.stopAccel = -2.0
    CP.stoppingDecelRate = 0.2
    CP.startAccel = 0.5
    CP.vEgoStarting = 0.3
    CP.startingState = False
    CP.minSteerSpeed = 0.0

    CP.longitudinalTuning.kpBP = [0.0]
    CP.longitudinalTuning.kpV = [1.0]
    CP.longitudinalTuning.kiBP = [0.0]
    CP.longitudinalTuning.kiV = [0.1]
    CP.longitudinalTuning.kf = 1.0

    CI = object()

    CS = car.CarState.new_message()
    CS.vEgo = 12.5
    CS.aEgo = 0.1
    CS.steeringAngleDeg = -1.5
    CS.steeringRateDeg = -3.0
    CS.steeringPressed = False
    CS.brakePressed = False
    CS.standstill = False
    CS.cruiseState.standstill = False

    params = log.LiveParametersData.new_message()
    params.steerRatio = 12.0
    params.stiffnessFactor = 0.9
    params.angleOffsetDeg = 0.2
    params.roll = math.radians(1.0)

    VM = VehicleModel(CP)
    VM.update_params(params.stiffnessFactor, params.steerRatio)

    return CP, CI, CS, params, VM

  CP = SimCarParams()
  CI = SimCarInterface()
  CS = SimCarState(
    vEgo=12.5,
    aEgo=0.1,
    steeringAngleDeg=-1.5,
    steeringRateDeg=-3.0,
    steeringPressed=False,
    brakePressed=False,
    standstill=False,
  )
  params = SimLiveParameters(
    steerRatio=12.0,
    stiffnessFactor=0.9,
    angleOffsetDeg=0.2,
    roll=math.radians(1.0),
  )
  VM = SimVehicleModel()
  VM.update_params(params.stiffnessFactor, params.steerRatio)

  return CP, CI, CS, params, VM


def test_lat_control_update_returns_expected_types(sample_inputs):
  CP, CI, CS, params, VM = sample_inputs

  controller = LatControlStub(CP, CI)
  steer, desired_angle, log_msg = controller.update(
    active=True,
    CS=CS,
    VM=VM,
    params=params,
    steer_limited_by_controls=False,
    desired_curvature=0.01,
    calibrated_pose=None,
    curvature_limited=False,
  )

  assert isinstance(steer, float)
  assert isinstance(desired_angle, float)
  assert hasattr(log_msg, "active")
  assert getattr(log_msg, "active") is True


def test_long_control_update_returns_float(sample_inputs):
  CP, _, CS, _, _ = sample_inputs

  controller = LongControlStub(CP)
  accel = controller.update(
    active=True,
    CS=CS,
    a_target=0.3,
    should_stop=False,
    accel_limits=(-2.5, 1.8),
  )

  assert isinstance(accel, float)
  assert accel == pytest.approx(0.0)
