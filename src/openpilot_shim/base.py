from __future__ import annotations

class BaseLatControl:
  """Minimal replica of openpilot LatControl interface."""

  def __init__(self, CP, CI):
    self.CP = CP
    self.CI = CI
    self.steer_max = 1.0
    self.sat_count = 0.0

  def reset(self) -> None:
    self.sat_count = 0.0

  def update(self, *args, **kwargs):
    raise NotImplementedError


class BaseLongControl:
  """Minimal replica of openpilot LongControl interface."""

  def __init__(self, CP):
    self.CP = CP
    self.long_control_state = "off"

  def reset(self) -> None:
    self.long_control_state = "off"

  def update(self, *args, **kwargs):
    raise NotImplementedError
