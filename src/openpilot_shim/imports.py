from __future__ import annotations

import os

try:
  from cereal import log as cereal_log  # type: ignore
except ImportError:
  cereal_log = None  # type: ignore

_use_openpilot = os.environ.get("CONTROL_SCAFFOLD_USE_OPENPILOT")
_use_openpilot = _use_openpilot is not None and _use_openpilot not in ("0", "")

if _use_openpilot:
  try:
    from openpilot.selfdrive.controls.lib.latcontrol import LatControl as _LatControl  # type: ignore
  except ImportError:
    _LatControl = None
else:
  _LatControl = None

if _use_openpilot:
  try:
    from openpilot.selfdrive.controls.lib.longcontrol import LongControl as _LongControl  # type: ignore
    from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState as _LongCtrlState  # type: ignore
  except ImportError:
    _LongControl = None
    _LongCtrlState = None
else:
  _LongControl = None
  _LongCtrlState = None

if _LatControl is not None:
  LatControlBase = _LatControl
else:
  from .base import BaseLatControl as LatControlBase

if _LongControl is not None:
  LongControlBase = _LongControl
  LongCtrlState = _LongCtrlState
else:
  from .base import BaseLongControl as LongControlBase

  class _StubLongCtrlState:
    off = "off"
    pid = "pid"
    stopping = "stopping"
    starting = "starting"

  LongCtrlState = _StubLongCtrlState


if cereal_log is not None:
  def make_lat_log():
    return cereal_log.ControlsState.LateralPIDState.new_message()
else:
  from .logs import LatControlLog

  def make_lat_log():
    return LatControlLog()
