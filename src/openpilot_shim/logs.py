from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict


@dataclass
class LatControlLog:
  active: bool = False
  steeringAngleDeg: float = 0.0
  steeringAngleDesiredDeg: float = 0.0
  angleError: float = 0.0
  p: float = 0.0
  i: float = 0.0
  f: float = 0.0
  output: float = 0.0
  saturated: bool = False

  def to_dict(self) -> Dict[str, Any]:
    return asdict(self)
