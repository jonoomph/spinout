"""Controller base classes for the Spinout simulator."""

from __future__ import annotations

from typing import Optional

from src.sim.control_api import DriverCommand, TelemetrySnapshot


class BaseController:
    """Base class for driving controllers.

    Subclasses implement :meth:`step` to return a :class:`DriverCommand` each
    time the environment polls them.  The environment takes care of scheduling
    so controllers can declare a slower ``control_rate_hz`` than the physics
    engine.
    """

    def __init__(
        self,
        name: str = "controller",
        *,
        control_rate_hz: float = 10.0,
        preview_rate_hz: float | None = None,
    ) -> None:
        self.name = name
        self.control_rate_hz = float(control_rate_hz)
        self.preview_rate_hz = float(preview_rate_hz) if preview_rate_hz else None
        self.enabled: bool = False
        self.dt: Optional[float] = None
        self.physics_dt: Optional[float] = None
        self._env = None

    # ------------------------------------------------------------------
    # Lifecycle helpers

    def attach(self, env) -> None:
        """Attach the controller to an environment instance."""

        self._env = env
        self.physics_dt = getattr(env, "dt", None)
        rate = max(self.control_rate_hz, 0.0)
        if rate > 0.0:
            self.dt = 1.0 / rate
        else:
            self.dt = self.physics_dt
        self.on_attach()

    def detach(self) -> None:
        if self._env is not None:
            self.disable()
            self.on_detach()
            self._env = None

    def enable(self) -> None:
        if not self.enabled:
            self.enabled = True
            self.reset()

    def disable(self) -> None:
        if self.enabled:
            self.enabled = False
            self.on_disable()

    def toggle(self) -> bool:
        """Toggle the controller enabled state and return the new state."""

        if self.enabled:
            self.disable()
        else:
            self.enable()
        return self.enabled

    # ------------------------------------------------------------------
    # Hooks for subclasses

    def on_attach(self) -> None:
        """Hook called when the controller is attached to an environment."""

    def on_detach(self) -> None:
        """Hook called when the controller is detached."""

    def reset(self) -> None:
        """Reset internal state when the controller is enabled or the env resets."""

    def on_disable(self) -> None:
        """Hook called when the controller is disabled."""

    # ------------------------------------------------------------------
    # Action update API

    def step(
        self, telemetry: TelemetrySnapshot, manual: DriverCommand
    ) -> DriverCommand:
        """Return the command to apply for the next simulation step."""

        return manual
