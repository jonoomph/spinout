# environment.py
"""Gym-style environment wrapper for the Spinout simulator.

This module exposes a minimal :class:`Environment` class that follows the
`gymnasium` API.  The environment is intentionally lightweight – in
``train`` mode nothing is rendered which makes it suitable for automated
testing.  When constructed with ``mode="eval"`` the caller can initialise
the renderer via :meth:`init_renderer` to display a pygame window each step.

To keep the module focused the unit test helpers that previously lived here
have been moved back into ``tests/tests.py`` where they belong.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple
from itertools import product

import numpy as np

from .physics import Terrain, Car
from .roads.plan import generate_plan, get_safe_start_position_and_rot
from .roads.build import apply_plan, build_road_vertices, build_speed_sign_vertices
from .colors import (
    ROAD_ASPHALT_COLOR,
    ROAD_CONCRETE_COLOR,
    ROAD_GRAVEL_COLOR,
    TERRAIN_GRASS_COLOR,
    TERRAIN_SAND_COLOR,
    TERRAIN_SNOW_COLOR,
    TERRAIN_DIRT_COLOR,
)
from .effects import SkidMarkSystem
from .buildings import generate_buildings
from .wind import WindSystem, WindSample
from .control_api import DriverCommand, TelemetrySnapshot, VehicleState
from .planner import PlannerPreviewer

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from src.controllers.controller import BaseController


# ---------------------------------------------------------------------------
# Constants shared with the original ``game.py`` script.  They are kept here so
# both the interactive game and the programmatic environment can use the same
# defaults without duplicating definitions.  ``WIDTH`` and ``HEIGHT`` are
# populated at runtime based on the current display.

WIDTH = 0
HEIGHT = 0

WEATHER_MODIFIERS = {"dry": 1.0, "wet": 0.7}

ROAD_TYPES = {
    "asphalt": {"color": ROAD_ASPHALT_COLOR, "friction": 1.0},
    "concrete": {"color": ROAD_CONCRETE_COLOR, "friction": 0.95},
    "gravel": {"color": ROAD_GRAVEL_COLOR, "friction": 0.8},
}

TERRAIN_TYPES = {
    "grass": {"color": TERRAIN_GRASS_COLOR, "friction": 0.7},
    "sand": {"color": TERRAIN_SAND_COLOR, "friction": 0.6},
    "dirt": {"color": TERRAIN_DIRT_COLOR, "friction": 0.6},
    "snow": {"color": TERRAIN_SNOW_COLOR, "friction": 0.5},
}

SURFACES = list(product(WEATHER_MODIFIERS, TERRAIN_TYPES))

# Absolute path to bundled data (e.g. car definitions)
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


# ---------------------------------------------------------------------------
# Simple Gym-style environment ------------------------------------------------


class Environment:
    """Spinout simulator wrapped in a Gym-style API.

    Parameters
    ----------
    cfg:
        Optional dictionary with configuration overrides.  The most relevant
        keys are:

        ``dt`` (float):      Size of each simulation step in seconds.
        ``substeps`` (int):  Number of physics sub-steps per step.
        ``max_steps`` (int): Maximum number of steps per episode.
        ``time_limit`` (float): Optional wall clock time limit for an episode.
        ``cost_limit`` (float): Optional accumulated cost limit for an episode.
        ``flat`` (bool):     When ``True`` the environment is a simple flat
                             terrain used by the unit tests.
        ``car_index`` (int): Index into ``data/cars.json`` selecting which car
                             to spawn.

    mode:
        ``"train"`` or ``"eval"``.  In evaluation mode the simulator renders a
        pygame window each step.  In training mode no rendering occurs which
        keeps the simulation lightweight.
    """

    def __init__(
        self,
        cfg: Optional[Dict] = None,
        mode: str = "train",
        status_callback: Optional[Callable[[float, str], None]] = None,
    ):
        self.cfg = cfg or {}
        self.mode = mode
        self.status_callback = status_callback
        self._status_progress = 0.0
        self._status_message = ""

        # Simulation parameters -------------------------------------------------
        self.dt = float(self.cfg.get("dt", 0.01))
        self.substeps = int(self.cfg.get("substeps", 5))
        self.max_steps = (
            int(self.cfg["max_steps"]) if "max_steps" in self.cfg else None
        )
        self.time_limit = (
            float(self.cfg["time_limit"]) if "time_limit" in self.cfg else None
        )
        self.cost_limit = (
            float(self.cfg["cost_limit"]) if "cost_limit" in self.cfg else None
        )

        # Internal state --------------------------------------------------------
        self.rng = np.random.default_rng(self.cfg.get("seed"))
        self.step_count = 0
        self.time = 0.0
        self.episode_cost = 0.0
        self.plan: Dict = {}
        self.termination_reason: Optional[str] = None
        self._planner = PlannerPreviewer()
        self._telemetry = TelemetrySnapshot()
        self._prev_velocity = np.zeros(3, dtype=float)
        self._controller: Optional[BaseController] = None
        self._controller_period = 0.0
        self._controller_timer = 0.0
        self._controller_last_command: Optional[DriverCommand] = None
        self._last_driver_command = DriverCommand()
        self._controller_hud = ("No Controller", "Manual Steer")
        self._controller_enabled_last_step = False

        # Rendering state ------------------------------------------------------
        self.render_mode = 1
        self.camera_mode = 0
        self.use_bbmodel = False
        self.surface_idx = 0
        self.surface_info = ""
        self.car_info = ""
        self._last_car_camera_mode = 0
        self._free_camera_active = False
        self.free_cam_pos = np.zeros(3, dtype=float)
        self.free_cam_yaw = 0.0
        self.free_cam_pitch = 0.0
        self.free_cam_speed = 20.0
        self.road_layers: dict[str, tuple] = {}

        # Window dimensions (populated when the renderer initialises)
        self.width = 0
        self.height = 0

        # Placeholders; the actual world is created on ``reset``
        self.terrain: Optional[Terrain] = None
        self.car: Optional[Car] = None
        self.rp = None
        self.skidmarks = SkidMarkSystem()
        self.wind_system: Optional[WindSystem] = None
        self.wind_sample: Optional[WindSample] = None
        self.wind_vectors_enabled = False

    # ------------------------------------------------------------------
    # World generation helpers

    def attach_controller(self, controller: Optional["BaseController"]) -> None:
        """Attach ``controller`` so :meth:`step` can poll it automatically."""

        if controller is self._controller:
            return
        if self._controller is not None:
            self._controller.detach()
        self._controller = controller
        self._controller_last_command = None
        self._controller_timer = 0.0
        self._update_controller_period()
        if controller is not None:
            controller.attach(self)
            controller.reset()
        self._controller_enabled_last_step = bool(controller and controller.enabled)
        self._controller_hud = self._resolve_hud_labels()

    def _update_controller_period(self) -> None:
        if self._controller is None:
            self._controller_period = 0.0
            return
        rate = max(float(self._controller.control_rate_hz), 1e-3)
        self._controller_period = 1.0 / rate

    def _resolve_hud_labels(self) -> Tuple[str, str]:
        if self._controller is not None and self._controller.enabled:
            module = self._controller.__class__.__module__.split(".")[-1]
            controller_name = (
                f"{module}.py controller" if module else self._controller.__class__.__name__
            )
            if module == "pid":
                steer_label = "PID Steer"
            else:
                pretty = self._controller.__class__.__name__.replace("Controller", "").strip()
                if not pretty:
                    pretty = module or "Auto"
                steer_label = f"{pretty.replace('_', ' ')} Steer"
            return controller_name, steer_label
        return "No Controller", "Manual Steer"

    def _compute_state(self, prev_velocity: np.ndarray) -> VehicleState:
        if self.car is None:
            return VehicleState()
        vel = self.car.body.vel.copy()
        pos = self.car.body.pos.copy()
        dt = max(self.dt, 1e-6)
        accel = (vel - prev_velocity) / dt
        fwd = self.car.body.rot.rotate(np.array([0, 0, 1]))
        right = self.car.body.rot.rotate(np.array([1, 0, 0]))
        speed = float(np.linalg.norm(vel))
        v_ego = float(np.dot(vel, fwd))
        lat_vel_right = float(np.dot(vel, right))
        lat_acc_right = float(np.dot(accel, right))
        long_acc = float(np.dot(accel, fwd))
        roll_axis = self.car.body.rot.rotate(np.array([0, 1, 0]))
        roll = float(math.atan2(roll_axis[0], roll_axis[1]))
        roll_lat = 9.81 * math.sin(roll)
        yaw = float(math.atan2(fwd[0], fwd[2]))
        yaw_rate = float(np.dot(self.car.body.angvel, np.array([0.0, 1.0, 0.0])))

        # Keep all intermediate blending in the "right-hand" frame so the
        # existing heuristics continue to behave the same.  Afterwards convert
        # back to a left-positive convention so the telemetry matches the road
        # planner, which treats positive lateral quantities as "left of path".
        centripetal_right = -v_ego * yaw_rate
        if abs(lat_acc_right) < abs(centripetal_right) * 0.25 or lat_acc_right * centripetal_right <= 0:
            lat_acc_right = centripetal_right
        else:
            lat_acc_right = 0.5 * (lat_acc_right + centripetal_right)
        lat_vel = -lat_vel_right
        lat_acc = -lat_acc_right
        return VehicleState(
            speed=speed,
            v_ego=v_ego,
            lat_velocity=lat_vel,
            lat_accel=lat_acc,
            long_accel=long_acc,
            roll=roll,
            roll_lataccel=roll_lat,
            yaw=yaw,
            yaw_rate=yaw_rate,
            position=(float(pos[0]), float(pos[1]), float(pos[2])),
            velocity=(float(vel[0]), float(vel[1]), float(vel[2])),
        )

    def _build_snapshot(
        self,
        prev_velocity: np.ndarray,
        preview_hz: Optional[float] = None,
    ) -> TelemetrySnapshot:
        if self.car is None:
            return TelemetrySnapshot()
        state = self._compute_state(prev_velocity)
        preview = self._planner.preview(self.car.body.pos, state.speed, preview_hz)
        target = self._planner.immediate_target(
            self.car.body.pos,
            state.speed,
            state.yaw,
            preview,
        )
        return TelemetrySnapshot(state=state, target=target, future=preview)

    def _refresh_initial_telemetry(self) -> None:
        if self.car is None:
            self._telemetry = TelemetrySnapshot()
            self._prev_velocity = np.zeros(3, dtype=float)
            return
        vel = self.car.body.vel.copy()
        self._prev_velocity = vel.copy()
        self._telemetry = self._build_snapshot(vel)

    def _set_status(self, progress: float, message: str) -> None:
        self._status_progress = progress
        self._status_message = message
        if self.status_callback:
            self.status_callback(progress, message)

    def status(self) -> Tuple[float, str]:
        """Return the last status update produced during setup."""
        return self._status_progress, self._status_message

    def _build_world(self) -> None:
        """(Re)create terrain, roads and the player's car."""

        if getattr(self, "_free_camera_active", False):
            self._exit_free_camera()
        self.camera_mode = 0
        self._last_car_camera_mode = 0
        self.wind_vectors_enabled = False

        self._set_status(0.2, "Generating terrain...")
        if self.cfg.get("flat"):
            # Deterministic flat terrain used for the physics tests
            self.weather = "dry"
            self.road_type = "asphalt"
            self.terrain_type = "asphalt"
            self.precipitation = "none"
            self.precipitation_strength = 0.0
            self.terrain = Terrain(
                res=120,
                height_scale=0,
                sigma=0,
                terrain_type="asphalt",
                color=[0.2, 0.2, 0.2, 1.0],
            )
            self.terrain.heights[:] = 0
            rp = None
            self.plan = {}
            self._planner.set_plan(None, None)
            self.buildings = {"vertices": np.zeros((0, 10), dtype="f4"), "instances": [], "palette": None, "noise_scale": 0.0}
            self.wind_system = WindSystem(self.rng, self.weather, self.precipitation, calm=True)
            self.wind_sample = self.wind_system.update(0.0)
        else:
            # Procedurally generate a random driving environment
            precip_override = self.cfg.get("precipitation")
            if precip_override not in ("none", "rain"):
                precip_override = None
            strength_override = self.cfg.get("precipitation_strength")
            if strength_override is not None:
                try:
                    strength_override = float(strength_override)
                except (TypeError, ValueError):
                    strength_override = None
            if strength_override is not None:
                strength_override = float(np.clip(strength_override, 0.0, 1.0))

            self.precipitation = "none"
            self.precipitation_strength = 0.0

            weather_override = self.cfg.get("weather")
            if weather_override in ("dry", "wet"):
                self.weather = weather_override
            else:
                self.weather = "wet" if float(self.rng.random()) < 0.30 else "dry"

            if precip_override is not None:
                self.precipitation = precip_override
                if self.precipitation == "rain":
                    if strength_override is not None:
                        self.precipitation_strength = strength_override
                    else:
                        self.precipitation_strength = float(self.rng.random())
                else:
                    self.precipitation_strength = 0.0
            else:
                if self.weather == "wet" and float(self.rng.random()) < 0.5:
                    self.precipitation = "rain"
                    if strength_override is not None:
                        self.precipitation_strength = strength_override
                    else:
                        self.precipitation_strength = float(self.rng.random())
                else:
                    self.precipitation = "none"
                    self.precipitation_strength = 0.0

            if (
                self.precipitation == "rain"
                and self.weather != "wet"
                and weather_override not in ("dry", "wet")
            ):
                self.weather = "wet"
            self.road_type = self.rng.choice(list(ROAD_TYPES), p=[0.7, 0.2, 0.1])
            self.terrain_type = self.rng.choice(
                list(TERRAIN_TYPES), p=[0.55, 0.15, 0.15, 0.15]
            )

            weather_mod = WEATHER_MODIFIERS[self.weather]
            t = TERRAIN_TYPES[self.terrain_type]
            self.terrain = Terrain(
                res=120,
                terrain_type=self.terrain_type,
                color=t["color"],
                friction=t["friction"] * weather_mod,
            )

            rp, plan = generate_plan(
                self.terrain,
                rng=self.rng,
                road_type=self.road_type,
                weather=self.weather,
                terrain_type=self.terrain_type,
                road_color=ROAD_TYPES[self.road_type]["color"],
                skirt_color=self.terrain.color,
                road_friction=ROAD_TYPES[self.road_type]["friction"] * weather_mod,
            )
            self._set_status(0.5, "Laying roads...")
            apply_plan(self.terrain, rp, plan, rng=self.rng)
            self.plan = plan
            self._planner.set_plan(
                self.plan.get("drive_line"),
                self.plan.get("speed_limits"),
            )
            self.buildings = generate_buildings(self.terrain, rp, plan, rng=self.rng)
            self.wind_system = WindSystem(
                self.rng,
                self.weather,
                getattr(self, "precipitation", "none"),
            )
            self.wind_sample = self.wind_system.update(0.0)

        self.rp = rp

        if self.terrain_type in TERRAIN_TYPES:
            self.surface_idx = SURFACES.index((self.weather, self.terrain_type))
        else:
            self.surface_idx = 0
        precip_text = "Rainy" if getattr(self, "precipitation", "none") == "rain" else "Clear"
        self.surface_info = (
            f"{precip_text} {self.weather.title()} {self.road_type.title()} | {self.terrain_type.title()}"
        )

        self._set_status(0.7, "Loading car...")

        # Load car data and spawn the car --------------------------------------
        with open(DATA_DIR / "cars.json") as f:
            cars = json.load(f)

        car_index = int(self.cfg.get("car_index", 0))
        car_data = cars[car_index]
        self.car = Car(self.terrain, car_data)
        self.car.show_wind_vectors = self.wind_vectors_enabled
        self.car_info = f"{car_data['make']} {car_data['model']} ({car_data['year']})"

        if self.cfg.get("flat"):
            start_x = self.terrain.width / 4
            start_z = self.terrain.height / 4
            rest_y = (
                self.terrain.get_height(start_x, start_z)
                + self.car.cg_height_m
            )
            self.car.body.pos = np.array([start_x, rest_y, start_z])
        else:
            pos, rot = get_safe_start_position_and_rot(self.terrain, rp, 15.0)
            self.car.body.pos, self.car.body.rot = pos, rot

        self.skidmarks.reset()

        # Reset counters --------------------------------------------------------
        self.step_count = 0
        self.time = 0.0
        self.episode_cost = 0.0
        self.termination_reason = None

    def _set_surface(self, weather: str, terrain_type: str) -> None:
        """Update terrain/weather without regenerating the world."""
        weather_mod = WEATHER_MODIFIERS[weather]
        t = TERRAIN_TYPES[terrain_type]
        self.weather = weather
        self.terrain_type = terrain_type
        precip_text = "Rainy" if getattr(self, "precipitation", "none") == "rain" else "Clear"
        self.surface_info = (
            f"{precip_text} {weather.title()} {self.road_type.title()} | {terrain_type.title()}"
        )
        calm = bool(self.cfg.get("flat"))
        self.wind_system = WindSystem(
            self.rng,
            weather,
            getattr(self, "precipitation", "none"),
            calm=calm,
        )
        self.wind_sample = self.wind_system.update(0.0)
        self.render_ctx.setup_weather(
            weather,
            terrain_type,
            self.road_type,
            getattr(self, "precipitation", "none"),
            getattr(self, "precipitation_strength", 0.0),
        )

        self.terrain.terrain_type = terrain_type
        self.terrain.color = t["color"]
        friction = t["friction"] * weather_mod
        self.terrain.base_friction = friction
        self.terrain.surface_friction.fill(friction)
        road_mu = ROAD_TYPES[self.road_type]["friction"] * weather_mod
        self.terrain.road_friction[self.terrain.road_friction > 0] = road_mu
        col = np.array(t["color"], dtype="f4")
        if hasattr(self, "t_vertices") and getattr(self, "t_vbo", None) is not None:
            self.t_vertices[:, 3:7] = col
            self.t_vbo.write(self.t_vertices.tobytes())

        if self.plan:
            self.plan["skirt_color"] = list(map(float, t["color"]))
            if getattr(self, "render_ctx", None):
                self._build_road_layers()

    def _cycle_surface(self) -> None:
        self.surface_idx = (self.surface_idx + 1) % len(SURFACES)
        weather, terrain = SURFACES[self.surface_idx]
        self._set_surface(weather, terrain)

    def switch_car(self, car_index: int) -> None:
        """Swap to a different car without rebuilding the world.

        The simulation counters are reset but terrain, road and weather remain
        unchanged.  The new car spawns at the original start position so the
        user can quickly compare vehicles on the same track.
        """

        with open(DATA_DIR / "cars.json") as f:
            cars = json.load(f)

        car_index = int(car_index)
        car_data = cars[car_index]

        # Spawn new car at the same safe start used during reset
        if self.cfg.get("flat"):
            start_x = self.terrain.width / 4
            start_z = self.terrain.height / 4
            ground_h = self.terrain.get_height(start_x, start_z)
            rot = self.car.body.rot
            self.car = Car(self.terrain, car_data)
            self.car.show_wind_vectors = self.wind_vectors_enabled
            self.car.body.pos = np.array([start_x, ground_h + self.car.cg_height_m, start_z])
            self.car.body.rot = rot
        else:
            pos, rot = get_safe_start_position_and_rot(self.terrain, self.rp, 15.0)
            self.car = Car(self.terrain, car_data)
            self.car.show_wind_vectors = self.wind_vectors_enabled
            self.car.body.pos, self.car.body.rot = pos, rot
        self.car_info = f"{car_data['make']} {car_data['model']} ({car_data['year']})"
        self.cfg["car_index"] = car_index

        self.skidmarks.reset()

        # Reset episode counters
        self.step_count = 0
        self.time = 0.0
        self.episode_cost = 0.0
        self.termination_reason = None

    # ------------------------------------------------------------------
    # Rendering helpers

    def _compute_follow_camera_pose(self, mode: int, car_dir: np.ndarray, car_up_vec: np.ndarray):
        car_right = self.car.body.rot.rotate(np.array([1, 0, 0]))
        world_up = np.array([0.0, 1.0, 0.0])
        if mode == 2:
            cam_offset = car_up_vec * 0.30 - car_dir * 0.18
            cam_pos = self.car.body.pos + cam_offset
            forward = car_dir / (np.linalg.norm(car_dir) + 1e-8)
            right = car_right / (np.linalg.norm(car_right) + 1e-8)
            up_vec = car_up_vec / (np.linalg.norm(car_up_vec) + 1e-8)
        else:
            cam_dist = 8 if mode == 0 else 4
            cam_hgt = 2 if mode == 0 else 1.2
            cam_pos = self.car.body.pos - car_dir * cam_dist + np.array([0.0, cam_hgt, 0.0])
            forward = self.car.body.pos - cam_pos
            forward /= np.linalg.norm(forward) + 1e-8
            right = np.cross(world_up, forward)
            if np.linalg.norm(right) < 1e-6:
                right = np.array([1.0, 0.0, 0.0])
            else:
                right /= np.linalg.norm(right)
            up_vec = np.cross(forward, right)
            up_vec /= np.linalg.norm(up_vec) + 1e-8
        return cam_pos, forward, right, up_vec

    def _free_camera_axes(self):
        cos_pitch = math.cos(self.free_cam_pitch)
        view_dir = np.array(
            [
                math.sin(self.free_cam_yaw) * cos_pitch,
                math.sin(self.free_cam_pitch),
                math.cos(self.free_cam_yaw) * cos_pitch,
            ],
            dtype=float,
        )
        norm = np.linalg.norm(view_dir)
        if norm < 1e-6:
            view_dir = np.array([0.0, 0.0, 1.0], dtype=float)
            norm = 1.0
        view_dir /= norm
        forward = view_dir
        world_up = np.array([0.0, 1.0, 0.0], dtype=float)
        right = np.cross(world_up, forward)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            right /= right_norm
        up_vec = np.cross(forward, right)
        up_norm = np.linalg.norm(up_vec)
        if up_norm < 1e-6:
            up_vec = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            up_vec /= up_norm
        return forward, right, up_vec

    def _enter_free_camera(self):
        if self.car is None:
            return
        car_dir = self.car.body.rot.rotate(np.array([0, 0, 1]))
        car_up_vec = self.car.body.rot.rotate(np.array([0, 1, 0]))
        cam_pos, forward, _, _ = self._compute_follow_camera_pose(
            getattr(self, "_last_car_camera_mode", 0), car_dir, car_up_vec
        )
        self.free_cam_pos = np.array(cam_pos, dtype=float)
        forward_norm = forward / (np.linalg.norm(forward) + 1e-8)
        self.free_cam_yaw = math.atan2(forward_norm[0], forward_norm[2])
        self.free_cam_pitch = math.asin(float(np.clip(forward_norm[1], -0.999, 0.999)))
        self._free_camera_active = True
        if self.mode == "eval":
            try:
                import pygame

                if pygame.get_init():
                    pygame.mouse.set_visible(False)
                    pygame.event.set_grab(True)
                    pygame.mouse.get_rel()
            except Exception:
                pass

    def _exit_free_camera(self):
        if not self._free_camera_active:
            return
        self._free_camera_active = False
        if self.mode == "eval":
            try:
                import pygame

                if pygame.get_init():
                    pygame.event.set_grab(False)
                    pygame.mouse.set_visible(True)
                    pygame.mouse.get_rel()
            except Exception:
                pass

    def _set_camera_mode(self, new_mode: int):
        new_mode = int(np.clip(new_mode, 0, 3))
        prev = self.camera_mode
        if prev == new_mode:
            return
        if prev == 3:
            self._exit_free_camera()
        if new_mode == 3:
            if prev != 3:
                self._last_car_camera_mode = prev
            self.camera_mode = new_mode
            self._enter_free_camera()
        else:
            self.camera_mode = new_mode
            self._last_car_camera_mode = new_mode

    def _cycle_camera_mode(self):
        self._set_camera_mode((self.camera_mode + 1) % 4)

    def _toggle_free_camera(self):
        if self.camera_mode == 3:
            self._set_camera_mode(getattr(self, "_last_car_camera_mode", 0))
        else:
            self._set_camera_mode(3)

    def _update_free_camera(self, pygame_module):
        if not self._free_camera_active:
            return
        mx, my = pygame_module.mouse.get_rel()
        sensitivity = 0.0025
        self.free_cam_yaw += mx * sensitivity
        self.free_cam_pitch -= my * sensitivity
        limit = math.radians(89.0)
        self.free_cam_pitch = float(np.clip(self.free_cam_pitch, -limit, limit))

        move = np.zeros(3, dtype=float)
        forward, right, _ = self._free_camera_axes()
        view_forward = forward
        view_right = right
        up = np.array([0.0, 1.0, 0.0], dtype=float)
        keys = pygame_module.key.get_pressed()
        if keys[pygame_module.K_w]:
            move += view_forward
        if keys[pygame_module.K_s]:
            move -= view_forward
        if keys[pygame_module.K_a]:
            move -= view_right
        if keys[pygame_module.K_d]:
            move += view_right
        if keys[pygame_module.K_SPACE]:
            move += up
        if keys[pygame_module.K_LSHIFT] or keys[pygame_module.K_RSHIFT]:
            move -= up

        norm = np.linalg.norm(move)
        if norm > 1e-6:
            move /= norm
            self.free_cam_pos += move * self.free_cam_speed * self.dt

    def _release_road_layers(self) -> None:
        if not getattr(self, "road_layers", None):
            return
        for vao, vbo, _ in self.road_layers.values():
            try:
                vbo.release()
            except Exception:
                pass
            try:
                vao.release()
            except Exception:
                pass
        self.road_layers = {}

    def _build_road_layers(self) -> None:
        if not getattr(self, "render_ctx", None):
            return
        self._release_road_layers()
        if self.rp is None or not self.plan:
            return

        road_layers = build_road_vertices(self.terrain, self.rp, **self.plan)

        def _store_layer(name: str, verts: np.ndarray, noise: float = 0.0):
            if verts.size == 0:
                return
            vbo = self.render_ctx.ctx.buffer(verts.tobytes())
            vao = self.render_ctx.ctx.vertex_array(
                self.render_ctx.prog, vbo, "in_vert", "in_color"
            )
            self.road_layers[name] = (vao, vbo, noise)

        _store_layer("skirt", road_layers.get("skirt", np.zeros(0, dtype="f4")), 0.0)
        _store_layer(
            "deck",
            road_layers.get("deck", np.zeros(0, dtype="f4")),
            self.render_ctx.road_noise,
        )
        _store_layer("lines", road_layers.get("lines", np.zeros(0, dtype="f4")), 0.0)
        _store_layer("driveline", road_layers.get("driveline", np.zeros(0, dtype="f4")), 0.0)

    def init_renderer(self) -> None:
        """Initialise pygame and GPU resources if ``mode`` is ``eval``.

        The method is intentionally separate from :meth:`__init__` so that a
        caller can display loading progress before the OpenGL context is
        created.  It can safely be called multiple times (e.g. after
        :meth:`reset`).
        """

        if self.mode != "eval":
            return

        # Local imports to avoid pygame dependency in training mode
        import pygame
        import moderngl
        from .render import RenderContext
        from .terrain import build_terrain_triangles
        from .bbmodel import load_bbmodel
        from .signs.build import generate_speed_limit_sign

        self._set_status(0.8, "Building meshes...")

        if not pygame.display.get_init():
            pygame.display.init()
        if pygame.display.get_surface():
            self.width, self.height = pygame.display.get_surface().get_size()
        else:
            info = pygame.display.Info()
            self.width, self.height = info.current_w, info.current_h
        global WIDTH, HEIGHT
        WIDTH, HEIGHT = self.width, self.height

        pygame.display.set_mode(
            (self.width, self.height), pygame.OPENGL | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("Spinout")
        self.clock = pygame.time.Clock()
        self.render_ctx = RenderContext(self.width, self.height)
        self.render_ctx.setup_weather(self.weather, self.terrain_type, self.road_type)
        if self.car is not None:
            self.car.show_wind_vectors = self.wind_vectors_enabled

        # Terrain
        tb, _ = build_terrain_triangles(self.terrain)
        self.t_vertices = tb
        self.t_vbo = self.render_ctx.ctx.buffer(tb.tobytes())
        self.t_vao = self.render_ctx.ctx.vertex_array(
            self.render_ctx.prog, self.t_vbo, "in_vert", "in_color"
        )

        # Roads (if any)
        self._build_road_layers()
        if self.rp is not None and self.plan:
            posts, billboards = build_speed_sign_vertices(
                self.terrain,
                self.rp,
                self.plan["lane_width"],
                self.plan["lanes"],
                self.plan["shoulder"],
                self.plan.get("speed_limits"),
            )
            if len(posts):
                self.sign_post_vbo = self.render_ctx.ctx.buffer(posts.tobytes())
                self.sign_post_vao = self.render_ctx.ctx.vertex_array(
                    self.render_ctx.prog, self.sign_post_vbo, "in_vert", "in_color"
                )
            else:
                self.sign_post_vao = None
            self.sign_billboards = []
            for bb in billboards:
                img = generate_speed_limit_sign(bb["speed"])
                tex = self.render_ctx.ctx.texture(img.size, 4, img.tobytes())
                tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
                vbo = self.render_ctx.ctx.buffer(bb["verts"].tobytes())
                vao = self.render_ctx.ctx.vertex_array(
                    self.render_ctx.prog_tex, vbo, "in_vert", "in_tex"
                )
                self.sign_billboards.append((vao, tex))
        else:
            self.road_layers = {}
            self.sign_post_vao = None
            self.sign_billboards = []

        building_data = getattr(self, "buildings", None)
        if building_data and building_data.get("vertices") is not None:
            verts = building_data["vertices"]
            if isinstance(verts, np.ndarray) and len(verts):
                self.building_vbo = self.render_ctx.ctx.buffer(verts.tobytes())
                self.building_vao = self.render_ctx.ctx.vertex_array(
                    self.render_ctx.prog_lit,
                    [(self.building_vbo, "3f 3f 4f", "in_vert", "in_normal", "in_color")],
                )
                self.building_noise = float(building_data.get("noise_scale", 0.0))
            else:
                self.building_vao = None
                self.building_noise = 0.0
        else:
            self.building_vao = None
            self.building_noise = 0.0

        self.wheel_spin = [0.0] * 4
        self.font_small = pygame.font.SysFont(None, 24)
        self.font_big = pygame.font.SysFont(None, 48)
        self.hud_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        # Car model for solid rendering
        car_model = load_bbmodel(DATA_DIR / "car.bbmodel")
        tex = self.render_ctx.ctx.texture(
            car_model["texture_size"], 4, car_model["texture_bytes"]
        )
        tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.render_ctx.car_model_tex = tex
        self.car_model_data = car_model

        self._set_status(1.0, "Starting engines...")

    def _render(self, dt: float) -> None:
        import pygame
        import numpy as np
        from .car import collect_car_vertices
        from .bbmodel import collect_car_model_vertices
        from .hud import render_hud
        from .utils import compute_mvp

        car_dir = self.car.body.rot.rotate(np.array([0, 0, 1]))
        car_up_vec = self.car.body.rot.rotate(np.array([0, 1, 0]))

        if self.camera_mode == 3 and self._free_camera_active:
            forward, right, up_vec = self._free_camera_axes()
            cam_pos = self.free_cam_pos
        else:
            effective_mode = self.camera_mode if self.camera_mode != 3 else getattr(
                self, "_last_car_camera_mode", 0
            )
            cam_pos, forward, right, up_vec = self._compute_follow_camera_pose(
                effective_mode, car_dir, car_up_vec
            )
            if self.camera_mode == 3 and not self._free_camera_active:
                self.free_cam_pos = np.array(cam_pos, dtype=float)

        mvp = compute_mvp(self.width, self.height, cam_pos, right, forward, up_vec)
        self.render_ctx.set_camera_pose(cam_pos, forward, right, up_vec)
        self.render_ctx.set_mode(self.render_mode)
        self.render_ctx.clear()

        self.render_ctx.render_terrain(self.t_vao, mvp, self.render_ctx.road_noise)
        for name in ("skirt", "deck", "lines", "driveline"):
            layer = self.road_layers.get(name)
            if not layer:
                continue
            vao, _vbo, noise = layer
            self.render_ctx.render_terrain(vao, mvp, noise)
        skid_vertices = self.skidmarks.get_vertices()
        if skid_vertices.size:
            self.render_ctx.render_skid_marks(skid_vertices, mvp)
        if self.sign_post_vao is not None:
            self.render_ctx.render_signs(self.sign_post_vao, mvp)
        for vao, tex in self.sign_billboards:
            self.render_ctx.render_billboard(vao, tex, mvp)
        if getattr(self, "building_vao", None) is not None:
            self.render_ctx.render_lit_mesh(self.building_vao, mvp, noise_scale=getattr(self, "building_noise", 0.0))

        car_lines = collect_car_vertices(self.car, car_up_vec, car_dir, dt, self.wheel_spin)
        if getattr(self, "use_bbmodel", False):
            model_verts = collect_car_model_vertices(self.car, self.car_model_data)
            self.render_ctx.render_car_model(model_verts, mvp)
        else:
            self.render_ctx.render_car(car_lines, mvp)
        self.render_ctx.render_weather(mvp, dt)

        self.hud_surf.fill((0, 0, 0, 0))
        speed_mph = np.linalg.norm(self.car.body.vel) * 2.23694
        fps_r = self.clock.get_fps()
        fps_p = fps_r * self.substeps
        steer_angle = next(w.steer_angle for w in self.car.wheels if w.is_front)
        wind_speed = 0.0
        wind_dir = 0.0
        wind_label = "Calm"
        if self.wind_sample is not None:
            wind_speed = self.wind_sample.speed_mph
            wind_dir = self.wind_sample.direction_deg
            if wind_speed > 0.05:
                wind_label = self.wind_sample.compass_label
        render_hud(
            self.hud_surf,
            self.font_small,
            self.font_big,
            speed_mph,
            fps_r,
            fps_p,
            steer_angle,
            car_info=self.car_info,
            rpm=getattr(self.car, "engine_rpm", None),
            gear=getattr(self.car, "current_gear", None),
            surface_info=self.surface_info,
            render_mode=self.render_mode,
            camera_mode=self.camera_mode,
            wind_speed_mph=wind_speed,
            wind_direction_deg=wind_dir,
            wind_label=wind_label,
            wind_vectors_enabled=self.wind_vectors_enabled,
            controller_name=self._controller_hud[0],
            steer_label=self._controller_hud[1],
        )
        self.render_ctx.render_hud(self.hud_surf)
        pygame.display.flip()

    # ------------------------------------------------------------------
    # API methods

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset the environment and return the initial observation."""

        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if options is not None:
            self.cfg.update(options)
        self._build_world()
        self._refresh_initial_telemetry()
        self._controller_timer = 0.0
        self._controller_last_command = None
        if self._controller is not None:
            self._update_controller_period()
            self._controller.reset()
        self._controller_enabled_last_step = bool(self._controller and self._controller.enabled)
        self._controller_hud = self._resolve_hud_labels()
        return self._get_obs()

    # ------------------------------------------------------------------
    def _get_obs(self) -> Dict:
        """Assemble the observation dict for the current state."""

        return self._telemetry.as_observation()

    # ------------------------------------------------------------------
    def step(self, command: DriverCommand | Dict[str, float] | None = None, events=None):
        """Advance the simulation by one step.

        Parameters
        ----------
        command:
            ``DriverCommand`` instance or legacy dictionary with ``steer``,
            ``accel`` and ``brake`` entries.  When ``None`` the command defaults
            to zero.
        events:
            Optional iterable of pygame events to process for keybinds when in
            evaluation mode.  If ``None`` the function will poll ``pygame``
            directly.
        """

        if self.mode == "eval":
            import pygame

            if events is None:
                events = pygame.event.get()
            for e in events:
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_F1:
                        self.render_mode = 0
                    elif e.key == pygame.K_F2:
                        self.render_mode = 1
                    elif e.key == pygame.K_ESCAPE:
                        if self.camera_mode == 3:
                            self._set_camera_mode(getattr(self, "_last_car_camera_mode", 0))
                    elif e.key == pygame.K_c:
                        self._cycle_camera_mode()
                    elif e.key == pygame.K_f:
                        self._toggle_free_camera()
                    elif e.key == pygame.K_b:
                        self.use_bbmodel = not getattr(self, "use_bbmodel", False)
                    elif e.key == pygame.K_v:
                        self.render_ctx.wetness = (
                            0.0 if self.render_ctx.wetness > 0.0 else 1.0
                        )
                    elif e.key == pygame.K_t:
                        self._cycle_surface()
                    elif e.key == pygame.K_w:
                        self.wind_vectors_enabled = not self.wind_vectors_enabled
                        if self.car is not None:
                            self.car.show_wind_vectors = self.wind_vectors_enabled
                    elif e.key == pygame.K_r:
                        obs = self.reset()
                        self.init_renderer()
                        return obs, 0.0, False, False, {"reset": True}
                    elif pygame.K_1 <= e.key <= pygame.K_6:
                        self.switch_car(e.key - pygame.K_1)

            if self.camera_mode == 3:
                self._update_free_camera(pygame)

        if self.car is None:
            raise RuntimeError("Environment must be reset before stepping.")

        manual_cmd = (
            command
            if isinstance(command, DriverCommand)
            else DriverCommand.from_action(command)
        ).clipped()

        controller_preview = None
        controller_enabled = self._controller is not None and self._controller.enabled
        if controller_enabled:
            controller_preview = self._controller.preview_rate_hz
            if not self._controller_enabled_last_step:
                self._controller_last_command = None
                self._controller_timer = 0.0
        telemetry_before = self._build_snapshot(
            self._prev_velocity,
            preview_hz=controller_preview,
        )

        applied_cmd = manual_cmd
        if controller_enabled:
            period = self._controller_period or self.dt
            self._controller_timer += self.dt
            needs_update = (
                self._controller_last_command is None
                or self._controller_timer >= max(period - 1e-9, 0.0)
            )
            if needs_update:
                applied_cmd = (
                    self._controller
                    .step(telemetry_before, manual_cmd)
                    .clipped()
                )
                self._controller_last_command = applied_cmd
                self._controller_timer = 0.0
            else:
                applied_cmd = self._controller_last_command or manual_cmd
        else:
            self._controller_last_command = None
            self._controller_timer = 0.0

        self._last_driver_command = applied_cmd
        self._controller_hud = self._resolve_hud_labels()
        self._controller_enabled_last_step = controller_enabled

        self.car.steer = float(applied_cmd.steer)
        self.car.accel = float(applied_cmd.throttle)
        self.car.brake = float(applied_cmd.brake)

        if self.wind_system is not None:
            self.wind_sample = self.wind_system.update(self.dt)
            if self.car is not None:
                self.car.set_wind(self.wind_sample.vector)
        else:
            if self.car is not None:
                self.car.set_wind(None)
        if getattr(self, "render_ctx", None) and self.car is not None:
            self.car.show_wind_vectors = self.wind_vectors_enabled

        sub_dt = self.dt / self.substeps
        velocity_before = self.car.body.vel.copy()
        for _ in range(self.substeps):
            self.car.update(sub_dt)
            events = getattr(self.car, "slip_events", [])
            self.skidmarks.step(sub_dt, events)

        self.step_count += 1
        self.time += self.dt

        self._telemetry = self._build_snapshot(velocity_before)
        # ``_prev_velocity`` should always contain the velocity from the start
        # of the last physics step so that the next controller tick observes
        # the most recent acceleration.  This ensures that lateral acceleration
        # reported to controllers reflects the change in velocity that occurred
        # during the previous update rather than the (already updated) current
        # velocity, which would otherwise appear as zero acceleration.
        self._prev_velocity = velocity_before
        obs = self._get_obs()

        # Cost-first logic: each step incurs a time cost.  Reward is negative
        # cost so RL algorithms can still optimise it in the usual manner.
        step_cost = self.dt
        self.episode_cost += step_cost

        terminated = False
        self.termination_reason = None
        pos = self.car.body.pos
        if (
            pos[0] < 0
            or pos[0] > self.terrain.width
            or pos[2] < 0
            or pos[2] > self.terrain.height
        ):
            terminated = True
            self.termination_reason = "off_terrain"
        elif getattr(self.car, "is_upside_down", False):
            terminated = True
            self.termination_reason = "crash"

        truncated = False
        trunc_reason = None
        if self.max_steps is not None and self.step_count >= self.max_steps:
            truncated = True
            trunc_reason = "max_steps"
        elif self.time_limit is not None and self.time >= self.time_limit:
            truncated = True
            trunc_reason = "time_limit"
        elif self.cost_limit is not None and self.episode_cost >= self.cost_limit:
            truncated = True
            trunc_reason = "cost_limit"

        info = {
            "step_cost": step_cost,
            "episode_cost": self.episode_cost,
            "plan": self.plan,
            "driver_command": self._last_driver_command.as_dict(),
            "telemetry": self._telemetry.as_observation(),
        }
        if self._controller is not None and self._controller.enabled:
            info["controller"] = self._controller.name
        if terminated:
            info["reason"] = self.termination_reason
        elif truncated:
            info["reason"] = trunc_reason

        reward = -step_cost
        if self.mode == "eval" and getattr(self, "render_ctx", None):
            self._render(self.dt)
            self.clock.tick(60)
        return obs, reward, terminated, truncated, info
