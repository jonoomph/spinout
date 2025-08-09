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
from typing import Callable, Dict, Optional, Tuple
from itertools import product

import numpy as np

from .physics import Terrain, Car
from .roads.plan import generate_plan, get_safe_start_position_and_rot
from .roads.build import apply_plan
from .colors import (
    ROAD_ASPHALT_COLOR,
    ROAD_CONCRETE_COLOR,
    ROAD_GRAVEL_COLOR,
    TERRAIN_GRASS_COLOR,
    TERRAIN_SAND_COLOR,
    TERRAIN_SNOW_COLOR,
    TERRAIN_DIRT_COLOR,
)


# ---------------------------------------------------------------------------
# Constants shared with the original ``game.py`` script.  They are kept here so
# both the interactive game and the programmatic environment can use the same
# defaults without duplicating definitions.

WIDTH, HEIGHT = 1854, 1168

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

        # Rendering state ------------------------------------------------------
        self.render_mode = 1
        self.camera_mode = 0
        self.use_bbmodel = False
        self.surface_idx = 0
        self.surface_info = ""
        self.car_info = ""

        # Placeholders; the actual world is created on ``reset``
        self.terrain: Optional[Terrain] = None
        self.car: Optional[Car] = None
        self.rp = None

    # ------------------------------------------------------------------
    # World generation helpers

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

        self._set_status(0.2, "Generating terrain...")
        if self.cfg.get("flat"):
            # Deterministic flat terrain used for the physics tests
            self.weather = "dry"
            self.road_type = "asphalt"
            self.terrain_type = "asphalt"
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
        else:
            # Procedurally generate a random driving environment
            self.weather = self.rng.choice(["dry", "wet"], p=[0.7, 0.3])
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

        self.rp = rp

        if self.terrain_type in TERRAIN_TYPES:
            self.surface_idx = SURFACES.index((self.weather, self.terrain_type))
        else:
            self.surface_idx = 0
        self.surface_info = (
            f"{self.weather.title()} {self.road_type.title()} | {self.terrain_type.title()}"
        )

        self._set_status(0.7, "Loading car...")

        # Load car data and spawn the car --------------------------------------
        with open(DATA_DIR / "cars.json") as f:
            cars = json.load(f)

        car_index = int(self.cfg.get("car_index", 0))
        car_data = cars[car_index]
        self.car = Car(self.terrain, car_data)
        self.car_info = f"{car_data['make']} {car_data['model']} ({car_data['year']})"

        if self.cfg.get("flat"):
            start_x = self.terrain.width / 4
            start_z = self.terrain.height / 4
            rest_y = (
                self.terrain.get_height(start_x, start_z)
                + self.car.cg_height
            )
            self.car.body.pos = np.array([start_x, rest_y, start_z])
        else:
            pos, rot = get_safe_start_position_and_rot(self.terrain, rp, 15.0)
            self.car.body.pos, self.car.body.rot = pos, rot

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
        self.surface_info = (
            f"{weather.title()} {self.road_type.title()} | {terrain_type.title()}"
        )
        self.render_ctx.setup_weather(weather, terrain_type, self.road_type)

        self.terrain.terrain_type = terrain_type
        self.terrain.color = t["color"]
        friction = t["friction"] * weather_mod
        self.terrain.base_friction = friction
        self.terrain.surface_friction.fill(friction)
        road_mu = ROAD_TYPES[self.road_type]["friction"] * weather_mod
        self.terrain.road_friction[self.terrain.road_friction > 0] = road_mu
        col = np.array(t["color"], dtype="f4")
        self.t_vertices[:, 3:7] = col
        self.t_vbo.write(self.t_vertices.tobytes())

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
            self.car.body.pos = np.array([start_x, ground_h + self.car.cg_height, start_z])
            self.car.body.rot = rot
        else:
            pos, rot = get_safe_start_position_and_rot(self.terrain, self.rp, 15.0)
            self.car = Car(self.terrain, car_data)
            self.car.body.pos, self.car.body.rot = pos, rot
        self.car_info = f"{car_data['make']} {car_data['model']} ({car_data['year']})"
        self.cfg["car_index"] = car_index

        # Reset episode counters
        self.step_count = 0
        self.time = 0.0
        self.episode_cost = 0.0
        self.termination_reason = None

    # ------------------------------------------------------------------
    # Rendering helpers

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
        from .roads.build import build_road_vertices, build_speed_sign_vertices
        from .bbmodel import load_bbmodel
        from .signs.build import generate_speed_limit_sign

        self._set_status(0.8, "Building meshes...")

        pygame.display.set_mode((WIDTH, HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
        pygame.display.set_caption("Spinout")
        self.clock = pygame.time.Clock()
        self.render_ctx = RenderContext(WIDTH, HEIGHT)
        self.render_ctx.setup_weather(self.weather, self.terrain_type, self.road_type)

        # Terrain
        tb, _ = build_terrain_triangles(self.terrain)
        self.t_vertices = tb
        self.t_vbo = self.render_ctx.ctx.buffer(tb.tobytes())
        self.t_vao = self.render_ctx.ctx.vertex_array(
            self.render_ctx.prog, self.t_vbo, "in_vert", "in_color"
        )

        # Roads (if any)
        if self.rp is not None and self.plan:
            road_verts = build_road_vertices(self.terrain, self.rp, **self.plan)
            self.road_vbo = self.render_ctx.ctx.buffer(road_verts.tobytes())
            self.road_vao = self.render_ctx.ctx.vertex_array(
                self.render_ctx.prog, self.road_vbo, "in_vert", "in_color"
            )
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
            self.road_vao = None
            self.sign_post_vao = None
            self.sign_billboards = []

        self.wheel_spin = [0.0] * 4
        self.font_small = pygame.font.SysFont(None, 24)
        self.font_big = pygame.font.SysFont(None, 48)
        self.hud_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

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

        if self.camera_mode == 2:
            car_right = self.car.body.rot.rotate(np.array([1, 0, 0]))
            cam_offset = car_up_vec * 0.30 - car_dir * 0.18
            cam_pos = self.car.body.pos + cam_offset
            forward = -car_dir / np.linalg.norm(car_dir)
            right = car_right / np.linalg.norm(car_right)
            up_vec = car_up_vec / np.linalg.norm(car_up_vec)
        else:
            cam_dist = 8 if self.camera_mode == 0 else 4
            cam_hgt = 2 if self.camera_mode == 0 else 1.2
            cam_pos = self.car.body.pos - car_dir * cam_dist + np.array([0, cam_hgt, 0])
            forward = -(self.car.body.pos - cam_pos)
            forward /= np.linalg.norm(forward)
            right = np.cross(forward, np.array([0, 1, 0]))
            right /= np.linalg.norm(right)
            up_vec = np.cross(right, forward)
            up_vec /= np.linalg.norm(up_vec)

        mvp = compute_mvp(WIDTH, HEIGHT, cam_pos, right, forward, up_vec)
        self.render_ctx.set_camera(cam_pos)
        self.render_ctx.set_mode(self.render_mode)
        self.render_ctx.clear()

        if self.road_vao is not None:
            self.render_ctx.render_terrain(self.road_vao, mvp, self.render_ctx.road_noise)
        self.render_ctx.render_terrain(self.t_vao, mvp, self.render_ctx.road_noise)
        if self.sign_post_vao is not None:
            self.render_ctx.render_signs(self.sign_post_vao, mvp)
        for vao, tex in self.sign_billboards:
            self.render_ctx.render_billboard(vao, tex, mvp)

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
        return self._get_obs()

    # ------------------------------------------------------------------
    def _get_obs(self) -> Dict:
        """Assemble the observation dict for the current state."""

        speed = float(np.linalg.norm(self.car.body.vel))
        fwd = self.car.body.rot.rotate(np.array([0, 0, 1]))
        v_ego = float(np.dot(self.car.body.vel, fwd))
        roll_axis = self.car.body.rot.rotate(np.array([0, 1, 0]))
        roll = float(math.atan2(roll_axis[0], roll_axis[1]))

        obs = {
            "current": {
                "speed": speed,
                "lateral_acc": 0.0,  # Placeholder for future extension
                "vEgo": v_ego,
                "aEgo": 0.0,
                "roll": roll,
            },
            "target": {
                "speed": self.plan.get("target_speed", 0.0),
                "lateral_acc": 0.0,
                "vEgo": 0.0,
                "aEgo": 0.0,
                "roll": 0.0,
            },
        }
        return obs

    # ------------------------------------------------------------------
    def step(self, action: Dict[str, float], events=None):
        """Advance the simulation by one step.

        Parameters
        ----------
        action:
            Dictionary with ``steer``, ``accel`` and ``brake`` values in the
            range ``[-1,1]`` / ``[0,1]``.
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
                    elif e.key == pygame.K_c:
                        self.camera_mode = (self.camera_mode + 1) % 3
                    elif e.key == pygame.K_b:
                        self.use_bbmodel = not getattr(self, "use_bbmodel", False)
                    elif e.key == pygame.K_v:
                        self.render_ctx.wetness = (
                            0.0 if self.render_ctx.wetness > 0.0 else 1.0
                        )
                    elif e.key == pygame.K_t:
                        self._cycle_surface()
                    elif e.key == pygame.K_r:
                        obs = self.reset()
                        self.init_renderer()
                        return obs, 0.0, False, False, {"reset": True}
                    elif pygame.K_1 <= e.key <= pygame.K_6:
                        self.switch_car(e.key - pygame.K_1)

        self.car.steer = float(action.get("steer", 0.0))
        self.car.accel = float(action.get("accel", 0.0))
        self.car.brake = float(action.get("brake", 0.0))

        for _ in range(self.substeps):
            self.car.update(self.dt / self.substeps)

        self.step_count += 1
        self.time += self.dt

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
        }
        if terminated:
            info["reason"] = self.termination_reason
        elif truncated:
            info["reason"] = trunc_reason

        reward = -step_cost
        if self.mode == "eval" and getattr(self, "render_ctx", None):
            self._render(self.dt)
            self.clock.tick(60)
        return obs, reward, terminated, truncated, info
