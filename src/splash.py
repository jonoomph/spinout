import threading
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import math
import threading

import numpy as np
import pygame

from src.sim.environment import Environment

# Visual constants
BG_COLOR = (0, 0, 0)
BAR_COLOR = (80, 180, 255)
BAR_BG = (40, 40, 40)
TEXT_COLOR = (255, 255, 255)


@dataclass
class SplashConfig:
    """User-tunable parameters controlling the splash animation.

    Adjust these to experiment with different starting conditions without
    digging through the logic below.
    """

    duration: float = 3.2  # total length of the animation (s)
    accel_time: float = 1.2  # how long to accelerate before braking (s)
    brake_time: float = 1.0  # braking window (s)
    coast_time: float = 0.25  # pause between throttle and braking (s)
    settle_time: float = 0.8  # additional time to let suspension settle (s)
    throttle: float = 1.0  # throttle input during the acceleration phase
    brake_strength: float = 0.95  # brake input during the braking phase
    sim_dt: float = 1 / 240  # physics step used for the logo simulation
    start_offset_px: Optional[float] = None  # how far left the text begins (px)


@dataclass
class LogoState:
    """Captured state of the simulated car projected onto the logo."""

    time: float
    distance: float
    velocity: float
    pitch: float
    heave: float


class LogoMotion:
    """Pre-simulate the car motion that drives the splash logo."""

    def __init__(self, config: SplashConfig):
        self.dt = float(config.sim_dt)
        self.accel_time = float(config.accel_time)
        self.brake_time = float(config.brake_time)
        self.coast_time = float(config.coast_time)
        self.settle_time = float(config.settle_time)
        self.throttle = float(config.throttle)
        self.brake_strength = float(config.brake_strength)

        env = Environment(
            cfg={"flat": True, "dt": self.dt, "substeps": 5, "seed": 0},
            mode="train",
        )
        env.reset()

        car = env.car
        assert car is not None  # appease static checkers
        self._start_pos = car.body.pos.copy()
        base_compression = np.array([w.compression for w in car.wheels], dtype=float)

        self.states = [self._capture_state(car, 0.0, base_compression)]
        total_time = (
            self.accel_time
            + self.coast_time
            + self.brake_time
            + self.settle_time
        )
        steps = int(math.ceil(total_time / self.dt))
        time = 0.0
        for _ in range(steps):
            time += self.dt
            if time <= self.accel_time:
                car.accel = self.throttle
                car.brake = 0.0
            elif time <= self.accel_time + self.coast_time:
                car.accel = 0.0
                car.brake = 0.0
            elif time <= self.accel_time + self.coast_time + self.brake_time:
                car.accel = 0.0
                car.brake = self.brake_strength
            else:
                car.accel = 0.0
                car.brake = 0.0
            car.steer = 0.0
            car.update(self.dt)
            self.states.append(self._capture_state(car, time, base_compression))

        extra_steps = 0
        while extra_steps < 600:
            last = self.states[-1]
            if last.velocity < 0.05 and abs(last.heave) < 0.0005:
                break
            time += self.dt
            car.accel = 0.0
            car.brake = 0.0
            car.steer = 0.0
            car.update(self.dt)
            self.states.append(self._capture_state(car, time, base_compression))
            extra_steps += 1

        self.duration = self.states[-1].time
        self.total_distance = max(state.distance for state in self.states)
        if self.total_distance <= 0:
            self.total_distance = 1.0
        self.max_velocity = max(state.velocity for state in self.states)
        if self.max_velocity <= 0:
            self.max_velocity = 1.0
        self._index = 0

    def _capture_state(
        self,
        car,
        time: float,
        base_compression: np.ndarray,
    ) -> LogoState:
        pos = car.body.pos
        distance = float(pos[2] - self._start_pos[2])
        velocity = float(np.linalg.norm(car.body.vel))
        forward = car.body.rot.rotate(np.array([0.0, 0.0, 1.0]))
        pitch = float(math.atan2(forward[1], forward[2]))
        compressions = np.array([w.compression for w in car.wheels], dtype=float)
        heave = float(np.mean(compressions - base_compression))
        return LogoState(time=time, distance=distance, velocity=velocity, pitch=pitch, heave=heave)

    def sample(self, t: float) -> LogoState:
        t = max(0.0, min(t, self.duration))
        while self._index < len(self.states) - 2 and self.states[self._index + 1].time <= t:
            self._index += 1
        s0 = self.states[self._index]
        s1 = self.states[self._index + 1]
        if t <= s0.time or s0.time == s1.time:
            return s0
        alpha = (t - s0.time) / (s1.time - s0.time)
        return LogoState(
            time=t,
            distance=s0.distance + (s1.distance - s0.distance) * alpha,
            velocity=s0.velocity + (s1.velocity - s0.velocity) * alpha,
            pitch=s0.pitch + (s1.pitch - s0.pitch) * alpha,
            heave=s0.heave + (s1.heave - s0.heave) * alpha,
        )


def _shear_surface(surf: pygame.Surface, shear: float) -> Tuple[pygame.Surface, int]:
    """Return a copy of ``surf`` sheared horizontally, anchored at the base."""

    w, h = surf.get_size()
    if h <= 1:
        return surf.copy(), 0
    max_offset = abs(shear) * (h - 1)
    new_width = int(math.ceil(w + max_offset))
    sheared = pygame.Surface((new_width, h), pygame.SRCALPHA)
    base_offset = int(round(max_offset)) if shear < 0 else 0
    for y in range(h):
        line = surf.subsurface((0, y, w, 1))
        offset = shear * (h - 1 - y)
        x = base_offset + int(round(offset))
        sheared.blit(line, (x, y))
    bottom_left = base_offset
    return sheared, bottom_left


def _load_environment(progress: Dict[str, float]) -> Tuple[Environment, Dict]:
    """Create the environment while streaming progress updates."""

    def status_cb(p: float, _msg: str) -> None:
        progress["target"] = max(progress.get("target", 0.0), p)

    env = Environment(
        cfg={"dt": 1 / 120, "seed": 0},
        mode="eval",
        status_callback=status_cb,
    )
    obs = env.reset()
    progress["target"] = max(progress.get("target", 0.0), 1.0)
    progress["ready"] = True
    return env, obs


def run(screen: pygame.Surface, config: SplashConfig = SplashConfig()) -> Tuple[Environment, Dict]:
    """Render the animated splash screen on ``screen``.

    ``Environment.reset`` executes on a background thread while a progress bar
    fills on the main thread.  Once the environment is ready a short
    Spinout simulation drives the ``SPINOUT`` logo into frame before braking to
    a stop, leaning based on the simulated suspension.  After the animation an
    overlay is displayed while the renderer initialises, then the ready
    ``Environment`` and its initial observation are returned.
    """

    width, height = screen.get_size()
    bar_rect = pygame.Rect(int(width * 0.1), int(height * 0.85), int(width * 0.8), int(height * 0.035))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 160)
    text_surf = font.render("SPINOUT", True, TEXT_COLOR)
    text_rect = text_surf.get_rect()
    text_rect.centery = height // 2
    bar_font = pygame.font.SysFont(None, 36)

    start_offset = (
        config.start_offset_px
        if config.start_offset_px is not None
        else text_rect.width * 0.5
    )

    progress: Dict[str, float] = {
        "value": 0.0,
        "target": 0.0,
        "ready": False,
    }

    env_container: Dict[str, Environment] = {}
    obs_container: Dict[str, Dict] = {}
    error_container: Dict[str, BaseException] = {}

    def env_thread() -> None:
        try:
            env, obs = _load_environment(progress)
            env_container["env"] = env
            obs_container["obs"] = obs
        except BaseException as e:  # capture and re-raise later
            error_container["exc"] = e
            progress["target"] = 1.0
            progress["ready"] = True

    threading.Thread(target=env_thread, daemon=True).start()

    logo_motion = LogoMotion(config)
    splash_duration = max(config.duration, logo_motion.duration)
    sim_time = 0.0
    anim_started = False
    progress_start = 0.0

    distance_px = width / 2 + start_offset
    px_per_m = distance_px / logo_motion.total_distance
    baseline_y = text_rect.bottom
    vertical_scale = text_rect.height * 3.5

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if error_container.get("exc"):
            progress["value"] += (1.0 - progress["value"]) * min(8.0 * dt, 1.0)
            if progress["value"] >= 0.999:
                running = False
        elif not progress["ready"]:
            progress["value"] += (progress["target"] - progress["value"]) * min(
                8.0 * dt, 1.0
            )
        else:
            if not anim_started:
                anim_started = True
                progress_start = progress["value"]
            else:
                sim_time = min(sim_time + dt, splash_duration)
                progress["value"] = min(
                    progress_start + (1.0 - progress_start) * (sim_time / splash_duration),
                    1.0,
                )

        screen.fill(BG_COLOR)

        # Draw progress bar
        pygame.draw.rect(screen, BAR_BG, bar_rect)
        fill_w = int(bar_rect.width * progress["value"])
        if fill_w > 0:
            fill = pygame.Rect(bar_rect.left, bar_rect.top, fill_w, bar_rect.height)
            pygame.draw.rect(screen, BAR_COLOR, fill)
        pygame.draw.rect(screen, TEXT_COLOR, bar_rect, 2)
        progress_txt = bar_font.render(
            f"Loading {int(progress['value'] * 100)}%", True, TEXT_COLOR
        )
        txt_rect = progress_txt.get_rect(midbottom=(bar_rect.centerx, bar_rect.top - 5))
        screen.blit(progress_txt, txt_rect)

        # Update animation once loading is finished
        if anim_started:
            state = logo_motion.sample(sim_time)
            screen_x = -start_offset + state.distance * px_per_m
            distance_ratio = max(0.0, min(state.distance / logo_motion.total_distance, 1.0))
            distance_ease = 1.0 - (1.0 - distance_ratio) ** 3
            speed_ratio = max(0.0, min(state.velocity / logo_motion.max_velocity, 1.0))
            scale = 0.55 + 0.45 * distance_ease + 0.15 * speed_ratio
            scale = max(0.45, min(scale, 1.25))
            scaled_size = (
                max(1, int(round(text_rect.width * scale))),
                max(1, int(round(text_rect.height * scale))),
            )
            scaled = pygame.transform.smoothscale(text_surf, scaled_size)
            shear = max(-0.5, min(0.5, -state.pitch * 12.0))
            skewed, bottom_left = _shear_surface(scaled, shear)
            bottom_center = bottom_left + scaled_size[0] / 2
            baseline = baseline_y + state.heave * vertical_scale
            r = skewed.get_rect()
            r.left = int(round(screen_x - bottom_center))
            r.bottom = int(round(baseline))
            screen.blit(skewed, r)

            if sim_time >= splash_duration and progress["value"] >= 0.999:
                running = False

        pygame.display.flip()

    if error_container.get("exc"):
        raise error_container["exc"]

    # Display a short overlay while the renderer initialises to bridge the
    # potential freeze when the OpenGL context is created.
    overlay = pygame.Surface((width, height), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))
    msg = bar_font.render("Finalizing...", True, TEXT_COLOR)
    msg_rect = msg.get_rect(center=(width // 2, height // 2))
    overlay.blit(msg, msg_rect)
    screen.blit(overlay, (0, 0))
    pygame.display.flip()

    env = env_container["env"]
    env.init_renderer()

    return env, obs_container.get("obs", {})


if __name__ == "__main__":
    pygame.init()
    demo_screen = pygame.display.set_mode((1280, 720), pygame.DOUBLEBUF)
    run(demo_screen)
