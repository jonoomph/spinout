import threading
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import pygame

from sim.environment import Environment

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

    duration: float = 2.2  # total length of the animation (s)
    accel_time: float = 0.8  # how long to accelerate before braking (s)
    accel: float = 12.0  # forward acceleration in m/s^2
    brake: float = 15.0  # braking deceleration in m/s^2
    start_offset_px: Optional[float] = None  # how far left the text begins (px)


def _shear_surface(surf: pygame.Surface, shear: float) -> Tuple[pygame.Surface, int]:
    """Return a copy of ``surf`` sheared horizontally.

    ``shear`` is the horizontal offset per vertical pixel (i.e. tangent of the
    shear angle). Positive values lean the top of the surface to the right while
    keeping the bottom fixed, negative values lean to the left.
    """

    w, h = surf.get_size()
    offset = int(abs(shear) * h)
    sheared = pygame.Surface((w + offset, h), pygame.SRCALPHA)
    if shear >= 0:
        for y in range(h):
            line = surf.subsurface((0, y, w, 1))
            dx = int(shear * (h - y))
            sheared.blit(line, (dx, y))
    else:
        for y in range(h):
            line = surf.subsurface((0, y, w, 1))
            dx = int(shear * (h - y))
            sheared.blit(line, (dx + offset, y))
    return sheared, offset


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
    fills on the main thread.  Once the environment is ready a simple
    kinematic model drives the ``SPINOUT`` logo into frame before braking to a
    stop.  After the animation an overlay is displayed while the renderer
    initialises, then the ready ``Environment`` and its initial observation are
    returned.
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

    splash_duration = config.duration
    sim_time = 0.0
    accel_phase = config.accel_time
    anim_started = False
    progress_start = 0.0
    pos = 0.0  # metres
    vel = 0.0  # m/s

    # Pre-compute scaling so final position lands at screen centre
    t1 = accel_phase
    t2 = max(0.001, splash_duration - accel_phase)
    v1 = config.accel * t1
    disp1 = 0.5 * config.accel * t1 * t1
    disp2 = v1 * t2 - 0.5 * config.brake * t2 * t2
    total_disp = disp1 + disp2
    if total_disp <= 0:
        total_disp = 1.0
    distance_px = width / 2 + start_offset
    px_per_m = distance_px / total_disp

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
            if pos >= total_disp and vel <= 0.0:
                acc = 0.0
                vel = 0.0
                pos = total_disp
            else:
                acc = config.accel if sim_time < accel_phase else -config.brake
                vel += acc * dt
                pos += vel * dt
                if pos >= total_disp:
                    pos = total_disp
                    vel = 0.0
            g_force = acc / 9.81
            shear = max(-0.5, min(0.5, -g_force * 0.3))
            screen_x = -start_offset + pos * px_per_m
            sim_time += dt

            skewed, offset = _shear_surface(text_surf, shear)
            r = skewed.get_rect()
            r.centerx = int(screen_x + (offset if shear > 0 else -offset) / 2)
            r.centery = text_rect.centery
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
