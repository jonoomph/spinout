# game.py
"""Minimal entry point demonstrating the :class:`Environment` API.

This script mirrors the behaviour of the old ``game.py`` but delegates the
heavy lifting to :class:`environment.Environment`.  When run it opens a pygame
window and allows manual control of the car using the existing ``get_controls``
helper.  The script is intentionally small – all simulation details live in
``environment.py``.
"""

import time
import pygame

from src.sim.environment import Environment
from src.sim.controls import (
    get_controls,
    STEER_MAX,
    ACCEL_MAX,
    BRAKE_MAX,
)
def show_loading(progress, message, surface, loading_font, logo_font) -> None:
    """Draw a centered loading bar with static SPINOUT branding."""
    w, h = surface.get_size()
    surface.fill((0, 0, 0))

    # Render logo at top quarter of the screen
    logo_img = logo_font.render("SPINOUT", True, (255, 255, 255))
    logo_rect = logo_img.get_rect(center=(w // 2, h // 4))
    surface.blit(logo_img, logo_rect)

    bar_w, bar_h = int(w * 0.6), 40
    x, y = (w - bar_w) // 2, h // 2 + 20
    pygame.draw.rect(surface, (255, 255, 255), (x, y, bar_w, bar_h), 2)
    inner_w = int(bar_w * progress)
    pygame.draw.rect(surface, (255, 255, 255), (x, y, inner_w, bar_h))
    txt = loading_font.render(message, True, (255, 255, 255))
    surface.blit(txt, txt.get_rect(center=(w // 2, y - 30)))
    pygame.display.flip()


def main() -> None:
    pygame.init()
    info = pygame.display.Info()
    width, height = info.current_w, info.current_h
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Loading...")
    loading_font = pygame.font.SysFont(None, 48)
    logo_font = pygame.font.SysFont(None, 96)

    def status_cb(progress, message):
        show_loading(progress, message, screen, loading_font, logo_font)

    env = Environment({}, mode="eval", status_callback=status_cb)
    obs = env.reset()
    env.init_renderer()
    # Briefly show the completed loading bar
    time.sleep(0.5)

    done = False
    info = {}
    while not done:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        steer, accel, brake, _ = get_controls(keys)
        action = {
            "steer": steer / STEER_MAX,
            "accel": accel / ACCEL_MAX,
            "brake": brake / BRAKE_MAX,
        }

        obs, reward, terminated, truncated, info = env.step(action, events)
        done = done or terminated or truncated

    print("episode_cost:", info.get("episode_cost"))
    print("reason:", info.get("reason"))
    pygame.quit()


if __name__ == "__main__":
    main()

