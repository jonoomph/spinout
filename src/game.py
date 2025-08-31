# game.py
"""Minimal entry point demonstrating the :class:`Environment` API.

This script mirrors the behaviour of the old ``game.py`` but delegates the
heavy lifting to :class:`environment.Environment`.  When run it opens a pygame
window and allows manual control of the car using the existing ``get_controls``
helper.  The script is intentionally small – all simulation details live in
``environment.py``.
"""

import pygame

from src import splash
from src.sim.controls import (
    get_controls,
    STEER_MAX,
    ACCEL_MAX,
    BRAKE_MAX,
)


def main() -> None:
    pygame.init()
    display_info = pygame.display.Info()
    width, height = display_info.current_w, display_info.current_h
    screen = pygame.display.set_mode((width, height), pygame.DOUBLEBUF)
    pygame.display.set_caption("SPINOUT")

    # Show the animated splash screen while the environment initialises.
    env, obs = splash.run(screen)

    done = False
    info: dict = {}
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

