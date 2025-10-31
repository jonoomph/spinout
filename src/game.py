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
from src.controllers.pid import PIDSteeringController
from src.sim.control_api import DriverCommand
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

    controller = PIDSteeringController()
    controller.attach(env)
    env.attach_controller(controller)
    auto_steer_enabled = False

    done = False
    info: dict = {}
    while not done:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    auto_steer_enabled = controller.toggle()
                    status = "enabled" if auto_steer_enabled else "disabled"
                    print(f"Auto-steer {status}")

        keys = pygame.key.get_pressed()
        steer, accel, brake, _ = get_controls(keys)
        command = DriverCommand(
            steer=steer / STEER_MAX,
            throttle=accel / ACCEL_MAX,
            brake=brake / BRAKE_MAX,
        )

        obs, reward, terminated, truncated, info = env.step(command, events)
        done = done or terminated or truncated

        if info.get("reset"):
            controller.attach(env)
            env.attach_controller(controller)
            if auto_steer_enabled:
                controller.reset()

    print("episode_cost:", info.get("episode_cost"))
    print("reason:", info.get("reason"))
    pygame.quit()


if __name__ == "__main__":
    main()

