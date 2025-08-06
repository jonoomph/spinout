"""Input handling for keyboard and controller.

This module provides PS4/5 controller support via ``pygame.joystick``.  Steering
and pedal inputs are quantised to discrete integer steps which are then
normalised by the game loop before being applied to the physics layer.
"""

import pygame
from pygame.locals import *

STEER_MAX = 128
ACCEL_MAX = 32
BRAKE_MAX = 32

# Cache joystick state so we don't repeatedly create/initialise it every frame
_steer_idx = 0
_joystick = None
_is_wheel = False


def _update_steer(keys):
    """Return the discrete steering index for keyboard control."""
    global _steer_idx
    left = keys[K_LEFT]
    right = keys[K_RIGHT]
    if left and not right:
        _steer_idx = min(_steer_idx + 1, STEER_MAX)
    elif right and not left:
        _steer_idx = max(_steer_idx - 1, -STEER_MAX)
    else:
        if _steer_idx > 0:
            _steer_idx -= 1
        elif _steer_idx < 0:
            _steer_idx += 1
    return _steer_idx


def _quantize(value, maximum):
    """Clamp and quantise ``value`` in ``[-1,1]`` or ``[0,1]`` to ``maximum`` steps."""
    return max(-maximum, min(maximum, int(round(value * maximum))))


def _trigger_value(raw):
    """Convert raw trigger axis reading to ``[0,1]``.

    Some controllers report triggers in ``[-1,1]`` with ``-1`` at rest while
    others use ``[0,1]``.  This helper normalises both conventions.  A small
    deadzone is also applied to filter noise from inactive axes.
    """

    if raw < 0:
        raw = (raw + 1) / 2
    value = max(0.0, min(1.0, raw))
    return value if value > 0.05 else 0.0


def _read_trigger(joystick, axes):
    """Return the best reading from ``axes`` on ``joystick``.

    Different systems expose PS4/5 triggers on different axes.  We sample a
    set of candidates and return the largest value after normalisation.
    """

    values = [_trigger_value(joystick.get_axis(a)) for a in axes]
    return max(values) if values else 0.0


def _init_joystick():
    """Lazily initialise and cache the first available joystick."""
    global _joystick, _is_wheel
    if _joystick is not None:
        return
    # Joystick module may need explicit initialisation
    if not pygame.joystick.get_init():
        pygame.joystick.init()
    if pygame.joystick.get_count() > 0:
        js = pygame.joystick.Joystick(0)
        js.init()
        name = js.get_name().lower()
        wheel_names = ("wheel", "logitech", "thrustmaster", "g29", "g920")
        _joystick = js
        _is_wheel = any(n in name for n in wheel_names)


def get_controls(keys):
    """Return the current control state as discrete step values.

    Handles Logitech-style wheels, PS4/5 controllers (with correct axis and sign), and keyboard input.
    Returns (steer, accel, brake, car_index)
    """
    _init_joystick()

    if _joystick is not None:
        pygame.event.pump()  # update joystick state

        if _is_wheel:
            # Logitech-style wheel
            steer_axis = -_joystick.get_axis(0)  # Flip axis!
            accel_axis = _joystick.get_axis(2)
            brake_axis = _joystick.get_axis(3)
            steer_val = _quantize(steer_axis, STEER_MAX)
            accel_val = _quantize((1 - accel_axis) / 2, ACCEL_MAX)
            brake_val = _quantize((1 - brake_axis) / 2, BRAKE_MAX)
        else:
            # PS4/5 controller: axis 0 (steer), axis 2 (brake), axis 5 (accel)
            steer_axis = -_joystick.get_axis(0)        # Flip for correct steering!
            brake_axis = _joystick.get_axis(2)         # L2: -1 (rest), +1 (pressed)
            accel_axis = _joystick.get_axis(5)         # R2: -1 (rest), +1 (pressed)

            # Debug print (optional)
            # print(f"PS4/5: steer_axis={steer_axis:.2f}, brake_axis={brake_axis:.2f}, accel_axis={accel_axis:.2f}")

            steer_val = _quantize(steer_axis, STEER_MAX)
            accel_val = _quantize((accel_axis + 1) / 2, ACCEL_MAX)
            brake_val = _quantize((brake_axis + 1) / 2, BRAKE_MAX)

    else:
        # Keyboard fallback
        steer_val = _update_steer(keys)
        accel_val = 0
        brake_val = 0
        if keys[K_UP] and not keys[K_DOWN]:
            accel_val = ACCEL_MAX
        elif keys[K_DOWN] and not keys[K_UP]:
            brake_val = BRAKE_MAX

    car_index = None
    for i in range(1, 10):
        if keys[K_1 + i - 1]:
            car_index = i - 1
            break

    return steer_val, accel_val, brake_val, car_index
