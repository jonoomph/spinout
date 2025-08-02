# controls.py
from pygame.locals import *

STEER_MAX = 128
_steer_idx = 0


def _update_steer(keys):
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
    return _steer_idx / STEER_MAX


def get_controls(keys):
    steer = _update_steer(keys)
    accel = 1 if keys[K_UP] else 0
    brake = 1 if keys[K_DOWN] else 0
    car_index = None
    for i in range(1, 10):
        if keys[K_1 + i - 1]:
            car_index = i - 1
            break
    return steer, accel, brake, car_index
