# controls.py
from pygame.locals import *

def get_controls(keys):
    steer = 0
    if keys[K_LEFT]:
        steer -= 1
    if keys[K_RIGHT]:
        steer += 1
    accel = 1 if keys[K_UP] else 0
    brake = 1 if keys[K_DOWN] else 0
    return steer, accel, brake