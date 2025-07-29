# controls.py
from pygame.locals import *

def get_controls(keys):
    steer = 0
    if keys[K_LEFT]:
        steer += 1
    if keys[K_RIGHT]:
        steer -= 1
    accel = 1 if keys[K_UP] else 0
    brake = 1 if keys[K_DOWN] else 0
    car_index = None
    for i in range(1, 10):
        if keys[K_1 + i - 1]:
            car_index = i - 1
            break
    return steer, accel, brake, car_index
