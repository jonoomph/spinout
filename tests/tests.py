import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.physics import Terrain, Car

TARGET_SPEED_MPS = 60 * 0.44704  # 60 mph to m/s


def simulate(car, accel=True):
    dt = 0.01
    substeps = 5
    time = 0.0
    if accel:
        car.accel = 1
        car.brake = 0
        condition = lambda: np.linalg.norm(car.body.vel) < TARGET_SPEED_MPS
    else:
        car.accel = 0
        car.brake = 1
        condition = lambda: np.linalg.norm(car.body.vel) > 0.1

    while condition() and time < 60:
        for _ in range(substeps):
            car.update(dt / substeps)
        time += dt
    return time


def run_test(car_data):
    terrain = Terrain(size=200, res=2, height_scale=0, sigma=0)
    terrain.heights[:] = 0
    car = Car(terrain, car_data)
    start = terrain.size / 2
    car.body.pos = np.array([start, terrain.get_height(start, start) + 1, start])
    accel_time = simulate(car, accel=True)
    # small step with no input before braking
    car.accel = 0
    for _ in range(5):
        car.update(0.002)
    brake_time = simulate(car, accel=False)
    return accel_time, brake_time


with open(os.path.join(os.path.dirname(__file__), '../data/cars.json')) as f:
    CARS = json.load(f)


import pytest

@pytest.mark.parametrize('car_data', CARS, ids=[f"{c['make']}_{c['model']}" for c in CARS])
def test_car_performance(car_data):
    accel_exp = car_data['tests']['0_60_mph_accel_s']
    brake_exp = car_data['tests']['60_0_mph_brake_s']
    accel_t, brake_t = run_test(car_data)
    assert abs(accel_t - accel_exp) / accel_exp <= 0.05
    assert abs(brake_t - brake_exp) / brake_exp <= 0.05
