# physics.py
import math
import numpy as np
from scipy.ndimage import gaussian_filter

class Quaternion:
    def __init__(self, w=1, x=0, y=0, z=0):
        self.arr = np.array([w, x, y, z], dtype=float)

    @property
    def w(self):
        return self.arr[0]

    @property
    def x(self):
        return self.arr[1]

    @property
    def y(self):
        return self.arr[2]

    @property
    def z(self):
        return self.arr[3]

    def multiply(self, other):
        a = self.arr
        b = other.arr
        w = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
        x = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2]
        y = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1]
        z = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]
        return Quaternion(w, x, y, z)

    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def rotate(self, v):
        qv = Quaternion(0, v[0], v[1], v[2])
        res = self.multiply(qv).multiply(self.conjugate())
        return np.array([res.x, res.y, res.z])

    def from_axis_angle(self, axis, angle):
        s = math.sin(angle / 2)
        self.arr[0] = math.cos(angle / 2)
        self.arr[1:] = axis * s

    def normalize(self):
        mag = np.linalg.norm(self.arr)
        if mag > 0:
            self.arr /= mag

class RigidBody:
    def __init__(self, mass, inertia):
        self.mass = mass
        self.inertia = np.array(inertia, dtype=float)
        self.pos = np.zeros(3, dtype=float)
        self.rot = Quaternion()
        self.vel = np.zeros(3, dtype=float)
        self.angvel = np.zeros(3, dtype=float)
        self.force = np.zeros(3, dtype=float)
        self.torque = np.zeros(3, dtype=float)

    def apply_force(self, f, at_pos):
        self.force += f
        rel_pos = at_pos - self.pos
        torque = np.cross(rel_pos, f)
        self.torque += torque

    def update(self, dt):
        accel = self.force / self.mass
        self.vel += accel * dt
        self.pos += self.vel * dt
        angaccel = self.torque / self.inertia
        self.angvel += angaccel * dt
        if np.linalg.norm(self.angvel) > 0:
            angle = np.linalg.norm(self.angvel) * dt
            axis = self.angvel / np.linalg.norm(self.angvel)
            dq = Quaternion()
            dq.from_axis_angle(axis, angle)
            self.rot = dq.multiply(self.rot)
            self.rot.normalize()
        self.force[:] = 0
        self.torque[:] = 0

class Wheel:
    def __init__(self, rel_pos, radius, suspension_rest, spring_k, damper_k, is_front, is_driven):
        self.rel_pos = rel_pos
        self.radius = radius
        self.suspension_rest = suspension_rest
        self.spring_k = spring_k
        self.damper_k = damper_k
        self.is_front = is_front
        self.is_driven = is_driven
        self.steer_angle = 0
        self.ang_vel = 0
        self.target_steer = 0
        self.slip_ratio = 0.0

class Terrain:
    def __init__(self, size=400, res=50, height_scale=100, sigma=5):
        self.size = size
        self.res = res
        self.cell_size = size / (res - 1)
        noise = np.random.uniform(-height_scale, height_scale, (res, res))
        self.heights = gaussian_filter(noise, sigma=sigma)

    def get_height(self, x, z):
        if x < 0 or x > self.size or z < 0 or z > self.size:
            return 0
        ix = min(max(int(x / self.cell_size), 0), self.res - 2)
        iz = min(max(int(z / self.cell_size), 0), self.res - 2)
        fx = (x - ix * self.cell_size) / self.cell_size
        fz = (z - iz * self.cell_size) / self.cell_size
        h00 = self.heights[ix, iz]
        h10 = self.heights[ix + 1, iz]
        h01 = self.heights[ix, iz + 1]
        h11 = self.heights[ix + 1, iz + 1]
        h0 = h00 * (1 - fx) + h10 * fx
        h1 = h01 * (1 - fx) + h11 * fx
        return h0 * (1 - fz) + h1 * fz

    def get_normal(self, x, z):
        dx = 0.1
        dyx = self.get_height(x + dx, z) - self.get_height(x - dx, z)
        dyz = self.get_height(x, z + dx) - self.get_height(x, z - dx)
        normal = np.array([-dyx / (2 * dx), 1, -dyz / (2 * dx)])
        return normal / np.linalg.norm(normal)

class Car:
    def __init__(self, terrain):
        self.terrain = terrain
        mass = 1500
        inertia = np.array([2000, 3000, 2500], dtype=float)
        self.body = RigidBody(mass, inertia)
        wheelbase = 2.5
        track = 1.5
        radius = 0.35
        suspension_rest = 0.4
        spring_k = 35000
        damper_k = 3000
        fl = Wheel(np.array([-track / 2, -suspension_rest, wheelbase / 2], dtype=float), radius, suspension_rest, spring_k, damper_k, True, False)
        fr = Wheel(np.array([track / 2, -suspension_rest, wheelbase / 2], dtype=float), radius, suspension_rest, spring_k, damper_k, True, False)
        rl = Wheel(np.array([-track / 2, -suspension_rest, -wheelbase / 2], dtype=float), radius, suspension_rest, spring_k, damper_k, False, True)
        rr = Wheel(np.array([track / 2, -suspension_rest, -wheelbase / 2], dtype=float), radius, suspension_rest, spring_k, damper_k, False, True)
        self.wheels = [fl, fr, rl, rr]
        self.steer = 0
        self.accel = 0
        self.brake = 0
        self.drag_coeff = 0.3
        self.steer_limit = math.radians(30)

    def update(self, dt):
        gravity = np.array([0, -9.81, 0], dtype=float) * self.body.mass
        self.body.apply_force(gravity, self.body.pos)
        vel_mag = np.linalg.norm(self.body.vel)
        if vel_mag > 0:
            drag_dir = -self.body.vel / vel_mag
            drag_force = self.drag_coeff * vel_mag**2 * drag_dir
            self.body.apply_force(drag_force, self.body.pos)

        gravity_per_wheel = np.array([0, -9.81, 0], dtype=float) * self.body.mass / len(self.wheels)
        for wheel in self.wheels:
            wheel_world_pos = self.body.pos + self.body.rot.rotate(wheel.rel_pos)
            if wheel_world_pos[0] < 0 or wheel_world_pos[0] > self.terrain.size or wheel_world_pos[2] < 0 or wheel_world_pos[2] > self.terrain.size:
                continue
            ground_h = self.terrain.get_height(wheel_world_pos[0], wheel_world_pos[2])
            compression = ground_h + wheel.radius - wheel_world_pos[1]
            rel_pos = wheel_world_pos - self.body.pos
            vel_at_wheel = self.body.vel + np.cross(self.body.angvel, rel_pos)
            spring_force = wheel.spring_k * compression
            damper_force = -wheel.damper_k * vel_at_wheel[1]
            normal_force = max(0, spring_force + damper_force)
            if normal_force > 0:
                normal = self.terrain.get_normal(wheel_world_pos[0], wheel_world_pos[2])
                force_vec = normal * normal_force
                self.body.apply_force(force_vec, wheel_world_pos)
                load = normal_force
                contact_vel = vel_at_wheel - normal * np.dot(vel_at_wheel, normal)
                if wheel.is_front:
                    target_steer = -self.steer * self.steer_limit
                    wheel.target_steer = max(-self.steer_limit, min(self.steer_limit, target_steer))
                    wheel.steer_angle += (wheel.target_steer - wheel.steer_angle) * 5 * dt
                else:
                    wheel.steer_angle = 0
                local_forward = np.array([math.sin(wheel.steer_angle), 0, math.cos(wheel.steer_angle)], dtype=float)
                wheel_forward = self.body.rot.rotate(local_forward)
                forward_tang = wheel_forward - normal * np.dot(wheel_forward, normal)
                forward_tang_norm = np.linalg.norm(forward_tang)
                forward_tang = forward_tang / forward_tang_norm if forward_tang_norm > 0 else forward_tang
                right_tang = np.cross(normal, forward_tang)
                right_tang_norm = np.linalg.norm(right_tang)
                right_tang = right_tang / right_tang_norm if right_tang_norm > 0 else right_tang
                long_vel = np.dot(contact_vel, forward_tang)
                lat_vel = np.dot(contact_vel, right_tang)
                C_a = 80000
                C_long = 80000
                alpha = math.atan2(lat_vel, abs(long_vel) + 0.1)
                slip = (wheel.ang_vel * wheel.radius - long_vel) / (abs(long_vel) + 0.1)
                long_force = C_long * slip
                lat_force = -C_a * alpha
                mu_static = 1.4
                mu_dynamic = 1.2
                is_static_condition = self.brake > 0 and abs(long_vel) < 0.5 and abs(wheel.ang_vel) < 0.1
                mu = mu_static if is_static_condition else mu_dynamic
                max_fric = mu * load
                if is_static_condition:
                    projected_g = gravity_per_wheel - np.dot(gravity_per_wheel, normal) * normal
                    required_long = -np.dot(projected_g, forward_tang)
                    long_force = np.clip(required_long, -max_fric, max_fric)
                total_slip_force = math.sqrt(long_force**2 + lat_force**2)
                long_slip_ratio = abs(slip)
                lateral_slip_ratio = abs(alpha) / 0.15
                wheel.slip_ratio = 0.0 if abs(long_vel) < 0.05 or (self.brake > 0 and abs(long_vel) < 0.5) else min(max(long_slip_ratio, lateral_slip_ratio), 1.0)
                if total_slip_force > max_fric and max_fric > 0:
                    scale = max_fric / total_slip_force
                    long_force *= scale
                    lat_force *= scale
                tire_force = forward_tang * long_force + right_tang * lat_force
                self.body.apply_force(tire_force, wheel_world_pos)
                drive_torque = self.accel * 1400 if wheel.is_driven else 0
                brake_sign = math.copysign(1, wheel.ang_vel) if abs(wheel.ang_vel) > 0 else 0
                brake_torque = -self.brake * 2800 * brake_sign if self.brake > 0 else 0
                friction_torque = 0 if is_static_condition else -long_force * wheel.radius
                wheel_inertia = 10
                if self.brake > 0 and abs(long_vel) < 0.5:
                    wheel.ang_vel = 0
                else:
                    ang_accel = (drive_torque + brake_torque + friction_torque) / wheel_inertia
                    wheel.ang_vel += ang_accel * dt
        self.body.update(dt)