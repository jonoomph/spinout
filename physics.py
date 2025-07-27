# physics.py
import math
import numpy as np
from scipy.ndimage import gaussian_filter

class Vector3:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar):
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar):
        return self * (1 / scalar)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def magnitude(self):
        return math.sqrt(self.dot(self))

    def normalize(self):
        mag = self.magnitude()
        if mag == 0:
            return Vector3()
        return self / mag

class Quaternion:
    def __init__(self, w=1, x=0, y=0, z=0):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def multiply(self, other):
        w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
        x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
        y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
        z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
        return Quaternion(w, x, y, z)

    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def rotate(self, v):
        qv = Quaternion(0, v.x, v.y, v.z)
        res = self.multiply(qv).multiply(self.conjugate())
        return Vector3(res.x, res.y, res.z)

    def from_axis_angle(self, axis, angle):
        s = math.sin(angle / 2)
        self.w = math.cos(angle / 2)
        self.x = axis.x * s
        self.y = axis.y * s
        self.z = axis.z * s

    def normalize(self):
        mag = math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        if mag > 0:
            self.w /= mag
            self.x /= mag
            self.y /= mag
            self.z /= mag

class RigidBody:
    def __init__(self, mass, inertia):
        self.mass = mass
        self.inertia = inertia  # Vector3 for principal moments
        self.pos = Vector3()
        self.rot = Quaternion()
        self.vel = Vector3()
        self.angvel = Vector3()
        self.force = Vector3()
        self.torque = Vector3()

    def apply_force(self, f, at_pos):
        self.force = self.force + f
        rel_pos = at_pos - self.pos
        torque = rel_pos.cross(f)
        self.torque = self.torque + torque

    def update(self, dt):
        # Linear motion
        accel = self.force * (1 / self.mass)
        self.vel = self.vel + accel * dt
        self.pos = self.pos + self.vel * dt

        # Angular motion
        angaccel = Vector3(
            self.torque.x / self.inertia.x,
            self.torque.y / self.inertia.y,
            self.torque.z / self.inertia.z
        )
        self.angvel = self.angvel + angaccel * dt
        if self.angvel.magnitude() > 0:
            angle = self.angvel.magnitude() * dt
            axis = self.angvel.normalize()
            dq = Quaternion()
            dq.from_axis_angle(axis, angle)
            self.rot = dq.multiply(self.rot)
            self.rot.normalize()

        # Reset accumulators
        self.force = Vector3()
        self.torque = Vector3()

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

class Terrain:
    def __init__(self, size=200, res=50, height_scale=100, sigma=5):
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
        return Vector3(-dyx / (2 * dx), 1, -dyz / (2 * dx)).normalize()

class Car:
    def __init__(self, terrain):
        self.terrain = terrain
        mass = 1500
        inertia = Vector3(2000, 3000, 2500)  # Approximate for a car
        self.body = RigidBody(mass, inertia)
        wheelbase = 2.5
        track = 1.5
        radius = 0.35
        suspension_rest = 0.4
        spring_k = 35000
        damper_k = 3000
        fl = Wheel(Vector3(-track / 2, -suspension_rest, wheelbase / 2), radius, suspension_rest, spring_k, damper_k, True, False)
        fr = Wheel(Vector3(track / 2, -suspension_rest, wheelbase / 2), radius, suspension_rest, spring_k, damper_k, True, False)
        rl = Wheel(Vector3(-track / 2, -suspension_rest, -wheelbase / 2), radius, suspension_rest, spring_k, damper_k, False, True)
        rr = Wheel(Vector3(track / 2, -suspension_rest, -wheelbase / 2), radius, suspension_rest, spring_k, damper_k, False, True)
        self.wheels = [fl, fr, rl, rr]
        self.steer = 0
        self.accel = 0
        self.brake = 0

    def update(self, dt):
        gravity = Vector3(0, -9.81, 0) * self.body.mass
        self.body.apply_force(gravity, self.body.pos)
        for wheel in self.wheels:
            wheel_world_pos = self.body.pos + self.body.rot.rotate(wheel.rel_pos)
            if wheel_world_pos.x < 0 or wheel_world_pos.x > self.terrain.size or wheel_world_pos.z < 0 or wheel_world_pos.z > self.terrain.size:
                continue
            ground_h = self.terrain.get_height(wheel_world_pos.x, wheel_world_pos.z)
            compression = ground_h + wheel.radius - wheel_world_pos.y
            rel_pos = wheel_world_pos - self.body.pos
            vel_at_wheel = self.body.vel + self.body.angvel.cross(rel_pos)
            spring_force = wheel.spring_k * compression
            damper_force = -wheel.damper_k * vel_at_wheel.y
            normal_force = max(0, spring_force + damper_force)
            if normal_force > 0:
                normal = self.terrain.get_normal(wheel_world_pos.x, wheel_world_pos.z)
                force_vec = normal * normal_force
                self.body.apply_force(force_vec, wheel_world_pos)
                load = normal_force
                contact_vel = vel_at_wheel - normal * vel_at_wheel.dot(normal)
                if wheel.is_front:
                    wheel.steer_angle = -self.steer * math.radians(25)
                else:
                    wheel.steer_angle = 0
                local_forward = Vector3(math.sin(wheel.steer_angle), 0, math.cos(wheel.steer_angle))
                wheel_forward = self.body.rot.rotate(local_forward)
                forward_tang = (wheel_forward - normal * wheel_forward.dot(normal)).normalize()
                right_tang = normal.cross(forward_tang).normalize()
                long_vel = contact_vel.dot(forward_tang)
                lat_vel = contact_vel.dot(right_tang)
                long_force = 0
                if wheel.is_driven:
                    long_force += self.accel * 4000
                if self.brake > 0:
                    brake_force = self.brake * 8000
                    long_force -= brake_force * (1 if long_vel > 0 else -1 if long_vel < 0 else 0)
                C_a = 12000  # Cornering stiffness
                alpha = math.atan2(lat_vel, abs(long_vel) + 0.1)
                lat_force = -C_a * alpha
                mu = 0.8
                max_fric = mu * load
                total_slip_force = math.sqrt(long_force**2 + lat_force**2)
                if total_slip_force > max_fric:
                    scale = max_fric / total_slip_force
                    long_force *= scale
                    lat_force *= scale
                tire_force = forward_tang * long_force + right_tang * lat_force
                self.body.apply_force(tire_force, wheel_world_pos)
        self.body.update(dt)