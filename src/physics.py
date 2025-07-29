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
    def __init__(self, rel_pos, radius, suspension_rest, spring_k, damper_k, is_front, is_driven, width):
        self.rel_pos = rel_pos
        self.radius = radius
        self.suspension_rest = suspension_rest
        self.spring_k = spring_k
        self.damper_k = damper_k
        self.is_front = is_front
        self.is_driven = is_driven
        self.width = width
        self.steer_angle = 0
        self.ang_vel = 0
        self.target_steer = 0
        self.slip_ratio = 0.0
        self.is_grounded = True

class Terrain:
    def __init__(self, size=400, res=50, height_scale=100, sigma=5):
        self.size = size
        self.res = res
        self.cell_size = size / (res - 1)
        noise = np.random.uniform(-height_scale, height_scale, (res, res))
        self.heights = gaussian_filter(noise, sigma=sigma)

    def get_height(self, x, z):
        if x < 0 or x > self.size or z < 0 or z > self.size:
            return float("-inf")
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
        if (
            x < dx
            or x > self.size - dx
            or z < dx
            or z > self.size - dx
        ):
            return np.array([0.0, 1.0, 0.0])
        dyx = self.get_height(x + dx, z) - self.get_height(x - dx, z)
        dyz = self.get_height(x, z + dx) - self.get_height(x, z - dx)
        normal = np.array([-dyx / (2 * dx), 1, -dyz / (2 * dx)])
        norm = np.linalg.norm(normal)
        if norm == 0 or not np.isfinite(norm):
            return np.array([0.0, 1.0, 0.0])
        return normal / norm

class Car:
    def __init__(self, terrain, car_data=None):
        self.terrain = terrain
        # Load parameters from car_data
        mass = car_data["mass_kg"]
        inertia = np.array(car_data["inertia_diagonal"], dtype=float)  # [Ixx, Iyy, Izz]
        wheelbase = car_data["wheelbase_m"]
        track = car_data["track_m"]
        radius = car_data["wheel_radius_m"]
        suspension_rest = car_data["suspension_rest_m"]
        tire_width = car_data["tire"]["width_mm"] / 1000.0
        self.engine_torque = car_data.get("engine_torque_nm", 180)
        self.brake_torque = 2800
        spring_k_front = car_data["spring_k_N_per_m"]["front"]
        spring_k_rear = car_data["spring_k_N_per_m"]["rear"]
        damper_k_front = car_data["damper_k_Ns_per_m"]["front"]
        damper_k_rear = car_data["damper_k_Ns_per_m"]["rear"]
        drag_coeff = car_data["drag_coeff"]
        dimensions = car_data["dimensions_m"]
        weight_distribution = car_data["weight_distribution_pct"]
        ground_clearance = car_data["ground_clearance_m"]
        drive_type = car_data["drive_type"]

        self.body = RigidBody(mass, inertia)
        self.dimensions = dimensions
        self.weight_distribution = weight_distribution
        self.ground_clearance = ground_clearance
        self.drag_coeff = drag_coeff
        self.tire_width = tire_width
        self.steer_limit = math.radians(30)
        self.is_upside_down = False

        # Calculate body_offset for rendering to ensure correct height after compression
        half_length = dimensions["length"] / 2
        half_width = dimensions["width"] / 2
        half_height = dimensions["height"] / 2
        g = 9.81
        front_pct = weight_distribution["front"] / 100
        rear_pct = weight_distribution["rear"] / 100
        front_load_per_wheel = mass * g * front_pct / 2
        rear_load_per_wheel = mass * g * rear_pct / 2
        front_compression = front_load_per_wheel / spring_k_front
        rear_compression = rear_load_per_wheel / spring_k_rear
        sag_cog = front_compression * front_pct + rear_compression * rear_pct
        # Keep the bottom clearance equal to the specified ground clearance when
        # the car is resting under its own weight.
        self.body_offset = ground_clearance - radius - suspension_rest + sag_cog + half_height

        # Define top corner collision points with offset
        self.collision_points = [
            np.array([half_width, half_height + self.body_offset, half_length]),
            np.array([-half_width, half_height + self.body_offset, half_length]),
            np.array([half_width, half_height + self.body_offset, -half_length]),
            np.array([-half_width, half_height + self.body_offset, -half_length]),
        ]

        # Wheel positions (no body_offset in physics)
        wheel_y = -suspension_rest
        fl = Wheel(
            np.array([-track / 2, wheel_y, wheelbase / 2], dtype=float),
            radius,
            suspension_rest,
            spring_k_front,
            damper_k_front,
            True,
            drive_type in ["FWD", "AWD"],
            tire_width,
        )
        fr = Wheel(
            np.array([track / 2, wheel_y, wheelbase / 2], dtype=float),
            radius,
            suspension_rest,
            spring_k_front,
            damper_k_front,
            True,
            drive_type in ["FWD", "AWD"],
            tire_width,
        )
        rl = Wheel(
            np.array([-track / 2, wheel_y, -wheelbase / 2], dtype=float),
            radius,
            suspension_rest,
            spring_k_rear,
            damper_k_rear,
            False,
            drive_type in ["RWD", "AWD"],
            tire_width,
        )
        rr = Wheel(
            np.array([track / 2, wheel_y, -wheelbase / 2], dtype=float),
            radius,
            suspension_rest,
            spring_k_rear,
            damper_k_rear,
            False,
            drive_type in ["RWD", "AWD"],
            tire_width,
        )
        self.wheels = [fl, fr, rl, rr]
        self.steer = 0
        self.accel = 0
        self.brake = 0

    def update(self, dt):
        gravity = np.array([0, -9.81, 0], dtype=float) * self.body.mass
        self.body.apply_force(gravity, self.body.pos)
        vel_mag = np.linalg.norm(self.body.vel)
        if vel_mag > 0:
            drag_dir = -self.body.vel / vel_mag
            drag_force = self.drag_coeff * vel_mag**2 * drag_dir
            self.body.apply_force(drag_force, self.body.pos)

        # Check car orientation and collision points
        car_up = self.body.rot.rotate(np.array([0, 1, 0]))
        self.is_upside_down = car_up[1] < -0.7  # Car is upside down if up vector points significantly downward
        wheels_grounded = 0
        gravity_front = np.array([0, -9.81, 0], dtype=float) * self.body.mass * self.weight_distribution["front"] / 100 / 2
        gravity_rear = np.array([0, -9.81, 0], dtype=float) * self.body.mass * self.weight_distribution["rear"] / 100 / 2

        # Handle wheel forces only for grounded wheels
        for idx, wheel in enumerate(self.wheels):
            wheel_world_pos = self.body.pos + self.body.rot.rotate(wheel.rel_pos)
            if wheel_world_pos[0] < 0 or wheel_world_pos[0] > self.terrain.size or wheel_world_pos[2] < 0 or wheel_world_pos[2] > self.terrain.size:
                wheel.is_grounded = False
                continue
            ground_h = self.terrain.get_height(wheel_world_pos[0], wheel_world_pos[2])
            compression = ground_h + wheel.radius - wheel_world_pos[1]
            wheel.is_grounded = compression > 0 and not self.is_upside_down
            if not wheel.is_grounded:
                continue
            wheels_grounded += 1
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
                width_factor = wheel.width / 0.2
                max_fric = mu * load * width_factor
                gravity_per_wheel = gravity_front if wheel.is_front else gravity_rear
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
                if wheel.is_grounded:
                    self.body.apply_force(tire_force, wheel_world_pos)
                drive_torque = (
                    self.accel * self.engine_torque * 8
                    if wheel.is_driven and wheel.is_grounded
                    else 0
                )
                brake_sign = math.copysign(1, wheel.ang_vel) if abs(wheel.ang_vel) > 0 else 0
                brake_torque = -self.brake * self.brake_torque * brake_sign if self.brake > 0 and wheel.is_grounded else 0
                friction_torque = 0 if is_static_condition else -long_force * wheel.radius
                wheel_inertia = 10
                if self.brake > 0 and abs(long_vel) < 0.5 and wheel.is_grounded:
                    wheel.ang_vel = 0
                else:
                    ang_accel = (drive_torque + brake_torque + friction_torque) / wheel_inertia
                    wheel.ang_vel += ang_accel * dt

        # Handle collision points for top corners when upside down
        if self.is_upside_down or wheels_grounded < 2:
            for point in self.collision_points:
                point_world_pos = self.body.pos + self.body.rot.rotate(point)
                ground_h = self.terrain.get_height(point_world_pos[0], point_world_pos[2])
                if point_world_pos[1] <= ground_h:
                    normal = self.terrain.get_normal(point_world_pos[0], point_world_pos[2])
                    rel_pos = point_world_pos - self.body.pos
                    vel_at_point = self.body.vel + np.cross(self.body.angvel, rel_pos)
                    penetration = ground_h - point_world_pos[1]
                    spring_force = 35000 * penetration  # Use similar spring constant as wheels
                    damper_force = -3000 * vel_at_point[1]  # Use similar damper constant
                    normal_force = max(0, spring_force + damper_force)
                    if normal_force > 0:
                        force_vec = normal * normal_force
                        self.body.apply_force(force_vec, point_world_pos)
                        # Apply friction
                        contact_vel = vel_at_point - normal * np.dot(vel_at_point, normal)
                        friction_dir = -contact_vel / (np.linalg.norm(contact_vel) + 0.01)
                        friction_force = friction_dir * normal_force * 0.8  # Friction coefficient
                        self.body.apply_force(friction_force, point_world_pos)

        self.body.update(dt)
