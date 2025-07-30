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
    def __init__(
        self,
        rel_pos,
        radius,
        suspension_rest,
        spring_k,
        damper_k,
        is_front,
        is_driven,
        width,
    ):
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
        # Amount of suspension compression used in physics calculations
        # (0 = fully extended, suspension_rest = fully compressed)
        self.compression = 0.0


class PowerTrain:
    def __init__(self, torque_curve, gear_ratios, final_drive):
        """Simple powertrain using engine torque curves and gearing."""
        # torque_curve keys come in as strings from JSON, convert and sort
        self.curve = sorted((float(rpm), torque) for rpm, torque in torque_curve.items())
        self.gear_ratios = gear_ratios
        self.final_drive = final_drive
        self.is_cvt = len(gear_ratios) == 2
        self.current_gear = 1
        self.rpm = 0.0
        # store rpm for peak torque for CVT target
        self.peak_rpm = max(self.curve, key=lambda x: x[1])[0]
        self.idle_rpm = self.curve[0][0]
        self.max_rpm = self.curve[-1][0]

    def _torque_at_rpm(self, rpm):
        if rpm <= self.curve[0][0]:
            return self.curve[0][1]
        if rpm >= self.curve[-1][0]:
            return self.curve[-1][1]
        for (r0, t0), (r1, t1) in zip(self.curve[:-1], self.curve[1:]):
            if r0 <= rpm <= r1:
                ratio = (rpm - r0) / (r1 - r0)
                return t0 + ratio * (t1 - t0)
        return self.curve[-1][1]

    def compute_wheel_torque(self, wheel_speed):
        """Return wheel torque and update current gear and rpm."""
        # wheel_speed is rad/s of the wheels (estimated from vehicle speed)
        if self.is_cvt:
            max_ratio = max(self.gear_ratios)
            min_ratio = min(self.gear_ratios)
            if wheel_speed > 0:
                desired = (self.peak_rpm * 2 * math.pi / 60) / (wheel_speed * self.final_drive)
            else:
                desired = max_ratio
            ratio = max(min(desired, max_ratio), min_ratio)
            mid = (max_ratio + min_ratio) / 2
            self.current_gear = 1 if ratio >= mid else 2
        else:
            current_ratio = self.gear_ratios[self.current_gear - 1]
            wheel_rpm = wheel_speed * 60 / (2 * math.pi)
            rpm_est = wheel_rpm * current_ratio * self.final_drive

            shift_up = self.max_rpm * 0.9
            shift_down = self.max_rpm * 0.4

            if rpm_est > shift_up and self.current_gear < len(self.gear_ratios):
                self.current_gear += 1
                current_ratio = self.gear_ratios[self.current_gear - 1]
                rpm_est = wheel_rpm * current_ratio * self.final_drive
            elif rpm_est < shift_down and self.current_gear > 1:
                self.current_gear -= 1
                current_ratio = self.gear_ratios[self.current_gear - 1]
                rpm_est = wheel_rpm * current_ratio * self.final_drive

            ratio = current_ratio
            rpm_est = max(self.idle_rpm, min(rpm_est, self.max_rpm))
            self.rpm = rpm_est
            engine_torque = self._torque_at_rpm(self.rpm)
            return engine_torque * ratio * self.final_drive


        rpm_est = wheel_speed * ratio * self.final_drive * 60 / (2 * math.pi)
        rpm_est = max(self.idle_rpm, min(rpm_est, self.max_rpm))
        self.rpm = rpm_est
        engine_torque = self._torque_at_rpm(self.rpm)
        return engine_torque * ratio * self.final_drive


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
        if x < dx or x > self.size - dx or z < dx or z > self.size - dx:
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
        # Allow per-car brake torque tuning; default to original value
        self.brake_torque = car_data.get("brake_torque_nm", 2800)
        engine_data = car_data.get("engine", {})
        torque_curve = engine_data.get("torque_curve", {})
        gear_ratios = engine_data.get("gear_ratios", [1.0])
        final_drive = engine_data.get("final_drive", 1.0)
        self.powertrain = PowerTrain(torque_curve, gear_ratios, final_drive)
        self.engine_rpm = 0.0
        self.current_gear = 1
        spring_k_front = car_data["spring_k_N_per_m"]["front"]
        spring_k_rear = car_data["spring_k_N_per_m"]["rear"]
        damper_k_front = car_data["damper_k_Ns_per_m"]["front"]
        damper_k_rear = car_data["damper_k_Ns_per_m"]["rear"]
        drag_coeff = car_data["drag_coeff"]
        self.rolling_resistance = car_data.get("rolling_resistance", 0.015)
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

        # Wheel positions
        wheel_y = -suspension_rest
        self.wheels = []
        self._build_wheels(
            wheel_y,
            track,
            wheelbase,
            radius,
            suspension_rest,
            spring_k_front,
            spring_k_rear,
            damper_k_front,
            damper_k_rear,
            drive_type,
            tire_width,
        )
        self.steer = 0
        self.accel = 0
        self.brake = 0
        self.drive_torque_per_wheel = 0.0

    def _update_powertrain(self):
        driven = [w for w in self.wheels if w.is_driven]
        if not driven:
            self.drive_torque_per_wheel = 0.0
            self.engine_rpm = self.powertrain.idle_rpm
            self.current_gear = 0
            return

        # estimate wheel speed from vehicle linear speed for more stable shifting
        speed = np.linalg.norm(self.body.vel)
        wheel_speed = speed / driven[0].radius

        base_torque = self.powertrain.compute_wheel_torque(wheel_speed)
        self.engine_rpm = self.powertrain.rpm
        self.current_gear = self.powertrain.current_gear
        if self.accel <= 0:
            self.drive_torque_per_wheel = 0.0
            return
        total_torque = base_torque * self.accel
        self.drive_torque_per_wheel = total_torque / len(driven)

    def _build_wheels(
        self,
        wheel_y,
        track,
        wheelbase,
        radius,
        suspension_rest,
        spring_k_front,
        spring_k_rear,
        damper_k_front,
        damper_k_rear,
        drive_type,
        tire_width,
    ):
        positions = [
            (-track / 2, wheel_y, wheelbase / 2),
            (track / 2, wheel_y, wheelbase / 2),
            (-track / 2, wheel_y, -wheelbase / 2),
            (track / 2, wheel_y, -wheelbase / 2),
        ]
        springs = [spring_k_front, spring_k_front, spring_k_rear, spring_k_rear]
        dampers = [damper_k_front, damper_k_front, damper_k_rear, damper_k_rear]
        fronts = [True, True, False, False]
        drivens = [
            drive_type in ["FWD", "AWD"],
            drive_type in ["FWD", "AWD"],
            drive_type in ["RWD", "AWD"],
            drive_type in ["RWD", "AWD"],
        ]
        for pos, sk, dk, f, d in zip(positions, springs, dampers, fronts, drivens):
            self.wheels.append(
                Wheel(
                    np.array(pos, dtype=float),
                    radius,
                    suspension_rest,
                    sk,
                    dk,
                    f,
                    d,
                    tire_width,
                )
            )

    def _apply_gravity_drag(self):
        gravity = np.array([0.0, -9.81, 0.0]) * self.body.mass
        self.body.apply_force(gravity, self.body.pos)
        speed = np.linalg.norm(self.body.vel)
        if speed:
            drag = -self.drag_coeff * speed * self.body.vel
            self.body.apply_force(drag, self.body.pos)
            rr = -self.rolling_resistance * self.body.mass * 9.81 * (
                self.body.vel / speed
            )
            self.body.apply_force(rr, self.body.pos)

    def _update_wheel(self, wheel, dt, g_front, g_rear):
        pos = self.body.pos + self.body.rot.rotate(wheel.rel_pos)
        if (
            pos[0] < 0
            or pos[0] > self.terrain.size
            or pos[2] < 0
            or pos[2] > self.terrain.size
        ):
            wheel.is_grounded = False
            wheel.compression = 0.0
            return 0
        ground_h = self.terrain.get_height(pos[0], pos[2])
        compression = ground_h + wheel.radius - pos[1]
        wheel.is_grounded = compression > 0 and not self.is_upside_down
        wheel.compression = max(0.0, compression) if wheel.is_grounded else 0.0
        if not wheel.is_grounded:
            return 0
        rel_pos = pos - self.body.pos
        vel_at = self.body.vel + np.cross(self.body.angvel, rel_pos)
        spring_f = wheel.spring_k * compression
        damper_f = -wheel.damper_k * vel_at[1]
        normal_f = max(0, spring_f + damper_f)
        if normal_f == 0:
            return 1
        normal = self.terrain.get_normal(pos[0], pos[2])
        self.body.apply_force(normal * normal_f, pos)
        load = normal_f
        contact_vel = vel_at - normal * np.dot(vel_at, normal)
        if wheel.is_front:
            target = -self.steer * self.steer_limit
            wheel.target_steer = np.clip(target, -self.steer_limit, self.steer_limit)
            wheel.steer_angle += (wheel.target_steer - wheel.steer_angle) * 5 * dt
        else:
            wheel.steer_angle = 0
        local_fwd = np.array(
            [math.sin(wheel.steer_angle), 0, math.cos(wheel.steer_angle)]
        )
        wheel_fwd = self.body.rot.rotate(local_fwd)
        forward = wheel_fwd - normal * np.dot(wheel_fwd, normal)
        forward /= np.linalg.norm(forward) or 1
        right = np.cross(normal, forward)
        right /= np.linalg.norm(right) or 1
        long_v = np.dot(contact_vel, forward)
        lat_v = np.dot(contact_vel, right)
        alpha = math.atan2(lat_v, abs(long_v) + 0.1)
        slip = (wheel.ang_vel * wheel.radius - long_v) / (abs(long_v) + 0.1)
        long_f = 80000 * slip
        lat_f = -80000 * alpha
        is_static = self.brake > 0 and abs(long_v) < 0.3 and abs(wheel.ang_vel) < 0.1
        mu = 1.4 if is_static else 1.2
        width_factor = wheel.width / 0.2
        max_fric = mu * load * width_factor
        gravity_pw = g_front if wheel.is_front else g_rear
        if is_static:
            proj_g = gravity_pw - np.dot(gravity_pw, normal) * normal
            required_long = -np.dot(proj_g, forward)
            long_f += required_long
        total_f = math.hypot(long_f, lat_f)
        if total_f > max_fric > 0:
            scale = max_fric / total_f
            long_f *= scale
            lat_f *= scale
        long_ratio = abs(slip)
        lat_ratio = abs(alpha) / 0.15
        wheel.slip_ratio = (
            0.0
            if abs(long_v) < 0.05 or (self.brake > 0 and abs(long_v) < 0.3)
            else min(max(long_ratio, lat_ratio), 1.0)
        )
        tire_f = forward * long_f + right * lat_f
        self.body.apply_force(tire_f, pos)
        drive_t = (
            self.drive_torque_per_wheel
            if wheel.is_driven and wheel.is_grounded
            else 0
        )
        brake_sign = 0
        if self.brake > 0 and wheel.is_grounded:
            if abs(wheel.ang_vel) > 0.1:
                brake_sign = math.copysign(1, wheel.ang_vel)
            elif abs(long_v) > 0.01:
                brake_sign = math.copysign(1, long_v)
        brake_t = -self.brake * self.brake_torque * brake_sign if brake_sign else 0
        friction_t = 0 if is_static else -long_f * wheel.radius
        if self.brake > 0 and abs(long_v) < 0.3 and wheel.is_grounded:
            wheel.ang_vel = 0
        else:
            ang_acc = (drive_t + brake_t + friction_t) / 10
            wheel.ang_vel += ang_acc * dt
        return 1

    def _handle_collisions(self):
        for point in self.collision_points:
            pos = self.body.pos + self.body.rot.rotate(point)
            ground_h = self.terrain.get_height(pos[0], pos[2])
            if pos[1] > ground_h:
                continue
            normal = self.terrain.get_normal(pos[0], pos[2])
            rel_pos = pos - self.body.pos
            vel = self.body.vel + np.cross(self.body.angvel, rel_pos)
            penetration = ground_h - pos[1]
            spring_f = 35000 * penetration
            damper_f = -3000 * vel[1]
            n_force = max(0, spring_f + damper_f)
            if n_force == 0:
                continue
            self.body.apply_force(normal * n_force, pos)
            contact_vel = vel - normal * np.dot(vel, normal)
            fric_dir = -contact_vel / (np.linalg.norm(contact_vel) + 0.01)
            fric_force = fric_dir * n_force * 0.8
            self.body.apply_force(fric_force, pos)

    def update(self, dt):
        self._apply_gravity_drag()
        self._update_powertrain()
        car_up = self.body.rot.rotate(np.array([0, 1, 0]))
        self.is_upside_down = car_up[1] < -0.7
        g_front = (
            np.array([0.0, -9.81, 0.0])
            * self.body.mass
            * self.weight_distribution["front"]
            / 100
            / 2
        )
        g_rear = (
            np.array([0.0, -9.81, 0.0])
            * self.body.mass
            * self.weight_distribution["rear"]
            / 100
            / 2
        )
        grounded = sum(self._update_wheel(w, dt, g_front, g_rear) for w in self.wheels)
        if self.is_upside_down or grounded < 2:
            self._handle_collisions()
        self.body.update(dt)
