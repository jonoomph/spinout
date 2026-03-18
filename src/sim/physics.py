# physics.py
import math
import numpy as np
from scipy.ndimage import gaussian_filter
from .colors import TERRAIN_DEFAULT_COLOR
from .constants import AIR_DENSITY

# Maximum discrete input steps shared with ``controls``.
STEER_MAX = 128
ACCEL_MAX = 32
BRAKE_MAX = 32


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
        # Compute world-space inertia tensor
        w, x, y, z = self.rot.arr
        R = np.array(
            [
                [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
            ]
        )
        I_body = np.diag(self.inertia)
        I_world = R @ I_body @ R.T
        angaccel = np.linalg.solve(
            I_world, self.torque - np.cross(self.angvel, I_world @ self.angvel)
        )
        self.angvel += angaccel * dt
        angvel_mag = np.linalg.norm(self.angvel)
        if angvel_mag > 0:
            angle = angvel_mag * dt
            axis = self.angvel / angvel_mag
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
        suspension_travel,
        spring_k,
        damper_k,
        is_front,
        is_driven,
        width,
    ):
        self.rel_pos = rel_pos
        self.radius = radius
        self.suspension_travel = suspension_travel
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
        # (0 = fully extended, suspension_travel = fully compressed)
        self.compression = 0.0
        # Flag indicating the suspension has reached its travel limit and
        # the wheel is "bottomed out" against the chassis
        self.bottomed = False


class PowerTrain:
    def __init__(self, torque_curve, gear_ratios, final_drive, stall_rpm=None):
        """Simple powertrain using engine torque curves and gearing."""
        # torque_curve keys come in as strings from JSON, convert and sort
        self.curve = sorted((float(rpm), torque) for rpm, torque in torque_curve.items())
        self.gear_ratios = gear_ratios
        self.final_drive = final_drive
        self.is_cvt = len(gear_ratios) == 2
        self.current_gear = 1
        self.current_ratio = gear_ratios[0] if gear_ratios else 1.0
        self.rpm = 0.0
        # For CVTs target the rpm that provides peak power rather than torque
        self.cvt_power_rpm = max(self.curve, key=lambda x: x[0] * x[1])[0]
        self.idle_rpm = self.curve[0][0]
        self.max_rpm = self.curve[-1][0]
        # stall speed assumption for automatic transmissions
        default_stall = min(2200, max(1500, self.max_rpm * 0.3))
        self.stall_rpm = stall_rpm if stall_rpm is not None else default_stall

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

    def compute_wheel_torque(self, wheel_speed, throttle):
        """Return wheel torque and update current gear and rpm."""
        # wheel_speed is rad/s of the wheels (estimated from vehicle speed)
        if self.is_cvt:
            max_ratio = max(self.gear_ratios)
            min_ratio = min(self.gear_ratios)
            if wheel_speed > 0:
                desired = (
                    self.cvt_power_rpm * 2 * math.pi / 60
                ) / (wheel_speed * self.final_drive)
            else:
                desired = max_ratio
            ratio = max(min(desired, max_ratio), min_ratio)
            self.current_ratio = ratio
            mid = (max_ratio + min_ratio) / 2
            self.current_gear = 1 if ratio >= mid else 2
        else:
            current_ratio = self.gear_ratios[self.current_gear - 1]
            wheel_rpm = wheel_speed * 60 / (2 * math.pi)
            rpm_est = wheel_rpm * current_ratio * self.final_drive
            # simple stall behaviour: allow engine to rev even at low wheel speed
            if throttle > 0 and wheel_speed < 5.0 and self.stall_rpm:
                rpm_est = max(rpm_est, self.stall_rpm)

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
            self.current_ratio = ratio
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
    def __init__(
        self,
        width: float = 400.0,
        height: float = 1200.0,
        # Use a coarse default resolution so rectangular maps match the
        # original ~50×50 square mesh in complexity
        res: int = 30,
        height_scale: float = 100.0,
        sigma: float = 5.0,
        terrain_type: str = "grass",
        color=None,
        friction: float = 1.0,
    ):
        self.width = width
        self.height = height
        # Maintain roughly square cells regardless of aspect ratio
        self.res_x = res
        self.res_z = int(round((height / width) * (res - 1))) + 1
        # Preserve legacy attribute for callers assuming square terrain
        self.res = self.res_x
        self.cell_size_x = width / (self.res_x - 1)
        self.cell_size_z = height / (self.res_z - 1)
        noise = np.random.uniform(-height_scale, height_scale, (self.res_x, self.res_z))
        # Use equal smoothing in both axes (meters)
        self.heights = gaussian_filter(noise, sigma=[sigma, sigma])
        self.terrain_type = terrain_type
        self.color = color or list(TERRAIN_DEFAULT_COLOR)
        self.base_friction = friction
        self.surface_friction = np.full((self.res_x, self.res_z), friction, dtype=float)
        # Separate map for road friction so terrain choice does not affect roads
        self.road_friction = np.zeros((self.res_x, self.res_z), dtype=float)
        # Optional road surface geometry used for collision queries
        self.road_surface = None

    def _height_from_grid(self, x, z):
        if x < 0 or x > self.width or z < 0 or z > self.height:
            return float("-inf")
        ix = min(max(int(x / self.cell_size_x), 0), self.res_x - 2)
        iz = min(max(int(z / self.cell_size_z), 0), self.res_z - 2)
        fx = (x - ix * self.cell_size_x) / self.cell_size_x
        fz = (z - iz * self.cell_size_z) / self.cell_size_z
        h00 = self.heights[ix, iz]
        h10 = self.heights[ix + 1, iz]
        h01 = self.heights[ix, iz + 1]
        h11 = self.heights[ix + 1, iz + 1]
        h0 = h00 * (1 - fx) + h10 * fx
        h1 = h01 * (1 - fx) + h11 * fx
        return h0 * (1 - fz) + h1 * fz

    def get_height(self, x, z, include_roads: bool = True):
        base = self._height_from_grid(x, z)
        if not include_roads:
            return base
        road_surface = getattr(self, "road_surface", None)
        if road_surface is not None:
            road_h = road_surface.height_at(x, z)
            if road_h is not None:
                return float(road_h)
        return base

    def get_normal(self, x, z):
        dx = 0.1
        if x < dx or x > self.width - dx or z < dx or z > self.height - dx:
            return np.array([0.0, 1.0, 0.0])
        dyx = self.get_height(x + dx, z) - self.get_height(x - dx, z)
        dyz = self.get_height(x, z + dx) - self.get_height(x, z - dx)
        normal = np.array([-dyx / (2 * dx), 1, -dyz / (2 * dx)])
        norm = np.linalg.norm(normal)
        if norm == 0 or not np.isfinite(norm):
            return np.array([0.0, 1.0, 0.0])
        return normal / norm

    def get_friction(self, x, z):
        if x < 0 or x > self.width or z < 0 or z > self.height:
            return self.base_friction
        ix = min(max(int(x / self.cell_size_x), 0), self.res_x - 1)
        iz = min(max(int(z / self.cell_size_z), 0), self.res_z - 1)
        road_mu = self.road_friction[ix, iz]
        if road_mu > 0:
            return float(road_mu)
        return float(self.surface_friction[ix, iz])


class Car:
    def __init__(self, terrain, car_data=None):
        self.terrain = terrain
        # Load parameters from car_data
        mass = car_data["mass_kg"]
        inertia = np.array(car_data["inertia_diagonal"], dtype=float)
        wheelbase = car_data["wheelbase_m"]
        track = car_data["track_m"]
        tire = car_data["tire"]
        tire_width = tire["width_mm"] / 1000.0
        sidewall = tire_width * (tire["aspect_ratio_pct"] / 100.0)
        rim_radius = tire["rim_diameter_in"] * 0.0254 / 2
        radius_calc = rim_radius + sidewall
        if "wheel_radius_m" in car_data:
            radius_raw = car_data["wheel_radius_m"]
            if abs(radius_raw - radius_calc) / radius_calc > 0.02:
                print(
                    f"Warning: wheel radius mismatch for {car_data['model']}, using tire-derived value"
                )
                radius = radius_calc
            else:
                radius = radius_raw
        else:
            radius = radius_calc
        self.cg_height_m = car_data.get("cg_height_m", radius)
        suspension_travel = car_data.get("suspension_travel_m", 0.25)
        brake_info = car_data.get("brakes", {"base_torque_nm": 8000, "front_bias_pct": 65})
        self.brake_base_torque = brake_info.get("base_torque_nm", 8000)
        self.brake_bias = brake_info.get("front_bias_pct", 65)
        self.ground_clearance = car_data.get("ground_clearance_m", 0.15)
        engine_data = car_data.get("engine", {})
        torque_curve = engine_data.get("torque_curve", {})
        gear_ratios = engine_data.get("gear_ratios", [1.0])
        final_drive = engine_data.get("final_drive", 1.0)
        stall_rpm = engine_data.get("stall_rpm")
        self.powertrain = PowerTrain(torque_curve, gear_ratios, final_drive, stall_rpm)
        self.drive_efficiency = engine_data.get("drivetrain_efficiency", 0.05)
        trans = car_data.get("transmission", {})
        self.trans_type = trans.get("type", "auto")
        self.lockup_speed = trans.get("lockup_speed_mps", 5.0)
        self.engine_rpm = 0.0
        self.current_gear = 1
        spring_k_front = car_data["spring_k_N_per_m"]["front"]
        spring_k_rear = car_data["spring_k_N_per_m"]["rear"]
        damper_k_front = car_data["damper_k_Ns_per_m"]["front"]
        damper_k_rear = car_data["damper_k_Ns_per_m"]["rear"]
        drag_coeff = car_data["drag_coeff"]
        self.frontal_area = car_data.get("frontal_area_m2", 2.2)
        rr_data = car_data.get("rolling_resistance", 0.015)
        if isinstance(rr_data, dict):
            self.c_rr0 = rr_data.get("c_rr0", 0.015)
            self.rr_k_v = rr_data.get("k_v", 0.0002)
        else:
            self.c_rr0 = rr_data
            self.rr_k_v = 0.0002
        dimensions = car_data["dimensions_m"]
        weight_distribution = car_data["weight_distribution_pct"]
        drive_type = car_data["drive_type"]

        # Optional steering response tuning. ``strength`` blends between
        # linear (0) and cubic (1) response while ``gain`` scales the result.
        steer_curve = car_data.get("steering_curve", {})
        self.steer_curve_gain = steer_curve.get("gain", 1.0)
        self.steer_curve_strength = steer_curve.get("strength", 0.5)
        # Maximum steering angle at very low speed and scaling factor for
        # speed-sensitive limiting (speed at which the limit halves).
        self.max_steer_angle = math.radians(car_data.get("steer_angle_deg", 30))
        self.speed_steer_scale = car_data.get("steering_speed_scale", 10.0)
        # Steering filter rate (s⁻¹).  τ = 1/rate.  Default 20 → τ=50 ms.
        # Real EPS systems: 25–50 s⁻¹ (τ≈20–40 ms).
        self.steering_rate = car_data.get("steering_rate", 20.0)
        self.wheelbase_m = wheelbase

        # Tire stiffness controls how quickly forces saturate with slip and slip angle.
        tire_stiffness = car_data.get("tire_stiffness", {})
        self.long_stiffness = tire_stiffness.get("longitudinal", 10.0)
        self.lat_stiffness = tire_stiffness.get("lateral", 5.0)
        self.front_lat_scale = car_data.get("front_lat_scale", 1.1)

        self.body = RigidBody(mass, inertia)
        # quick sanity check for inertia vs box estimate
        l, w, h = dimensions["length"], dimensions["width"], dimensions["height"]
        box = np.array([
            mass * (h**2 + l**2) / 12,
            mass * (w**2 + l**2) / 12,
            mass * (w**2 + h**2) / 12,
        ])
        diff = np.abs(inertia - box) / box
        if np.any(diff > 0.35):
            print(f"Warning: inertia for {car_data['model']} differs by >35%")
        self.dimensions = dimensions
        self.weight_distribution = weight_distribution
        self.drag_coeff = drag_coeff
        self.driveline_drag = car_data.get("driveline_drag_nm_s", 0.2)
        self.bearing_drag = car_data.get("bearing_drag_nm_s", 0.07)
        self.brake_pad_drag = car_data.get("brake_pad_drag_nm", 2.5)
        self.tire_width = tire_width
        self.is_upside_down = False
        self.slip_events: list[dict] = []
        self.wind_velocity = np.zeros(3, dtype=float)
        self.show_wind_vectors = False

        # Calculate body_offset so the body's geometric centre aligns with the CG
        half_length = dimensions["length"] / 2
        half_width = dimensions["width"] / 2
        half_height = dimensions["height"] / 2
        self.body_offset = half_height - self.cg_height_m + self.ground_clearance

        # Collision points aligned with the rendered box. Include edge
        # midpoints and center on the roof, floor, and sides so the car does
        # not clip into the ground when rolling onto its side or roof.
        top_y = half_height + self.body_offset
        bottom_y = -half_height + self.body_offset
        mid_y = self.body_offset
        self.collision_points = [
            # top corners
            np.array([ half_width,  top_y,  half_length]),
            np.array([-half_width, top_y,  half_length]),
            np.array([ half_width,  top_y, -half_length]),
            np.array([-half_width, top_y, -half_length]),
            # top edge midpoints and center
            np.array([0.0,      top_y,  half_length]),
            np.array([0.0,      top_y, -half_length]),
            np.array([ half_width, top_y, 0.0]),
            np.array([-half_width, top_y, 0.0]),
            np.array([0.0,      top_y, 0.0]),
            # bottom corners
            np.array([ half_width,  bottom_y,  half_length]),
            np.array([-half_width, bottom_y,  half_length]),
            np.array([ half_width,  bottom_y, -half_length]),
            np.array([-half_width, bottom_y, -half_length]),
            # bottom edge midpoints and center
            np.array([0.0,      bottom_y,  half_length]),
            np.array([0.0,      bottom_y, -half_length]),
            np.array([ half_width, bottom_y, 0.0]),
            np.array([-half_width, bottom_y, 0.0]),
            np.array([0.0,      bottom_y, 0.0]),
            # side centers and edge midpoints
            np.array([ half_width, mid_y,  half_length]),
            np.array([ half_width, mid_y, -half_length]),
            np.array([-half_width, mid_y,  half_length]),
            np.array([-half_width, mid_y, -half_length]),
            np.array([ half_width, mid_y, 0.0]),
            np.array([-half_width, mid_y, 0.0]),
        ]

        # Wheel positions with static compression relative to the CG
        wheel_y = -(self.cg_height_m - radius)
        g = 9.81
        front_load = mass * g * weight_distribution["front"] / 100 / 2
        rear_load = mass * g * weight_distribution["rear"] / 100 / 2
        front_comp = front_load / spring_k_front
        rear_comp = rear_load / spring_k_rear
        self.wheels = []
        self._build_wheels(
            wheel_y,
            track,
            wheelbase,
            radius,
            suspension_travel,
            spring_k_front,
            spring_k_rear,
            damper_k_front,
            damper_k_rear,
            drive_type,
            tire_width,
            weight_distribution,
            front_comp,
            rear_comp,
        )
        self.steer = 0
        self.accel = 0
        self.brake = 0
        self.drive_torque_per_wheel = 0.0
        engine_brake = engine_data.get("engine_brake", {})
        self.engine_brake_curve = sorted((float(rpm), torque) for rpm, torque in engine_brake.items())

    def apply_inputs(self, steer_steps, accel_steps, brake_steps):
        """Apply discrete control steps to the car.

        ``steer_steps`` is in ``[-128,128]`` and ``accel_steps``/``brake_steps``
        are in ``[0,32]``.  They are normalised here so the rest of the physics
        code continues to operate on ``[-1,1]`` / ``[0,1]`` ranges.
        """

        raw = steer_steps / STEER_MAX
        self.steer = self._steering_curve(raw)
        self.accel = accel_steps / ACCEL_MAX
        self.brake = brake_steps / BRAKE_MAX

    def set_wind(self, wind_vec):
        """Set the ambient wind velocity affecting the car's aerodynamics."""

        if wind_vec is None:
            self.wind_velocity[:] = 0.0
        else:
            self.wind_velocity[:] = wind_vec

    def _steering_curve(self, x):
        """Map raw steering input ``x`` in ``[-1,1]`` through a shallow S curve.

        ``steer_curve_strength`` blends between linear and cubic response while
        ``steer_curve_gain`` scales the result.  The output is clamped back to
        ``[-1, 1]``.
        """

        f = self.steer_curve_strength
        curved = (1 - f) * x + f * x**3
        curved *= self.steer_curve_gain
        return max(-1.0, min(1.0, curved))

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
        base_torque = self.powertrain.compute_wheel_torque(wheel_speed, self.accel)
        self.engine_rpm = self.powertrain.rpm
        self.current_gear = self.powertrain.current_gear
        if self.accel <= 0:
            engine_brake = self._engine_brake_torque(self.engine_rpm)
            if self.trans_type == "auto":
                factor = 0.3 if speed < self.lockup_speed else 1.0
            else:
                factor = 1.0 if (self.brake > 0 or self.accel > 0) else 0.0
            ratio = self.powertrain.current_ratio
            total_torque = (
                -engine_brake
                * ratio
                * self.powertrain.final_drive
                * self.drive_efficiency
                * factor
            )
        else:
            total_torque = base_torque * self.accel
        self.drive_torque_per_wheel = total_torque / len(driven)

    def _build_wheels(
        self,
        wheel_y,
        track,
        wheelbase,
        radius,
        suspension_travel,
        spring_k_front,
        spring_k_rear,
        damper_k_front,
        damper_k_rear,
        drive_type,
        tire_width,
        weight_distribution,
        front_comp,
        rear_comp,
    ):
        front_pct = weight_distribution["front"] / 100
        rear_pct = weight_distribution["rear"] / 100
        front_z = wheelbase * rear_pct
        rear_z = -wheelbase * front_pct
        positions = [
            (-track / 2, wheel_y - front_comp, front_z),
            (track / 2, wheel_y - front_comp, front_z),
            (-track / 2, wheel_y - rear_comp, rear_z),
            (track / 2, wheel_y - rear_comp, rear_z),
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
        comps = [front_comp, front_comp, rear_comp, rear_comp]
        for pos, sk, dk, f, d, comp in zip(positions, springs, dampers, fronts, drivens, comps):
            w = Wheel(
                np.array(pos, dtype=float),
                radius,
                suspension_travel,
                sk,
                dk,
                f,
                d,
                tire_width,
            )
            w.compression = comp
            self.wheels.append(w)

    def _apply_gravity_drag(self):
        gravity = np.array([0.0, -9.81, 0.0]) * self.body.mass
        self.body.apply_force(gravity, self.body.pos)
        rel_vel = self.body.vel - self.wind_velocity
        rel_speed = np.linalg.norm(rel_vel)
        if rel_speed > 1e-6:
            drag_mag = 0.5 * AIR_DENSITY * self.drag_coeff * self.frontal_area * rel_speed**2
            drag = -drag_mag * (rel_vel / rel_speed)
            self.body.apply_force(drag, self.body.pos)

        ground_speed = np.linalg.norm(self.body.vel)
        if ground_speed > 1e-6:
            rr_mag = self.body.mass * 9.81 * (self.c_rr0 + self.rr_k_v * ground_speed)
            rr = -rr_mag * (self.body.vel / ground_speed)
            self.body.apply_force(rr, self.body.pos)

    def _surface_mark_color(self, friction):
        base = np.array(getattr(self.terrain, "color", [0.12, 0.1, 0.08])[:3], dtype=float)
        if friction >= 0.93:
            return [0.05, 0.05, 0.05]
        if friction >= 0.8:
            return [0.08, 0.07, 0.06]
        dark = np.clip(base * 0.45, 0.02, 0.4)
        return dark.tolist()

    def _update_wheel(self, index, wheel, dt, g_front, g_rear):
        pos = self.body.pos + self.body.rot.rotate(wheel.rel_pos)
        if (
            pos[0] < 0
            or pos[0] > self.terrain.width
            or pos[2] < 0
            or pos[2] > self.terrain.height
        ):
            wheel.is_grounded = False
            wheel.compression = 0.0
            return 0
        ground_h = self.terrain.get_height(pos[0], pos[2])
        raw_comp = ground_h + wheel.radius - pos[1]
        wheel.is_grounded = raw_comp > 0 and not self.is_upside_down
        compression = (
            min(max(0.0, raw_comp), wheel.suspension_travel)
            if wheel.is_grounded
            else 0.0
        )
        wheel.bottomed = wheel.is_grounded and raw_comp > wheel.suspension_travel
        wheel.compression = compression
        if not wheel.is_grounded:
            wheel.bottomed = False
            return 0
        rel_pos = pos - self.body.pos
        vel_at = self.body.vel + np.cross(self.body.angvel, rel_pos)
        bump = 0.0
        if wheel.bottomed:
            bump = wheel.spring_k * 50 * (raw_comp - wheel.suspension_travel)
        elif compression > wheel.suspension_travel * 0.9:
            bump = wheel.spring_k * 10 * (compression - wheel.suspension_travel * 0.9)
        spring_f = wheel.spring_k * compression + bump
        damper_f = -wheel.damper_k * vel_at[1]
        normal_f = max(0, spring_f + damper_f)
        if normal_f == 0:
            return 1
        normal = self.terrain.get_normal(pos[0], pos[2])
        self.body.apply_force(normal * normal_f, pos)
        load = normal_f
        contact_vel = vel_at - normal * np.dot(vel_at, normal)
        if wheel.is_front:
            speed = np.linalg.norm(self.body.vel)
            steer_limit = self.max_steer_angle / (1 + speed / self.speed_steer_scale)
            target = -self.steer * steer_limit
            wheel.target_steer = np.clip(target, -steer_limit, steer_limit)
            wheel.steer_angle += (wheel.target_steer - wheel.steer_angle) * self.steering_rate * dt
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
        is_static = self.brake > 0 and abs(long_v) < 0.3 and abs(wheel.ang_vel) < 0.1
        base_mu = self.terrain.get_friction(pos[0], pos[2])
        mu = (1.4 if is_static else 1.2) * base_mu
        width_factor = wheel.width / 0.2
        max_fric = mu * load * width_factor
        long_f = max_fric * math.tanh(self.long_stiffness * slip)
        lat_stiff = self.lat_stiffness * (self.front_lat_scale if wheel.is_front else 1.0)
        lat_f = -max_fric * math.tanh(lat_stiff * alpha)
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
        grip_scale = 1.0 / max(0.55, base_mu + 1e-3)
        ref_angle = math.radians(14.0) * max(0.75, base_mu)
        long_ratio = abs(slip) * grip_scale
        lat_ratio = abs(alpha) / max(ref_angle, 1e-3)
        slip_index = max(long_ratio, lat_ratio)
        if slip_index < 0.03:
            slip_index = 0.0
        low_speed_cutoff = 0.08 if self.brake <= 0 else 0.35
        if abs(long_v) < low_speed_cutoff:
            wheel.slip_ratio = 0.0
        else:
            wheel.slip_ratio = min(slip_index, 1.0)
        slip_strength = wheel.slip_ratio
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
        front_bias = self.brake_bias / 100
        base = self.brake_base_torque * (front_bias if wheel.is_front else (1 - front_bias))
        brake_t = -self.brake * base * brake_sign if brake_sign else 0
        friction_t = 0 if is_static else -long_f * wheel.radius
        if self.brake > 0 and abs(long_v) < 0.3 and wheel.is_grounded:
            wheel.ang_vel = 0
        else:
            extra_t = 0.0
            if wheel.is_driven:
                extra_t -= self.driveline_drag * wheel.ang_vel
            extra_t -= self.bearing_drag * wheel.ang_vel
            pad_sign = 0.0
            if abs(wheel.ang_vel) > 0.1:
                pad_sign = math.copysign(1, wheel.ang_vel)
            elif abs(long_v) > 0.01:
                pad_sign = math.copysign(1, long_v)
            pad_t = -self.brake_pad_drag * pad_sign
            ang_acc = (drive_t + brake_t + friction_t + extra_t + pad_t) / 10
            wheel.ang_vel += ang_acc * dt
        event_kind = None
        slip_velocity = math.hypot(long_v, lat_v)
        slip_trigger = 0.42
        speed_trigger = 0.35
        if base_mu < 0.75:
            grip_modifier = max(0.6, base_mu / 0.75)
            slip_trigger *= grip_modifier
            speed_trigger *= grip_modifier
        if slip_strength > slip_trigger and slip_velocity > speed_trigger:
            if wheel.is_driven and self.accel > 0.4 and slip > 0.35:
                event_kind = "drive"
            elif self.brake > 0.35 and abs(long_v) > 0.5 and abs(slip) > 0.35:
                event_kind = "brake"
            elif abs(lat_v) > 0.8:
                event_kind = "slide"

        if wheel.is_grounded and event_kind is not None:
            contact = np.array([pos[0], ground_h + 0.02, pos[2]], dtype=float)
            norm_load = self.body.mass * 9.81 / 4.0
            load_scale = min(1.5, load / (norm_load + 1e-6)) if norm_load > 0 else 1.0
            slip_excess = max(0.0, slip_strength - slip_trigger)
            norm_excess = slip_excess / (1.0 - slip_trigger + 1e-6)
            mark_intensity = float(
                np.clip(norm_excess * (0.6 + 0.4 * load_scale), 0.0, 1.0)
            )
            event = {
                "index": index,
                "position": contact,
                "right": right.copy(),
                "forward": forward.copy(),
                "width": float(wheel.width),
                "intensity": mark_intensity,
                "kind": event_kind,
                "slip": float(slip),
                "lat_slip": float(alpha),
                "long_v": float(long_v),
                "load": float(load),
                "mu": float(base_mu),
                "base_color": self._surface_mark_color(base_mu),
                "driven": wheel.is_driven,
                "surface_height": float(ground_h),
            }
            self.slip_events.append(event)

        return 1

    def _engine_brake_torque(self, rpm):
        curve = self.engine_brake_curve
        if not curve:
            return 0.0
        if rpm <= curve[0][0]:
            return curve[0][1]
        if rpm >= curve[-1][0]:
            return curve[-1][1]
        for (r0, t0), (r1, t1) in zip(curve[:-1], curve[1:]):
            if r0 <= rpm <= r1:
                ratio = (rpm - r0) / (r1 - r0)
                return t0 + ratio * (t1 - t0)
        return curve[-1][1]

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
            fric_force = (
                fric_dir
                * n_force
                * 0.8
                * self.terrain.get_friction(pos[0], pos[2])
            )
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
        grounded = 0
        self.slip_events = []
        bottomed = False
        for i, w in enumerate(self.wheels):
            grounded += self._update_wheel(i, w, dt, g_front, g_rear)
            bottomed = bottomed or w.bottomed
        if self.is_upside_down or grounded < 2 or bottomed:
            self._handle_collisions()
        self.body.update(dt)
