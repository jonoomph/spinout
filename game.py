import os
import math
import time
import numpy as np
import pybullet as p
import pybullet_data

# ----------------------------
# Config
# ----------------------------
SEED = None  # set an int for repeatable terrain

# Physics
PHYSICS_HZ = 240
GRAVITY = -9.81

# Terrain scale:
# N x N samples, each cell size SX,SY meters, vertical scale SZ meters
# Keep cells smaller than the car footprint, but world big enough to drive ~30 s.
N = 512
SX, SY, SZ = 1.5, 1.5, 6.0  # ~766 m square area, ±12 m hills

# Car
CAR_SCALE = 2.4
MAX_STEER = math.radians(27)
STEER_RATE = math.radians(140)    # rad/s
THROTTLE_ACCEL = 2.0              # per second
THROTTLE_DECAY = 1.6
MAX_WHEEL_RAD_PER_SEC = 85.0
WHEEL_FORCE = 80.0
BRAKE_FORCE = 180.0

# Camera (debug visualizer)
CAM_DIST = 12.0
CAM_PITCH = -18.0  # degrees
CAM_LOOK_AHEAD = 2.0  # meters

# ----------------------------
# Noise helpers
# ----------------------------
def value_noise_fbm(n, octaves=(8, 16, 32, 64, 128), weights=None, seed=None):
    if weights is None:
        weights = [1.0, 0.6, 0.35, 0.2, 0.1][:len(octaves)]
    rng = np.random.default_rng(seed)
    h = np.zeros((n, n), dtype=np.float32)
    for s, w in zip(octaves, weights):
        g = max(2, n // s)
        control = rng.random((g, g), dtype=np.float32) * 2.0 - 1.0  # [-1,1]

        # Bilinear upsample
        y = np.linspace(0, g - 1, n, dtype=np.float32)
        x = np.linspace(0, g - 1, n, dtype=np.float32)
        yy, xx = np.meshgrid(y, x, indexing='ij')
        x0 = np.floor(xx).astype(np.int32)
        y0 = np.floor(yy).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, g - 1)
        y1 = np.clip(y0 + 1, 0, g - 1)
        sx = xx - x0
        sy = yy - y0
        sx2 = sx * sx * (3 - 2 * sx)
        sy2 = sy * sy * (3 - 2 * sy)
        c00 = control[y0, x0]; c10 = control[y0, x1]
        c01 = control[y1, x0]; c11 = control[y1, x1]
        i1 = c00 * (1 - sx2) + c10 * sx2
        i2 = c01 * (1 - sx2) + c11 * sx2
        h += w * (i1 * (1 - sy2) + i2 * sy2)

    m = np.max(np.abs(h)) + 1e-6
    h /= m
    return h.astype(np.float32)

def make_heightfield(n=N, sx=SX, sy=SY, sz=SZ, seed=SEED):
    base = value_noise_fbm(n, seed=seed)
    large = value_noise_fbm(n, octaves=(64, 128, 256), weights=[1.0, 0.5, 0.25],
                            seed=None if seed is None else seed + 1)
    h = 0.8 * base + 0.6 * large
    h = (h + 0.12).clip(-1.0, 1.0)  # fewer deep pits
    return h, (sx, sy, sz)

def height_at_world_xy(hmap, sx, sy, sz, world_xy):
    n = hmap.shape[0]
    size_x = (n - 1) * sx
    size_y = (n - 1) * sy
    xw, yw = world_xy
    u = (xw + size_x * 0.5) / sx
    v = (yw + size_y * 0.5) / sy
    i = int(np.clip(round(v), 0, n - 1))
    j = int(np.clip(round(u), 0, n - 1))
    return float(hmap[i, j] * sz)

def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def quat_to_forward(q):
    r = p.getMatrixFromQuaternion(q)
    return np.array([r[0], r[3], r[6]], dtype=np.float32)

# ----------------------------
# World setup
# ----------------------------
def create_world():
    cid = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, GRAVITY)
    p.setTimeStep(1.0 / PHYSICS_HZ)
    p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
    return cid

def create_terrain():
    hmap, scale = make_heightfield()
    sx, sy, sz = scale
    n = hmap.shape[0]

    # Heightfield lives in the collision shape; visual is applied via changeVisualShape.
    flags = p.GEOM_CONCAVE_INTERNAL_EDGE
    hdata = hmap.flatten(order="C")

    col = p.createCollisionShape(
        shapeType=p.GEOM_HEIGHTFIELD,
        meshScale=[sx, sy, sz],
        heightfieldData=hdata,
        numHeightfieldRows=n,
        numHeightfieldColumns=n,
        heightfieldTextureScaling=1.0,  # OK here; remove if your build complains
        flags=flags,
    )

    terrain = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=-1,  # no separate visual shape for heightfield
    )

    # Color the terrain (simple flat color)
    p.changeVisualShape(terrain, -1, rgbaColor=[0.55, 0.75, 0.55, 1.0])

    # Center the heightfield around world origin
    size_x = (n - 1) * sx
    size_y = (n - 1) * sy
    p.resetBasePositionAndOrientation(
        terrain,
        [-size_x * 0.5, -size_y * 0.5, 0.0],
        [0, 0, 0, 1]
    )

    # Friction tuned for drivable hills
    p.changeDynamics(terrain, -1, lateralFriction=1.0, rollingFriction=0.02, spinningFriction=0.02)

    return terrain, hmap, scale


def load_racecar(start_pos):
    car = p.loadURDF(
        os.path.join(pybullet_data.getDataPath(), "racecar/racecar.urdf"),
        start_pos,
        p.getQuaternionFromEuler([0, 0, 0]),
        globalScaling=CAR_SCALE,
    )

    # Disable default motors
    for j in range(p.getNumJoints(car)):
        p.setJointMotorControl2(car, j, p.VELOCITY_CONTROL, force=0)

    steering_joints, wheel_joints = [], []
    for j in range(p.getNumJoints(car)):
        name = p.getJointInfo(car, j)[1].decode("utf-8").lower()
        if "steer" in name:
            steering_joints.append(j)
        elif "wheel" in name:
            wheel_joints.append(j)

    if not steering_joints:
        steering_joints = [j for j in (4, 6) if j < p.getNumJoints(car)]
    if not wheel_joints:
        wheel_joints = [j for j in (2, 3, 5, 7) if j < p.getNumJoints(car)]

    for j in wheel_joints:
        p.changeDynamics(car, j, lateralFriction=1.2, rollingFriction=0.03, spinningFriction=0.02)

    return car, steering_joints, wheel_joints

# ----------------------------
# Controls and loop
# ----------------------------
def main():
    create_world()
    terrain, hmap, (sx, sy, sz) = create_terrain()

    # find true center of terrain
    aabb_min, aabb_max = p.getAABB(terrain)
    center_x = 0.5 * (aabb_min[0] + aabb_max[0])
    center_y = 0.5 * (aabb_min[1] + aabb_max[1])

    # ray‑cast down to get ground height
    def get_ground_z(x, y):
        ray_start = [x, y, sz * 1.5]
        ray_end   = [x, y, -sz * 1.5]
        hit = p.rayTest(ray_start, ray_end)[0]
        return hit[3][2] if hit[0] != -1 else 0.0

    # spawn the car just above terrain center
    ground_z = get_ground_z(center_x, center_y)
    z0 = ground_z + CAR_SCALE
    car, steer_joints, wheel_joints = load_racecar([center_x, center_y, z0])

    steer_angle = 0.0
    throttle    = 0.0
    brake       = 0.0
    cam_dist    = CAM_DIST

    p.setRealTimeSimulation(0)
    last = time.perf_counter()

    while p.isConnected():
        now = time.perf_counter()
        dt  = now - last
        last = now

        keys = p.getKeyboardEvents()
        left  = (p.B3G_LEFT_ARROW  in keys and keys[p.B3G_LEFT_ARROW]  & p.KEY_IS_DOWN) or \
                (ord('a') in keys and keys[ord('a')] & p.KEY_IS_DOWN)
        right = (p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN) or \
                (ord('d') in keys and keys[ord('d')] & p.KEY_IS_DOWN)
        up    = (p.B3G_UP_ARROW    in keys and keys[p.B3G_UP_ARROW]    & p.KEY_IS_DOWN) or \
                (ord('w') in keys and keys[ord('w')] & p.KEY_IS_DOWN)
        down  = (p.B3G_DOWN_ARROW  in keys and keys[p.B3G_DOWN_ARROW]  & p.KEY_IS_DOWN) or \
                (ord('s') in keys and keys[ord('s')] & p.KEY_IS_DOWN)
        space = (ord(' ') in keys and keys[ord(' ')] & p.KEY_IS_DOWN)
        reset = (ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED)

        # handle reset
        if reset:
            ground_z = get_ground_z(center_x, center_y)
            z0 = ground_z + CAR_SCALE
            p.resetBasePositionAndOrientation(car, [center_x, center_y, z0], [0, 0, 0, 1])
            p.resetBaseVelocity(car, [0, 0, 0], [0, 0, 0])
            steer_angle = throttle = brake = 0.0

        # steering input (fixed inversion)
        steer_input = (1.0 if left else 0.0) + (-1.0 if right else 0.0)
        if steer_input:
            steer_angle += steer_input * STEER_RATE * dt
        else:
            if steer_angle > 0:
                steer_angle = max(0.0, steer_angle - STEER_RATE * 0.8 * dt)
            elif steer_angle < 0:
                steer_angle = min(0.0, steer_angle + STEER_RATE * 0.8 * dt)
        steer_angle = clamp(steer_angle, -MAX_STEER, MAX_STEER)

        # throttle/brake
        if up:
            throttle = clamp(throttle + THROTTLE_ACCEL * dt, -1.0, 1.0)
        elif down:
            throttle = clamp(throttle - THROTTLE_ACCEL * dt, -1.0, 1.0)
        else:
            if throttle > 0:
                throttle = max(0.0, throttle - THROTTLE_DECAY * dt)
            elif throttle < 0:
                throttle = min(0.0, throttle + THROTTLE_DECAY * dt)
        brake = 1.0 if space else 0.0

        # apply steering
        for j in steer_joints:
            p.setJointMotorControl2(car, j, p.POSITION_CONTROL,
                                    targetPosition=steer_angle, force=150.0)

        # apply drive or brake
        if brake:
            for j in wheel_joints:
                p.setJointMotorControl2(car, j, p.VELOCITY_CONTROL,
                                        targetVelocity=0.0, force=BRAKE_FORCE)
        else:
            target_vel = throttle * MAX_WHEEL_RAD_PER_SEC
            for j in wheel_joints:
                p.setJointMotorControl2(car, j, p.VELOCITY_CONTROL,
                                        targetVelocity=target_vel, force=WHEEL_FORCE)

        # step physics
        for _ in range(max(1, int(dt * PHYSICS_HZ))):
            p.stepSimulation()

        # camera follow, raised above hills
        pos, orn = p.getBasePositionAndOrientation(car)
        fwd = quat_to_forward(orn)
        # compute yaw so camera sits directly behind the car
        cam_yaw = -math.degrees(math.atan2(fwd[1], fwd[0]))
        cam_target = [
            pos[0] + fwd[0] * CAM_LOOK_AHEAD,
            pos[1] + fwd[1] * CAM_LOOK_AHEAD,
            pos[2] + 2.0  # lift target 2m above car center to avoid clipping
        ]
        p.resetDebugVisualizerCamera(cam_dist, cam_yaw, CAM_PITCH, cam_target)

        # throttle CPU
        time.sleep(max(0.0, 1.0 / PHYSICS_HZ - (time.perf_counter() - now)))


if __name__ == "__main__":
    main()
