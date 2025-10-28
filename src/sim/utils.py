# utils.py
import math
import numpy as np


def compute_mvp(width, height, camera_pos, camera_right, camera_forward, camera_up):
    r = np.array(camera_right, dtype="f4")
    u = np.array(camera_up, dtype="f4")
    f = np.array(camera_forward, dtype="f4")

    def _safe_norm(vec):
        n = np.linalg.norm(vec)
        return n if n > 1e-8 else 1.0

    r /= _safe_norm(r)
    u /= _safe_norm(u)
    f /= _safe_norm(f)

    view = np.eye(4, dtype="f4")
    view[0, :3] = r
    view[1, :3] = u
    view[2, :3] = -f
    cam = np.array(camera_pos, dtype="f4")
    view[0, 3] = -np.dot(r, cam)
    view[1, 3] = -np.dot(u, cam)
    view[2, 3] = np.dot(f, cam)

    fov_rad = math.radians(60)
    aspect = width / height
    near, far = 0.1, 200.0
    proj = np.zeros((4,4), dtype='f4')
    proj[0,0] = 1 / (aspect * math.tan(fov_rad / 2))
    proj[1,1] = 1 / math.tan(fov_rad / 2)
    proj[2,2] = -(far + near) / (far - near)
    proj[2,3] = -2 * far * near / (far - near)
    proj[3,2] = -1

    return proj @ view @ np.eye(4, dtype="f4")
