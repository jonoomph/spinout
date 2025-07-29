# utils.py
import math
import numpy as np

def compute_mvp(width, height, camera_pos, camera_right, camera_forward, camera_up):
    basis = np.column_stack((camera_right, camera_up, camera_forward))
    view_rot = np.eye(4, dtype='f4')
    view_rot[:3, :3] = basis.T
    trans = np.eye(4, dtype='f4')
    trans[:3, 3] = -camera_pos
    view = view_rot @ trans

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
