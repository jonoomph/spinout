############################
# game.py
############################
import mujoco
import mujoco_viewer
import numpy as np
from noise import pnoise2
from PIL import Image
from controls import Controls

# Generate and save a heightmap image
def generate_terrain(filename, size=256, scale=0.1):
    data = np.zeros((size, size), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            n_val = pnoise2(i * scale, j * scale)
            v = (n_val + 1) * 0.5
            data[j, i] = int(np.clip(v * 255, 0, 255))
    img = Image.fromarray(data)
    img.save(filename)


def main():
    xml_path = 'car.xml'
    terrain_file = 'terrain.png'
    generate_terrain(terrain_file)

    # Load model
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    # Find the body ID for the 'car' body and set chase camera
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'car')
    viewer.cam.trackbodyid = body_id

    controls = Controls(viewer.window)

    while viewer.is_alive:
        # forward/backward motor: throttle=1->forward, brake->backward
        forward_val = controls.throttle - (1.0 if controls.brake else 0.0)
        steer_val = controls.steer
        data.ctrl[0] = forward_val
        data.ctrl[1] = steer_val

        # camera zoom
        viewer.cam.distance = 4.0 * controls.zoom

        mujoco.mj_step(model, data)
        viewer.render()

    viewer.close()

if __name__ == '__main__':
    main()
