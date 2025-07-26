############################
# controls.py
############################
import glfw

class Controls:
    def __init__(self, window):
        self.steer = 0.0
        self.throttle = 0.0
        self.brake = False
        self.zoom = 1.0
        # register GLFW callbacks
        glfw.set_key_callback(window, self.key_callback)
        glfw.set_scroll_callback(window, self.scroll_callback)

    def key_callback(self, window, key, scancode, action, mods):
        if action in (glfw.PRESS, glfw.REPEAT):
            if key == glfw.KEY_LEFT:
                self.steer = -1.0
            elif key == glfw.KEY_RIGHT:
                self.steer = 1.0
            elif key == glfw.KEY_UP:
                self.throttle = 1.0
            elif key == glfw.KEY_DOWN:
                self.brake = True
        elif action == glfw.RELEASE:
            if key in (glfw.KEY_LEFT, glfw.KEY_RIGHT):
                self.steer = 0.0
            elif key == glfw.KEY_UP:
                self.throttle = 0.0
            elif key == glfw.KEY_DOWN:
                self.brake = False

    def scroll_callback(self, window, xoffset, yoffset):
        self.zoom *= (1 + yoffset * 0.1)