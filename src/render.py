# render.py
import moderngl
import pygame
import numpy as np

class RenderContext:
    def __init__(self, width, height):
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.blend_equation = moderngl.FUNC_ADD
        self.width = width
        self.height = height
        from src.shaders import create_shaders
        self.prog, self.prog2d, self.prog_lit = create_shaders(self.ctx)
        self.ortho = np.eye(4, dtype='f4')  # Identity for NDC
        self.mode = 0  # 0=wireframe,1=solid,2=solid+sun
        self.light_dir = np.array([0.5, 1.0, 0.3], dtype='f4')
        hud_quad_data = np.array([
            0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 1.0,
            1.0, 0.0, 1.0, 0.0,
            1.0, 1.0, 1.0, 1.0,
            0.0, 1.0, 0.0, 1.0,
        ], dtype='f4')
        self.hud_vbo = self.ctx.buffer(hud_quad_data.tobytes())
        self.hud_vao = self.ctx.vertex_array(self.prog2d, self.hud_vbo, 'in_pos', 'in_tex')
        self.hud_tex = None
        self.main_vbo = None
        self.shock_vbo = None
        self.main_vao = None
        self.shock_vao = None

        # Gradient skybox: light gray ground -> white horizon band -> blue sky
        bottom = [0.8, 0.8, 0.8, 1.0]
        horizon = [1.0, 1.0, 1.0, 1.0]
        sky = [135/255, 206/255, 235/255, 1.0]
        skybox_data = np.array([
            -1.0, -1.0, 0.0, *bottom,
            1.0, -1.0, 0.0, *bottom,
            -1.0, 0.0, 0.0, *horizon,
            -1.0, 0.0, 0.0, *horizon,
            1.0, -1.0, 0.0, *bottom,
            1.0, 0.0, 0.0, *horizon,
            -1.0, 0.0, 0.0, *horizon,
            1.0, 0.0, 0.0, *horizon,
            -1.0, 1.0, 0.0, *sky,
            -1.0, 1.0, 0.0, *sky,
            1.0, 0.0, 0.0, *horizon,
            1.0, 1.0, 0.0, *sky,
        ], dtype='f4')
        self.sky_vbo = self.ctx.buffer(skybox_data.tobytes())
        self.sky_vao = self.ctx.vertex_array(self.prog, self.sky_vbo, 'in_vert', 'in_color')

    def set_mode(self, mode):
        self.mode = mode
        self.ctx.wireframe = mode == 0

    def clear(self):
        self.ctx.viewport = (0, 0, self.width, self.height)
        self.ctx.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)
        self.render_skybox()

    def render_terrain(self, terrain_vao, mvp):
        self.ctx.line_width = 1.0
        prog = terrain_vao.program
        prog['mvp'].write(mvp.T.tobytes())
        if 'light_dir' in prog:
            prog['light_dir'].value = tuple(self.light_dir)
        terrain_vao.render(moderngl.TRIANGLES)

    def render_car(self, vertices, mvp):
        main_vertices, shock_vertices = vertices
        self.prog['mvp'].write(mvp.T.tobytes())
        if main_vertices:
            main_data = np.array(main_vertices, dtype='f4').tobytes()
            if self.main_vbo is None or len(main_data) > self.main_vbo.size:
                if self.main_vbo:
                    self.main_vbo.release()
                self.main_vbo = self.ctx.buffer(main_data)
                self.main_vao = self.ctx.vertex_array(self.prog, self.main_vbo, 'in_vert', 'in_color')
            else:
                self.main_vbo.write(main_data, offset=0)
            self.ctx.line_width = 1.0
            self.main_vao.render(moderngl.LINES)
        if shock_vertices:
            shock_data = np.array(shock_vertices, dtype='f4').tobytes()
            if self.shock_vbo is None or len(shock_data) > self.shock_vbo.size:
                if self.shock_vbo:
                    self.shock_vbo.release()
                self.shock_vbo = self.ctx.buffer(shock_data)
                self.shock_vao = self.ctx.vertex_array(self.prog, self.shock_vbo, 'in_vert', 'in_color')
            else:
                self.shock_vbo.write(shock_data, offset=0)
            self.ctx.line_width = 3.0
            self.shock_vao.render(moderngl.LINES)

    def render_skybox(self):
        # Always render the skybox filled even if the main scene uses wireframe
        was_wireframe = self.ctx.wireframe
        self.ctx.wireframe = False
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.prog['mvp'].write(self.ortho.tobytes())
        self.sky_vao.render(moderngl.TRIANGLES)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.wireframe = was_wireframe

    def render_hud(self, hud_surf):
        self.ctx.viewport = (0, 0, self.width, self.height)
        hud_data = pygame.image.tostring(hud_surf, 'RGBA', True)
        if self.hud_tex is None:
            self.hud_tex = self.ctx.texture((self.width, self.height), 4, hud_data)
            self.hud_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        else:
            self.hud_tex.write(hud_data)
        was_wireframe = self.ctx.wireframe
        self.ctx.wireframe = False
        self.hud_tex.use(0)
        self.prog2d['tex'] = 0
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.hud_vao.render(moderngl.TRIANGLES)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.wireframe = was_wireframe
