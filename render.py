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
        from shaders import create_shaders
        self.prog, self.prog2d = create_shaders(self.ctx)
        self.ortho = np.eye(4, dtype='f4')  # Identity for NDC
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
        self.main_vbo = None
        self.shock_vbo = None
        self.main_vao = None
        self.shock_vao = None

    def clear(self):
        self.ctx.viewport = (0, 0, self.width, self.height)
        self.ctx.clear(135/255, 206/255, 235/255, 1.0, depth=1.0)

    def render_terrain(self, terrain_vao, mvp):
        self.ctx.line_width = 1.0
        self.prog['mvp'].write(mvp.T.tobytes())
        terrain_vao.render(moderngl.LINES)

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

    def render_hud(self, hud_surf):
        self.ctx.viewport = (0, 0, self.width, self.height)
        hud_data = pygame.image.tostring(hud_surf, 'RGBA', True)
        hud_tex = self.ctx.texture((self.width, self.height), 4, hud_data)
        hud_tex.use(0)
        self.prog2d['tex'] = 0
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.hud_vao.render(moderngl.TRIANGLES)
        self.ctx.enable(moderngl.DEPTH_TEST)
        hud_tex.release()