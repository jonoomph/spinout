# render.py
import moderngl
import pygame
import numpy as np

class RenderContext:
    def __init__(self, width, height):
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.width = width
        self.height = height
        from shaders import create_shaders
        self.prog, self.prog2d = create_shaders(self.ctx)
        self.ortho = np.array([
            [2.0 / width, 0.0, 0.0, 0.0],
            [0.0, 2.0 / height, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0, 1.0]
        ], dtype='f4')
        hud_quad_data = np.array([
            0, 0, 0, 1,
            width, 0, 1, 1,
            0, height, 0, 0,
            width, 0, 1, 1,
            width, height, 1, 0,
            0, height, 0, 0,
        ], dtype='f4')
        self.hud_vbo = self.ctx.buffer(hud_quad_data.tobytes())
        self.hud_vao = self.ctx.vertex_array(self.prog2d, self.hud_vbo, 'in_pos', 'in_tex')

    def clear(self):
        self.ctx.clear(135/255, 206/255, 235/255, 1.0, depth=1.0)

    def render_terrain(self, terrain_vao, mvp):
        self.ctx.line_width = 1.0
        self.prog['mvp'].write(mvp.T.tobytes())
        terrain_vao.render(moderngl.LINES)

    def render_car(self, vertices, mvp):
        main_vertices, shock_vertices = vertices
        self.prog['mvp'].write(mvp.T.tobytes())
        if main_vertices:
            main_vbo = self.ctx.buffer(np.array(main_vertices, dtype='f4').tobytes())
            main_vao = self.ctx.vertex_array(self.prog, main_vbo, 'in_vert', 'in_color')
            self.ctx.line_width = 1.0
            main_vao.render(moderngl.LINES)
        if shock_vertices:
            shock_vbo = self.ctx.buffer(np.array(shock_vertices, dtype='f4').tobytes())
            shock_vao = self.ctx.vertex_array(self.prog, shock_vbo, 'in_vert', 'in_color')
            self.ctx.line_width = 3.0
            shock_vao.render(moderngl.LINES)

    def render_hud(self, hud_surf):
        hud_data = pygame.image.tostring(hud_surf, 'RGBA', False)
        hud_tex = self.ctx.texture((self.width, self.height), 4, hud_data)
        hud_tex.use(0)
        self.prog2d['mvp'].write(self.ortho.T.tobytes())
        self.prog2d['tex'] = 0
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.hud_vao.render(moderngl.TRIANGLES)
        self.ctx.enable(moderngl.DEPTH_TEST)