# render.py
import moderngl
import pygame
import numpy as np
from .colors import (
    FOG_DEFAULT_COLOR,
    FOG_DUST_COLOR,
    SKY_BOTTOM_COLOR,
    SKY_HORIZON_COLOR,
    SKY_TOP_COLOR,
    SUN_LIGHT_COLOR,
)

class RenderContext:
    def __init__(self, width, height):
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND | moderngl.PROGRAM_POINT_SIZE)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.blend_equation = moderngl.FUNC_ADD
        self.width = width
        self.height = height
        from src.shaders import create_shaders
        self.prog, self.prog2d, self.prog_lit, self.prog_tex = create_shaders(self.ctx)
        self.ortho = np.eye(4, dtype='f4')  # Identity for NDC
        self.mode = 1  # 0=wireframe,1=textured
        self.light_dir = np.array([0.5, 1.0, 0.3], dtype='f4')
        self.light_color = np.array(SUN_LIGHT_COLOR, dtype='f4')
        self.camera_pos = np.zeros(3, dtype='f4')
        self.fog_density = 0.0
        self.fog_color = np.array(FOG_DEFAULT_COLOR, dtype='f4')
        self.wetness = 0.0
        self.road_noise = 0.0
        self.terrain_mode = 0
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
        self.model_vbo = None
        self.model_vao = None
        self.model_edge_vbo = None
        self.model_edge_vao = None
        self.car_model_tex = None

        # Gradient skybox: light gray ground -> white horizon band -> blue sky
        bottom = list(SKY_BOTTOM_COLOR)
        horizon = list(SKY_HORIZON_COLOR)
        sky = list(SKY_TOP_COLOR)
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

    def set_camera(self, pos):
        self.camera_pos = np.array(pos, dtype='f4')

    def setup_weather(self, weather, terrain_type, road_type):
        self.wetness = 1.0 if weather == 'wet' else 0.0
        self.fog_density = 0.0
        self.fog_color = np.array(FOG_DEFAULT_COLOR, dtype='f4')
        if self.wetness > 0.0:
            self.fog_density = 0.015
        elif terrain_type in ('sand', 'gravel', 'dirt'):
            self.fog_density = 0.001
            self.fog_color = np.array(FOG_DUST_COLOR, dtype='f4')

        terrain_map = {
            'grass': 1,
            'dirt': 2,
            'sand': 2,
            'snow': 3,
        }
        self.terrain_mode = terrain_map.get(terrain_type, 0)

        # subtle surface texture for roads
        noise_map = {
            'asphalt': 2.5,   # larger scale noise for visible motion
            'concrete': 1.5 , # negative selects groove pattern with wide spacing
            'gravel': 3.5,    # coarse noise for gravel
        }
        self.road_noise = noise_map.get(road_type, 0.0)

    def render_weather(self, mvp, dt):
        pass

    def set_mode(self, mode):
        self.mode = mode
        self.ctx.wireframe = mode == 0

    def clear(self):
        self.ctx.viewport = (0, 0, self.width, self.height)
        self.ctx.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)
        self.render_skybox()

    def _apply_common_uniforms(self, prog):
        if 'cam_pos' in prog:
            prog['cam_pos'].value = tuple(self.camera_pos)
        if 'fog_density' in prog:
            prog['fog_density'].value = self.fog_density
        if 'fog_color' in prog:
            prog['fog_color'].value = tuple(self.fog_color)
        if 'wetness' in prog:
            prog['wetness'].value = self.wetness
        if 'light_color' in prog:
            prog['light_color'].value = tuple(self.light_color)

    def render_terrain(self, terrain_vao, mvp, noise_scale=0.0, terrain_mode=None):
        self.ctx.line_width = 1.0
        prog = terrain_vao.program
        prog['mvp'].write(mvp.T.tobytes())
        self._apply_common_uniforms(prog)
        if 'noise_scale' in prog:
            prog['noise_scale'].value = noise_scale
        if 'terrain_mode' in prog:
            prog['terrain_mode'].value = self.terrain_mode if terrain_mode is None else terrain_mode
        if 'light_dir' in prog:
            prog['light_dir'].value = tuple(self.light_dir)
        terrain_vao.render(moderngl.TRIANGLES)

    def cycle_terrain_mode(self):
        self.terrain_mode = (self.terrain_mode + 1) % 4

    def render_signs(self, vao, mvp):
        """Render simple colored billboards such as sign posts."""
        was_wireframe = self.ctx.wireframe
        self.ctx.wireframe = False
        prog = vao.program
        prog['mvp'].write(mvp.T.tobytes())
        self._apply_common_uniforms(prog)
        vao.render(moderngl.TRIANGLES)
        self.ctx.wireframe = was_wireframe

    def render_billboard(self, vao, tex, mvp):
        """Render a textured billboard (e.g. the speed limit sign face)."""
        was_wireframe = self.ctx.wireframe
        self.ctx.wireframe = False
        prog = vao.program
        prog['mvp'].write(mvp.T.tobytes())
        self._apply_common_uniforms(prog)
        tex.use(0)
        prog['tex'] = 0
        vao.render(moderngl.TRIANGLES)
        self.ctx.wireframe = was_wireframe

    def render_car(self, vertices, mvp):
        main_vertices, shock_vertices = vertices
        self.prog['mvp'].write(mvp.T.tobytes())
        self._apply_common_uniforms(self.prog)
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

    def render_car_model(self, vertices, mvp):
        """Draw either wireframe or textured model using explicit VAO layouts."""
        tri_vertices, edge_vertices = vertices

        if self.ctx.wireframe:
            if not edge_vertices:
                return
            data = np.asarray(edge_vertices, 'f4').tobytes()
            if self.model_edge_vbo is None or len(data) > self.model_edge_vbo.size:
                if self.model_edge_vbo:
                    self.model_edge_vbo.release()
                self.model_edge_vbo = self.ctx.buffer(data)
                # 3 floats pos + 4 floats color
                self.model_edge_vao = self.ctx.vertex_array(
                    self.prog,
                    [(self.model_edge_vbo, '3f 4f', 'in_vert', 'in_color')],
                )
            else:
                self.model_edge_vbo.write(data, 0)
            self.prog['mvp'].write(mvp.T.tobytes())
            self._apply_common_uniforms(self.prog)
            self.ctx.line_width = 1.0
            self.model_edge_vao.render(moderngl.LINES)
            return

        if not tri_vertices:
            return
        data = np.asarray(tri_vertices, 'f4').tobytes()
        if self.model_vbo is None or len(data) > self.model_vbo.size:
            if self.model_vbo:
                self.model_vbo.release()
            self.model_vbo = self.ctx.buffer(data)
            # 3 floats pos + 2 floats UV
            self.model_vao = self.ctx.vertex_array(
                self.prog_tex,
                [(self.model_vbo, '3f 2f', 'in_vert', 'in_tex')],
            )
        else:
            self.model_vbo.write(data, 0)

        self.prog_tex['mvp'].write(mvp.T.tobytes())
        self._apply_common_uniforms(self.prog_tex)
        if self.car_model_tex:
            self.car_model_tex.use(0)
            self.prog_tex['tex'] = 0
        self.model_vao.render(moderngl.TRIANGLES)

    def render_skybox(self):
        # Always render the skybox filled even if the main scene uses wireframe
        was_wireframe = self.ctx.wireframe
        self.ctx.wireframe = False
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.prog['mvp'].write(self.ortho.tobytes())
        if 'cam_pos' in self.prog:
            self.prog['cam_pos'].value = (0.0, 0.0, 0.0)
        if 'fog_density' in self.prog:
            self.prog['fog_density'].value = 0.0
        if 'fog_color' in self.prog:
            self.prog['fog_color'].value = (0.8, 0.8, 0.8)
        if 'wetness' in self.prog:
            self.prog['wetness'].value = 0.0
        if 'noise_scale' in self.prog:
            self.prog['noise_scale'].value = 0.0
        if 'terrain_mode' in self.prog:
            self.prog['terrain_mode'].value = 0
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
