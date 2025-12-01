# render.py
import math
import moderngl
import pygame
import numpy as np
from .colors import (
    FOG_DEFAULT_COLOR,
    FOG_DUST_COLOR,
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
        from .shaders import create_shaders
        (
            self.prog,
            self.prog2d,
            self.prog_lit,
            self.prog_tex,
            self.prog_rain,
            self.prog_puddle,
            self.prog_fog_sheet,
            self.prog_sky,
        ) = create_shaders(self.ctx)
        if 'point_size' in self.prog:
            self.prog['point_size'].value = 1.0
        self.ortho = np.eye(4, dtype='f4')  # Identity for NDC
        self.mode = 1  # 0=wireframe,1=textured
        self.light_dir = np.array([0.5, 1.0, 0.3], dtype='f4')
        self.light_color = np.array(SUN_LIGHT_COLOR, dtype='f4')
        self.camera_pos = np.zeros(3, dtype='f4')
        self.camera_forward = np.array([0.0, 0.0, -1.0], dtype='f4')
        self.camera_right = np.array([1.0, 0.0, 0.0], dtype='f4')
        self.camera_up = np.array([0.0, 1.0, 0.0], dtype='f4')
        self._last_camera_pos = np.zeros(3, dtype='f4')
        self.camera_velocity = np.zeros(3, dtype='f4')
        self.fog_density = 0.0
        self.fog_color = np.array(FOG_DEFAULT_COLOR, dtype='f4')
        self.wetness = 0.0
        self.precipitation = "none"
        self.rain_intensity = 0.0
        self.road_noise = 0.0
        self.terrain_mode = 0
        self._rng = np.random.default_rng()
        self.rain_spawn_radius = 24.0
        self.rain_spawn_height = 20.0
        self.max_rain_drops = 2600
        self.rain_count = 0
        self._rain_time = 0.0
        self.rain_positions = np.zeros((self.max_rain_drops, 3), dtype='f4')
        self.rain_velocities = np.zeros((self.max_rain_drops, 3), dtype='f4')
        self.rain_lengths = np.zeros(self.max_rain_drops, dtype='f4')
        self._rain_vertices = np.zeros((self.max_rain_drops * 2, 7), dtype='f4')
        self.rain_vbo = self.ctx.buffer(self._rain_vertices.tobytes())
        self.rain_vao = self.ctx.vertex_array(
            self.prog_rain,
            self.rain_vbo,
            'in_vert',
            'in_color',
        )
        if 'fade_distance' in self.prog_rain:
            self.prog_rain['fade_distance'].value = 70.0
        self._rain_vertex_count = 0
        self.terrain_ref = None
        self.puddle_vbo = None
        self.puddle_vao = None
        self.puddle_vertices = None
        self.puddle_strength = 0.0
        self._rain_anchor = np.zeros(3, dtype='f4')
        self._rain_anchor_valid = False
        self.fog_vbo = None
        self.fog_vao = None
        self._fog_vertices = None
        self.fog_sheet_count = 0
        self._fog_sheet_offsets = None
        self._fog_sheet_radius = None
        self._fog_sheet_height = None
        self._fog_sheet_density = None
        self._fog_sheet_seed = None
        self._fog_center = self.camera_pos.copy()
        self._fog_scroll = np.zeros(2, dtype='f4')
        self._fog_field_clock = 0.0
        # Sun/orbit state
        self.sun_phase = 0.5  # 0..1 → 24h clock
        self.sun_time_hours = 12.0
        self.sun_east = np.array([1.0, 0.0, 0.0], dtype="f4")
        self.sun_north = np.array([0.0, 0.0, 1.0], dtype="f4")
        self.sun_up = np.array([0.0, 1.0, 0.0], dtype="f4")
        self.compass_north = self.sun_north.copy()
        self.compass_east = np.array([1.0, 0.0, 0.0], dtype="f4")
        self.scene_top_cardinal = "N"
        self.sun_cardinal = "N"
        self._sky_overcast = 0.3
        self._sky_turbidity = 2.0
        self._sky_day_fac = 1.0
        self.sun_radius = 400.0
        self.sun_world = np.zeros(3, dtype="f4")
        self.debug_vbo = None
        self.debug_vao = None
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
        self.hud_tex_size = None
        self.main_vbo = None
        self.shock_vbo = None
        self.main_vao = None
        self.shock_vao = None
        self.model_vbo = None
        self.model_vao = None
        self.model_edge_vbo = None
        self.model_edge_vao = None
        self.car_model_tex = None
        self.skid_vbo = None
        self.skid_vao = None

        # sky geometry-
        self._init_sky_mesh()
        self.sun_vbo = None
        self.sun_vao = None
        # headlight state
        self.headlight_pos = np.zeros((2, 3), dtype="f4")
        self.headlight_dir = np.array([0.0, 0.0, 1.0], dtype="f4")
        self.headlight_intensity = 0.0
        self.headlight_range = 0.0
        self._current_mvp = None
        self._sample_sun_state()
        self._init_fog_sheets()
        # Add this at the end of __init__ (after self._init_sky_mesh() and bef=====================ore self.update_view())
        aspect = self.width / self.height if self.height else 1.0
        fov = math.radians(60.0)
        near, far = 0.1, 10000.0
        tan_half_fov = math.tan(fov / 2.0)
        self.projection = np.array([
            [1.0 / (aspect * tan_half_fov), 0.0, 0.0, 0.0],
            [0.0, 1.0 / tan_half_fov, 0.0, 0.0],
            [0.0, 0.0, -(far + near) / (far - near), -1.0],
            [0.0, 0.0, -(2.0 * far * near) / (far - near), 0.0],
        ], dtype='f4').T  # transpose to column-major for OpenGL
        self.update_view()  # initial view

    def _init_sky_mesh(self):
        # Generate a simple sphere for the sky
        def generate_sphere(radius=1000.0, slices=32, stacks=16):
            verts = []
            for i in range(stacks + 1):
                theta = math.pi * i / stacks
                sin_theta = math.sin(theta)
                cos_theta = math.cos(theta)
                for j in range(slices):
                    phi = 2 * math.pi * j / slices
                    sin_phi = math.sin(phi)
                    cos_phi = math.cos(phi)
                    x = cos_phi * sin_theta
                    y = cos_theta
                    z = sin_phi * sin_theta
                    verts.append((x * radius, y * radius, z * radius))
            indices = []
            for i in range(stacks):
                for j in range(slices):
                    first = (i * slices) + j
                    second = first + slices
                    indices.extend([first, second, first + 1])
                    indices.extend([second, second + 1, first + 1])
            return np.array(verts, dtype='f4'), np.array(indices, dtype='i4')

        sky_verts, sky_inds = generate_sphere()
        self.sky_vbo = self.ctx.buffer(sky_verts.tobytes())
        self.sky_ibo = self.ctx.buffer(sky_inds.tobytes())
        self.sky_vao = self.ctx.vertex_array(self.prog_sky, self.sky_vbo, 'in_vert', index_buffer=self.sky_ibo)

    def update_view(self):
        self.view = np.eye(4, dtype='f4')
        self.view[:3, 0] = self.camera_right
        self.view[:3, 1] = self.camera_up
        self.view[:3, 2] = -self.camera_forward
        self.view[:3, 3] = -np.dot(self.camera_right, self.camera_pos), -np.dot(self.camera_up, self.camera_pos), np.dot(self.camera_forward, self.camera_pos)
        self.view_rot = self.view.copy()
        self.view_rot[:3, 3] = 0.0

    def set_projection(self, projection):
        self.projection = projection

    def _rotate_vec(self, v, axis, angle):
        axis = np.array(axis, dtype="f4")
        axis /= max(np.linalg.norm(axis), 1e-6)
        v = np.array(v, dtype="f4")
        c = math.cos(angle)
        s = math.sin(angle)
        return (
            v * c
            + np.cross(axis, v) * s
            + axis * np.dot(axis, v) * (1.0 - c)
        ).astype("f4")

    def _sample_day_phase(self):
        """
        Return a day-biased phase in [0,1) where 0.25=~6am, 0.5=noon, 0.75=6pm.
        Avoid deep night (11pm-4am) and weight toward daylight.
        """
        # Buckets around sunrise, morning, noon, afternoon, evening, late-evening
        buckets = np.array([0.25, 0.35, 0.50, 0.65, 0.75, 0.88], dtype="f4")
        stds = np.array([0.020, 0.035, 0.030, 0.035, 0.030, 0.025], dtype="f4")
        weights = np.array([1.1, 1.2, 1.4, 1.2, 1.0, 0.6], dtype="f4")
        weights /= weights.sum()
        i = int(self._rng.choice(len(buckets), p=weights))
        t = float(self._rng.normal(buckets[i], stds[i]))
        # Clip out hard night: ignore 11pm-4am
        t = float(np.clip(t, 4.0 / 24.0, 23.0 / 24.0))
        return t % 1.0

    def _sample_sun_basis(self):
        """
        Pick a random orientation for the sun arc: random compass yaw plus small seasonal tilt.
        """
        yaw = float(self._rng.uniform(0.0, 2.0 * np.pi))
        tilt = float(self._rng.uniform(-math.radians(12.0), math.radians(12.0)))
        east = np.array([math.cos(yaw), 0.0, -math.sin(yaw)], dtype="f4")
        north = np.array([math.sin(yaw), 0.0, math.cos(yaw)], dtype="f4")
        up = np.array([0.0, 1.0, 0.0], dtype="f4")
        if abs(tilt) > 1e-5:
            up = self._rotate_vec(up, east, tilt)
            north = self._rotate_vec(north, east, tilt)
        # Orthonormalize to be safe
        east /= max(np.linalg.norm(east), 1e-6)
        north = north - np.dot(north, east) * east
        north /= max(np.linalg.norm(north), 1e-6)
        up = np.cross(north, east)
        up /= max(np.linalg.norm(up), 1e-6)
        self.sun_east, self.sun_north, self.sun_up = east, north, up
        # Use same random yaw as the world compass orientation
        self.compass_east = east.copy()
        self.compass_north = north.copy()

    def _phase_to_angle(self, phase):
        # phase 0.25 = sunrise east horizon, 0.5 = noon overhead, 0.75 = sunset west horizon
        return (phase - 0.25) * 2.0 * np.pi

    def _phase_to_dir(self, phase):
        sunrise_phase = 6.0 / 24.0
        sunset_phase = 18.0 / 24.0
        delta_phase = sunset_phase - sunrise_phase
        scale = math.pi / delta_phase
        theta = (phase - sunrise_phase) * scale
        # Sun position on great-circle arc: east horizon -> overhead -> west horizon
        sun_pos = (
            math.cos(theta) * self.sun_east
            + math.sin(theta) * self.sun_up
        )
        sun_pos = sun_pos / max(np.linalg.norm(sun_pos), 1e-6)
        self.sun_world = sun_pos * float(self.sun_radius)
        # Light direction points from the scene toward the sun (for shading)
        return sun_pos.astype("f4")

    def _update_cardinal_labels(self):
        top = np.array([0.0, 0.0, 1.0], dtype="f4")
        def _cardinal(vec):
            v = vec.copy()
            v[1] = 0.0
            n = np.linalg.norm(v)
            if n < 1e-6:
                return "N"
            v /= n
            n_dot = float(np.dot(v, self.compass_north))
            e_dot = float(np.dot(v, self.compass_east))
            if abs(n_dot) >= abs(e_dot):
                return "N" if n_dot >= 0.0 else "S"
            return "E" if e_dot >= 0.0 else "W"
        self.scene_top_cardinal = _cardinal(top)
        self.sun_cardinal = _cardinal(self.light_dir)

    def _set_sun_phase(self, phase, keep_atmos=False):
        self.sun_phase = float(phase % 1.0)
        self.sun_time_hours = self.sun_phase * 24.0
        self.light_dir = self._phase_to_dir(self.sun_phase)
        self._update_cardinal_labels()
        # Update lighting/sky to follow the new phase
        self.regenerate_sky_for_sun()

    def shift_sun_phase(self, delta, keep_atmos=True):
        """Move the sun along its orbit by ``delta`` in phase units (0..1)."""
        self._set_sun_phase(self.sun_phase + delta, keep_atmos=keep_atmos)

    def set_sun_time_hours(self, hour, snap_to_hour=False, keep_atmos=True):
        """Convenience: set sun position by local time (0–24, 6= sunrise east, 18= sunset west)."""
        h = float(hour)
        if snap_to_hour:
            h = round(h) % 24.0
        self._set_sun_phase((h / 24.0) % 1.0, keep_atmos=keep_atmos)

    def _sample_sun_state(self):
        self._sample_sun_basis()
        self._sky_overcast = self._rng.uniform(0.0, 0.8)
        self._sky_turbidity = self._rng.uniform(1.2, 3.5)
        self._set_sun_phase(self._sample_day_phase())

    def regenerate_sky_for_sun(self):
        def smoothstep(edge0, edge1, x):
            t = min(max((x - edge0) / (edge1 - edge0), 0.0), 1.0)
            return t * t * (3.0 - 2.0 * t)

        sun_elev = self.light_dir[1]
        self.sun_intensity = max(0.0, sun_elev) ** 1.4
        # Update fog and light color based on sun
        day_scale = 0.01 + 0.99 * self.sun_intensity
        day_light = np.array([0.95, 0.95, 0.92], dtype="f4")
        tw_light = np.array([1.0, 0.5, 0.3], dtype="f4")
        night_light = np.array([0.1, 0.1, 0.2], dtype="f4")
        day_fac = smoothstep(-0.1, 0.1, sun_elev)
        tw_fac = smoothstep(0.0, 0.2, 0.2 - abs(sun_elev + 0.1)) * (1.0 - day_fac)
        night_fac = 1.0 - day_fac - tw_fac
        self.light_color = (day_light * day_fac + tw_light * tw_fac + night_light * night_fac) * (0.70 + 0.30 * (1.0 - self._sky_overcast)) * day_scale
        # Fog color dynamic with smooth blend
        day_fog = np.array([0.7, 0.8, 1.0], dtype="f4")
        twilight_fog = np.array([1.0, 0.4, 0.2], dtype="f4")
        night_fog = np.array([0.01, 0.01, 0.02], dtype="f4")
        self.base_fog_color = day_fog * day_fac + twilight_fog * tw_fac + night_fog * night_fac
        self.sky_brightness = np.dot(self.base_fog_color, [0.2126, 0.7152, 0.0722])
        self.fog_color = self.base_fog_color.copy()  # update fog to match

    def _init_fog_sheets(self):
        if self.prog_fog_sheet is None:
            self.fog_sheet_count = 0
            self._fog_vertices = None
            return

        rng = self._rng
        self.fog_sheet_count = 28
        angles = rng.uniform(0.0, 2.0 * np.pi, self.fog_sheet_count)
        radial = rng.random(self.fog_sheet_count) ** 1.55
        radii = 10.0 + radial * 70.0
        self._fog_sheet_offsets = np.column_stack(
            [np.cos(angles) * radii, np.sin(angles) * radii]
        ).astype('f4')
        self._fog_sheet_radius = rng.uniform(8.0, 26.0, self.fog_sheet_count).astype('f4')
        self._fog_sheet_height = rng.uniform(14.0, 32.0, self.fog_sheet_count).astype('f4')
        self._fog_sheet_density = rng.uniform(0.45, 1.15, self.fog_sheet_count).astype('f4')
        self._fog_sheet_seed = rng.uniform(0.0, 500.0, self.fog_sheet_count).astype('f4')
        self._fog_vertices = np.zeros((self.fog_sheet_count * 6, 7), dtype='f4')
        if self.fog_vbo is not None:
            self.fog_vbo.release()
        self.fog_vbo = self.ctx.buffer(reserve=self._fog_vertices.nbytes)
        self.fog_vao = self.ctx.vertex_array(
            self.prog_fog_sheet,
            [(self.fog_vbo, '3f 2f 1f 1f', 'in_vert', 'in_uv', 'in_sheet', 'in_density')],
        )
        self._fog_center = self.camera_pos.copy()
        self._fog_scroll[:] = 0.0
        self._fog_field_clock = 0.0

    def set_camera(self, pos):
        self.camera_pos = np.array(pos, dtype='f4')
        if not self._rain_anchor_valid:
            self._rain_anchor = self.camera_pos.copy()
        if not np.any(self._last_camera_pos):
            self._last_camera_pos = self.camera_pos.copy()
        self.update_view()

    def set_camera_pose(self, pos, forward, right, up):
        self.set_camera(pos)
        self.camera_forward = np.array(forward, dtype='f4')
        self.camera_right = np.array(right, dtype='f4')
        self.camera_up = np.array(up, dtype='f4')
        self.update_view()

    def setup_weather(
        self,
        weather,
        terrain_type,
        road_type,
        precipitation="none",
        rain_strength=0.0,
    ):
        self.wetness = 1.0 if weather == 'wet' else 0.0
        self.precipitation = precipitation
        self._init_fog_sheets()

        if precipitation == "rain":
            self.rain_intensity = float(np.clip(rain_strength, 0.0, 1.0))
        else:
            self.rain_intensity = 0.0
        self.rain_spawn_radius = 16.0 + 18.0 * self.rain_intensity
        self.rain_spawn_height = 18.0 + 12.0 * self.rain_intensity
        self._configure_rain_population(force=True)
        self.puddle_strength = float(np.clip(self.wetness * self.rain_intensity, 0.0, 1.0))
        self._rain_anchor = self.camera_pos.copy()
        self._rain_anchor_valid = self.rain_count > 0
        self._last_camera_pos = self.camera_pos.copy()
        self._build_puddle_mesh()

        # Sky brightness from regenerate_sky_for_sun controls base falloff
        self.regenerate_sky_for_sun()
        self.fog_density = (1.0 - self.sky_brightness) * 0.02

        # Start fog from the sky's horizon tone
        horizon = self.base_fog_color

        # Weather/terrain adjustments
        if self.wetness > 0.0:
            self.fog_density += 0.015
        elif terrain_type in ("sand", "gravel", "dirt"):
            self.fog_density += 0.001
            # Dusty terrain nudges the tint sandy
            horizon = 0.5 * horizon + 0.5 * np.array(FOG_DUST_COLOR, dtype="f4")

        if precipitation == "rain" and self.rain_intensity > 0.0:
            rain_fog = 0.022 + 0.020 * self.rain_intensity
            self.fog_density += rain_fog
            horizon = 0.65 * horizon + 0.35 * np.array(FOG_DEFAULT_COLOR, dtype="f4")

        # Keep it sane
        self.fog_density = float(min(self.fog_density, 0.075))

        # Final fog tint: as density rises, drift from horizon toward a neutral/white fog
        neutral = np.array(FOG_DEFAULT_COLOR, dtype="f4")
        d = self.fog_density / 0.05  # 0..1
        # Bias: mostly horizon at low fog, more neutral at high fog
        self.fog_color = ((1.0 - 0.6 * d) * horizon + (0.6 * d) * neutral).astype("f4")

        terrain_map = {'grass': 1, 'dirt': 2, 'sand': 2, 'snow': 3}
        self.terrain_mode = terrain_map.get(terrain_type, 0)

        noise_map = {'asphalt': 2.5, 'concrete': 1.5, 'gravel': 3.5}
        self.road_noise = noise_map.get(road_type, 0.0)

    def set_terrain(self, terrain):
        self.terrain_ref = terrain
        self._build_puddle_mesh()

    def _configure_rain_population(self, force=False):
        density = float(np.clip(self.rain_intensity, 0.0, 1.0))
        if density <= 0.02:
            target = 0
        else:
            eased = density ** 0.85
            blend = 0.35 + 0.50 * eased
            target = int(self.max_rain_drops * blend)
            target = max(int(180 * eased), target)
        target = min(target, self.max_rain_drops)

        if target == 0:
            self.rain_count = 0
            self._rain_vertex_count = 0
            self._rain_anchor_valid = False
            return

        if force or target != self.rain_count:
            self.rain_count = target
            for i in range(self.rain_count):
                self._respawn_drop(i, full_reset=True)
            self._rain_anchor = self.camera_pos.copy()
            self._rain_anchor_valid = True

    def _ground_height(self, x, z):
        if self.terrain_ref is not None:
            h = float(self.terrain_ref.get_height(x, z))
            if np.isfinite(h):
                return h + 0.12
        return float(self.camera_pos[1] - 1.2)

    def _respawn_drop(self, idx, full_reset=False):
        if idx >= self.max_rain_drops:
            return
        anchor = self._rain_anchor if self._rain_anchor_valid else self.camera_pos
        cx, cy, cz = anchor
        rad = self.rain_spawn_radius
        top = self.camera_pos[1] + self.rain_spawn_height
        horiz_vel = np.array([self.camera_velocity[0], 0.0, self.camera_velocity[2]], dtype='f4')
        drift = horiz_vel * 0.45
        speed = float(np.linalg.norm(horiz_vel[:2]))
        if speed > 1e-4:
            forward = horiz_vel / max(speed, 1e-5)
            forward[1] = 0.0
            lead = min(speed * (0.05 + 0.12 * self.rain_intensity), rad * 0.7)
            cx += forward[0] * lead
            cz += forward[2] * lead
        angle = float(self._rng.uniform(0.0, 2.0 * np.pi))
        near_bias = 0.65 + 0.25 * float(self.rain_intensity)
        if self._rng.random() < near_bias:
            local_rad = rad * 0.55
            exp = 0.85 + 0.45 * (1.0 - float(self.rain_intensity))
        else:
            local_rad = rad
            exp = 1.6 + 0.5 * (1.0 - float(self.rain_intensity))
        radius = local_rad * float(self._rng.random() ** exp)
        offset_x = np.cos(angle) * radius
        offset_z = np.sin(angle) * radius
        self.rain_positions[idx, 0] = float(cx + drift[0] + offset_x)
        self.rain_positions[idx, 2] = float(cz + drift[2] + offset_z)
        if full_reset:
            self.rain_positions[idx, 1] = float(top * self._rng.uniform(0.75, 1.05))
        else:
            self.rain_positions[idx, 1] = float(top * self._rng.uniform(0.9, 1.1))
        wind = -0.55 * horiz_vel
        self.rain_velocities[idx, 0] = float(wind[0] + self._rng.uniform(-2.0, 2.5))
        self.rain_velocities[idx, 1] = float(-(28.0 + self._rng.uniform(6.0, 16.0)))
        self.rain_velocities[idx, 2] = float(wind[2] + self._rng.uniform(-2.5, 2.2))
        length_base = 1.2 + 0.8 * self.rain_intensity
        self.rain_lengths[idx] = float(self._rng.uniform(length_base, length_base + 1.1))

    def _update_fog_sheets(self, dt):
        if self.fog_sheet_count == 0 or self._fog_vertices is None:
            return

        follow = float(np.clip(0.35 + dt * 4.2, 0.0, 0.95))
        horiz_vel = np.array([self.camera_velocity[0], 0.0, self.camera_velocity[2]], dtype='f4')
        speed = float(np.linalg.norm(horiz_vel[:2]))
        if speed > 1e-4:
            forward = horiz_vel.copy()
            forward[1] = 0.0
            forward_norm = float(np.linalg.norm(forward))
            forward = forward / max(forward_norm, 1e-5)
        else:
            forward = np.array([0.0, 0.0, 1.0], dtype='f4')

        lead = min(speed * (0.08 + 0.18 * self.rain_intensity), self.rain_spawn_radius * 0.85)
        target_center = self.camera_pos + forward * lead
        target_center[1] = self.camera_pos[1]
        self._fog_center += (target_center - self._fog_center) * follow

        wind = np.array([0.6, -0.35], dtype='f4') + np.array([horiz_vel[0], horiz_vel[2]], dtype='f4') * 0.02
        self._fog_scroll += wind * dt * (6.0 + 4.5 * self.rain_intensity)
        self._fog_field_clock += dt * (0.8 + 1.4 * self.rain_intensity)

        verts = self._fog_vertices
        offsets = self._fog_sheet_offsets
        widths = self._fog_sheet_radius
        heights = self._fog_sheet_height
        densities = self._fog_sheet_density
        seeds = self._fog_sheet_seed

        for i in range(self.fog_sheet_count):
            offset = offsets[i] + self._fog_scroll
            center = self._fog_center.copy()
            center[0] += offset[0]
            center[2] += offset[1]

            sway = np.sin(self._fog_field_clock * (0.6 + 0.2 * self.rain_intensity) + seeds[i] * 0.5)
            center += forward * sway * 2.5

            base_y = self._ground_height(center[0], center[2]) - 0.15
            height = heights[i] * (0.85 + 0.25 * np.sin(self._fog_field_clock * 0.5 + seeds[i]))
            top_y = base_y + height

            to_cam = self.camera_pos - center
            to_cam[1] = 0.0
            dist = float(np.linalg.norm(to_cam))
            if dist < 1e-4:
                to_cam = np.array([0.0, 0.0, 1.0], dtype='f4')
            else:
                to_cam /= dist
            right = np.array([to_cam[2], 0.0, -to_cam[0]], dtype='f4')
            rlen = float(np.linalg.norm(right))
            if rlen < 1e-4:
                right = np.array([1.0, 0.0, 0.0], dtype='f4')
            else:
                right /= rlen

            width = widths[i]
            edge_spread = 0.6 + 0.4 * np.sin(seeds[i] * 0.37 + self._fog_field_clock * 0.4)
            width *= edge_spread

            bottom_offset = np.array([right[0] * width, 0.0, right[2] * width], dtype='f4')
            left = center - bottom_offset
            right_pt = center + bottom_offset

            idx = i * 6
            seed_val = seeds[i]
            density = densities[i]
            verts[idx + 0] = (left[0], base_y, left[2], 0.0, 0.0, seed_val, density)
            verts[idx + 1] = (right_pt[0], base_y, right_pt[2], 1.0, 0.0, seed_val, density)
            verts[idx + 2] = (right_pt[0], top_y, right_pt[2], 1.0, 1.0, seed_val, density)
            verts[idx + 3] = (left[0], base_y, left[2], 0.0, 0.0, seed_val, density)
            verts[idx + 4] = (right_pt[0], top_y, right_pt[2], 1.0, 1.0, seed_val, density)
            verts[idx + 5] = (left[0], top_y, left[2], 0.0, 1.0, seed_val, density)

        if self.fog_vbo is not None:
            self.fog_vbo.write(verts.tobytes(), offset=0)

    def _update_rain(self, dt):
        if self.rain_count == 0:
            self._rain_vertex_count = 0
            return

        if not self._rain_anchor_valid:
            self._rain_anchor = self.camera_pos.copy()
            self._rain_anchor_valid = True
        else:
            horiz_vel = np.array([self.camera_velocity[0], 0.0, self.camera_velocity[2]], dtype='f4')
            speed = float(np.linalg.norm(horiz_vel[:2]))
            if speed > 1e-4:
                forward = horiz_vel / max(speed, 1e-5)
                forward[1] = 0.0
            else:
                forward = np.array([0.0, 0.0, 1.0], dtype='f4')
            lead = min(speed * (0.10 + 0.22 * self.rain_intensity), self.rain_spawn_radius * 0.9)
            target = self.camera_pos + forward * lead
            target[1] = self.camera_pos[1]
            follow = float(np.clip(0.18 + dt * 5.5, 0.0, 0.95))
            self._rain_anchor += (target - self._rain_anchor) * follow

        cx, cy, cz = self._rain_anchor
        rad = self.rain_spawn_radius
        alpha_head = 0.38 + 0.55 * self.rain_intensity
        alpha_tail = 0.04 + 0.18 * self.rain_intensity
        alpha_head = float(np.clip(alpha_head, 0.25, 0.85))
        alpha_tail = float(np.clip(alpha_tail, 0.04, 0.32))

        count = self.rain_count
        positions = self.rain_positions[:count]
        velocities = self.rain_velocities[:count]
        lengths = np.maximum(self.rain_lengths[:count], 0.2)

        # integrate positions
        positions += velocities * dt

        # quick reject using camera height before hitting expensive terrain queries
        low_mask = positions[:, 1] <= (self.camera_pos[1] + 0.8)

        # mask for drops that left the radial bounds
        offsets = positions[:, (0, 2)] - np.array([cx, cz], dtype='f4')
        out_of_radius = (np.abs(offsets[:, 0]) > rad * 1.2) | (np.abs(offsets[:, 1]) > rad * 1.2)

        if np.any(low_mask):
            low_indices = np.nonzero(low_mask)[0]
            ground_heights = np.fromiter(
                (self._ground_height(positions[i, 0], positions[i, 2]) for i in low_indices),
                dtype='f4',
                count=len(low_indices),
            )
            below_ground = np.zeros(count, dtype=bool)
            below_ground[low_indices] = positions[low_indices, 1] <= ground_heights
        else:
            below_ground = np.zeros(count, dtype=bool)

        needs_respawn = below_ground | out_of_radius
        if np.any(needs_respawn):
            respawn_indices = np.nonzero(needs_respawn)[0]
            for i in respawn_indices:
                self._respawn_drop(int(i))
            # refresh local views after respawn
            positions = self.rain_positions[:count]
            velocities = self.rain_velocities[:count]
            lengths = np.maximum(self.rain_lengths[:count], 0.2)

        speeds = np.linalg.norm(velocities, axis=1)
        speeds = np.clip(speeds, 1e-3, None)
        speeds = speeds.astype('f4', copy=False)
        dir_vecs = velocities / speeds[:, None]
        tails = positions - dir_vecs * lengths[:, None]

        brightness = np.clip(speeds / 26.0, 0.55, 1.25)
        brightness = brightness.astype('f4', copy=False)
        tail_cols = np.stack(
            [
                np.minimum(0.55 * brightness, 1.0),
                np.minimum(0.65 * brightness, 1.0),
                np.minimum(0.78 * brightness, 1.0),
                np.full(count, alpha_tail, dtype='f4'),
            ],
            axis=1,
        ).astype('f4', copy=False)
        head_cols = np.stack(
            [
                np.minimum(0.78 * brightness, 1.0),
                np.minimum(0.86 * brightness, 1.0),
                np.minimum(1.00 * brightness, 1.0),
                np.full(count, alpha_head, dtype='f4'),
            ],
            axis=1,
        ).astype('f4', copy=False)

        verts = self._rain_vertices[: count * 2]
        verts[0::2, :3] = tails
        verts[0::2, 3:] = tail_cols
        verts[1::2, :3] = positions
        verts[1::2, 3:] = head_cols

        self._rain_vertex_count = count * 2
        if self._rain_vertex_count > 0:
            self.rain_vbo.write(verts.tobytes(), offset=0)

    def _build_puddle_mesh(self):
        if self.terrain_ref is None:
            return

        terrain = self.terrain_ref
        density = float(np.clip(self.puddle_strength, 0.0, 1.0))

        def clear_puddles():
            if self.puddle_vbo is not None:
                self.puddle_vbo.release()
            self.puddle_vbo = None
            self.puddle_vao = None
            self.puddle_vertices = None

        if density <= 0.01:
            clear_puddles()
            return

        csx = float(getattr(terrain, 'cell_size_x', 1.0))
        csz = float(getattr(terrain, 'cell_size_z', 1.0))
        width = float(getattr(terrain, 'width', 400.0))
        height = float(getattr(terrain, 'height', 400.0))
        rng = np.random.default_rng(9421)
        patches = []
        placed_centers = []

        def too_close(x: float, z: float, radius: float) -> bool:
            for px, pz, pr in placed_centers:
                min_sep = 0.35 * (radius + pr) + 0.45
                if (x - px) ** 2 + (z - pz) ** 2 < min_sep * min_sep:
                    return True
            return False

        # ---- Road puddles -------------------------------------------------
        road_mask = np.asarray(getattr(terrain, 'road_friction', None))
        on_road = None
        if road_mask is not None and road_mask.size > 0 and np.any(road_mask > 0.0):
            road_mask = road_mask > 0.0
            res_x, res_z = road_mask.shape

            def _on_road(x: float, z: float) -> bool:
                if x < 0.0 or z < 0.0 or x > width or z > height:
                    return False
                ix = int(np.clip(round(x / csx), 0, res_x - 1))
                iz = int(np.clip(round(z / csz), 0, res_z - 1))
                return bool(road_mask[ix, iz])

            on_road = _on_road

            valid_cells = np.argwhere(road_mask)
            if valid_cells.size > 0:
                rng.shuffle(valid_cells)
                max_sites = int(min(220, len(valid_cells)) * density)
                if density > 0.0:
                    max_sites = max(1, max_sites)
                if max_sites <= 0:
                    valid_cells = np.asarray([], dtype=valid_cells.dtype).reshape(0, 2)
                neighbour_offsets = [
                    (-2.0, 0.0),
                    (2.0, 0.0),
                    (0.0, -2.0),
                    (0.0, 2.0),
                    (-2.0, -2.0),
                    (2.0, 2.0),
                    (-2.0, 2.0),
                    (2.0, -2.0),
                ]

                max_patch_budget = int(18000 * max(density, 0.15))
                for ix, iz in valid_cells[: max_sites * 4]:
                    if max_patch_budget and len(patches) > max_patch_budget:
                        break
                    if ix < 2 or iz < 2 or ix >= res_x - 2 or iz >= res_z - 2:
                        continue
                    local = road_mask[max(ix - 1, 0) : min(ix + 2, res_x), max(iz - 1, 0) : min(iz + 2, res_z)]
                    if float(np.mean(local)) < 0.45:
                        continue
                    cx = float(ix * csx)
                    cz = float(iz * csz)
                    cx += float(rng.uniform(-0.4 * csx, 0.4 * csx))
                    cz += float(rng.uniform(-0.35 * csz, 0.35 * csz))
                    if not on_road(cx, cz):
                        continue
                    h = float(terrain.get_height(cx, cz))
                    if not np.isfinite(h):
                        continue
                    normal = terrain.get_normal(cx, cz)
                    slope = 1.0 - float(np.clip(normal[1], 0.0, 1.0))
                    if slope > 0.18:
                        continue
                    neighbours = []
                    for dx, dz in neighbour_offsets:
                        nx = cx + dx
                        nz = cz + dz
                        if not on_road(nx, nz):
                            continue
                        nh = float(terrain.get_height(nx, nz))
                        if np.isfinite(nh):
                            neighbours.append(nh)
                    if len(neighbours) < 4:
                        continue
                    neighbour_avg = float(np.mean(neighbours))
                    depth = max(neighbour_avg - h, 0.0) + float(rng.uniform(0.0, 0.03))

                    cluster_scale = 0.6 + 0.4 * density
                    base_clusters = int(rng.integers(1, 4))
                    cluster_count = max(1, int(round(base_clusters * cluster_scale)))
                    for cluster in range(cluster_count):
                        seed = float(rng.random() + cluster * 0.137)
                        offset_angle = float(rng.uniform(0.0, 2.0 * np.pi))
                        offset_dist = float(rng.uniform(0.0, 1.35))
                        cluster_cx = cx + np.cos(offset_angle) * offset_dist
                        cluster_cz = cz + np.sin(offset_angle) * offset_dist
                        if not on_road(cluster_cx, cluster_cz):
                            continue

                        segments = int(rng.integers(8, 15))
                        major = 0.9 + float(rng.uniform(-0.2, 0.55))
                        minor = 0.9 + float(rng.uniform(-0.55, 0.25))
                        orientation = float(rng.uniform(0.0, 2.0 * np.pi))
                        base_radius = float(
                            np.clip(0.48 + depth * 7.0 + rng.uniform(-0.12, 0.6), 0.35, 2.6)
                        )
                        if slope > 0.12:
                            base_radius *= 0.7
                        lobe_freq = int(rng.integers(2, 5))
                        lobe_amp = float(rng.uniform(0.14, 0.36))
                        radii = []
                        for s in range(segments):
                            theta = (2.0 * np.pi * s) / segments
                            lobe_val = np.sin(theta * lobe_freq + seed * 6.0)
                            secondary_val = np.sin(theta * (lobe_freq + 1) + seed * 3.1)
                            jitter_val = 0.55 + 0.55 * rng.random()
                            radius_scale = base_radius * (1.0 + lobe_amp * lobe_val)
                            radius_scale *= 1.0 + 0.18 * secondary_val
                            radius_scale *= jitter_val
                            radii.append(radius_scale)
                        radii = np.asarray(radii, dtype='f4')
                        max_radius = float(np.max(radii))
                        footprint = max_radius * max(major, minor)
                        if too_close(cluster_cx, cluster_cz, footprint):
                            continue

                        verts = []
                        offroad = False
                        for s in range(segments):
                            angle = orientation + (2.0 * np.pi * s) / segments
                            local_x = np.cos(angle) * major
                            local_z = np.sin(angle) * minor
                            scale = radii[s]
                            dx = local_x * scale
                            dz = local_z * scale
                            world_x = cluster_cx + dx
                            world_z = cluster_cz + dz
                            if not on_road(world_x, world_z):
                                offroad = True
                                break
                            world_y = float(terrain.get_height(world_x, world_z))
                            uv_x = local_x * (scale / (max_radius + 1e-5))
                            uv_z = local_z * (scale / (max_radius + 1e-5))
                            verts.append((world_x, world_y + 0.015, world_z, uv_x, uv_z, seed))

                        if offroad or len(verts) != segments:
                            continue

                        placed_centers.append((cluster_cx, cluster_cz, footprint))

                        avg_height = float(np.mean([v[1] for v in verts])) if verts else h + 0.01
                        center = (cluster_cx, avg_height + 0.01, cluster_cz, 0.0, 0.0, seed)

                        for s in range(segments):
                            v0 = center
                            v1 = verts[s]
                            v2 = verts[(s + 1) % segments]
                            patches.extend([v0, v1, v2])

        # ---- Terrain puddles ---------------------------------------------
        heights = np.asarray(getattr(terrain, 'heights', None))
        if heights is not None and heights.size > 0 and heights.shape[0] >= 5 and heights.shape[1] >= 5:
            try:
                slope_x, slope_z = np.gradient(heights, csx, csz, edge_order=2)
            except Exception:  # fallback without spacing parameters
                slope_x, slope_z = np.gradient(heights)
            slope_map = np.sqrt(slope_x ** 2 + slope_z ** 2)

            try:
                window = np.lib.stride_tricks.sliding_window_view(heights, (5, 5))
            except AttributeError:
                window = None

            if window is not None:
                local_avg = window.mean(axis=(-2, -1))
                local_avg = np.pad(local_avg, 2, mode='edge')
                depth_map = np.clip(local_avg - heights, 0.0, None)
                positive = depth_map[depth_map > 0.02]
            else:
                depth_map = None
                positive = np.array([])

            if depth_map is not None and positive.size > 0:
                depth_thresh = max(float(np.percentile(positive, 65)), 0.05)
                candidate_mask = (depth_map >= depth_thresh) & (slope_map < 0.22)
                terrain_cells = np.argwhere(candidate_mask)
                rng.shuffle(terrain_cells)
                max_sites = int(min(90, len(terrain_cells)) * density)
                if density > 0.0:
                    max_sites = max(1, max_sites)
                if max_sites <= 0:
                    terrain_cells = np.asarray([], dtype=terrain_cells.dtype).reshape(0, 2)

                max_patch_budget = int(36000 * max(density, 0.15))
                for ix, iz in terrain_cells[: max_sites * 4]:
                    if max_patch_budget and len(patches) > max_patch_budget:
                        break
                    if ix < 2 or iz < 2 or ix >= heights.shape[0] - 2 or iz >= heights.shape[1] - 2:
                        continue
                    cx = float(ix * csx)
                    cz = float(iz * csz)
                    if cx < 0.0 or cz < 0.0 or cx > width or cz > height:
                        continue
                    depth = float(depth_map[ix, iz])
                    h = float(heights[ix, iz])
                    if not np.isfinite(h) or depth <= 0.0:
                        continue
                    if too_close(cx, cz, 1.5 + depth * 4.0):
                        continue

                    cluster_scale = 0.6 + 0.4 * density
                    base_clusters = int(rng.integers(1, 3))
                    cluster_count = max(1, int(round(base_clusters * cluster_scale)))
                    for cluster in range(cluster_count):
                        seed = float(rng.random() + 0.217 * cluster)
                        offset_angle = float(rng.uniform(0.0, 2.0 * np.pi))
                        offset_dist = float(rng.uniform(0.0, 3.5 + depth * 2.0))
                        cluster_cx = cx + np.cos(offset_angle) * offset_dist
                        cluster_cz = cz + np.sin(offset_angle) * offset_dist
                        if cluster_cx < 0.0 or cluster_cx > width or cluster_cz < 0.0 or cluster_cz > height:
                            continue
                        if on_road is not None and on_road(cluster_cx, cluster_cz):
                            continue

                        segments = int(rng.integers(10, 18))
                        major = 1.4 + float(rng.uniform(-0.3, 0.8))
                        minor = 1.1 + float(rng.uniform(-0.4, 0.6))
                        orientation = float(rng.uniform(0.0, 2.0 * np.pi))
                        base_radius = float(
                            np.clip(2.0 + depth * 4.5 + rng.uniform(-0.4, 1.6), 1.4, 7.5)
                        )
                        lobe_freq = int(rng.integers(2, 6))
                        lobe_amp = float(rng.uniform(0.10, 0.32))
                        radii = []
                        for s in range(segments):
                            theta = (2.0 * np.pi * s) / segments
                            base = 1.0 + lobe_amp * np.sin(theta * lobe_freq + seed * 4.7)
                            base *= 1.0 + 0.20 * np.sin(theta * (lobe_freq + 1) + seed * 2.3)
                            base *= 0.65 + 0.55 * rng.random()
                            radii.append(base_radius * base)
                        radii = np.asarray(radii, dtype='f4')
                        max_radius = float(np.max(radii))
                        footprint = max_radius * max(major, minor)
                        if too_close(cluster_cx, cluster_cz, footprint):
                            continue

                        verts = []
                        off_terrain = False
                        for s in range(segments):
                            angle = orientation + (2.0 * np.pi * s) / segments
                            local_x = np.cos(angle) * major
                            local_z = np.sin(angle) * minor
                            scale = radii[s]
                            dx = local_x * scale
                            dz = local_z * scale
                            world_x = cluster_cx + dx
                            world_z = cluster_cz + dz
                            if world_x < 0.0 or world_z < 0.0 or world_x > width or world_z > height:
                                off_terrain = True
                                break
                            world_y = float(terrain.get_height(world_x, world_z))
                            if not np.isfinite(world_y):
                                off_terrain = True
                                break
                            uv_x = local_x * (scale / (max_radius + 1e-5))
                            uv_z = local_z * (scale / (max_radius + 1e-5))
                            verts.append((world_x, world_y + 0.02, world_z, uv_x, uv_z, seed))

                        if off_terrain or len(verts) != segments:
                            continue

                        placed_centers.append((cluster_cx, cluster_cz, footprint))
                        avg_height = float(np.mean([v[1] for v in verts])) if verts else h + 0.025
                        center = (cluster_cx, avg_height + 0.012, cluster_cz, 0.0, 0.0, seed)
                        for s in range(segments):
                            v0 = center
                            v1 = verts[s]
                            v2 = verts[(s + 1) % segments]
                            patches.extend([v0, v1, v2])

        if not patches:
            clear_puddles()
            return

        self.puddle_vertices = np.asarray(patches, dtype='f4')
        if self.puddle_vbo is not None:
            self.puddle_vbo.release()
        self.puddle_vbo = self.ctx.buffer(self.puddle_vertices.tobytes())
        self.puddle_vao = self.ctx.vertex_array(
            self.prog_puddle,
            [(self.puddle_vbo, '3f 2f 1f', 'in_vert', 'in_uv', 'in_seed')],
        )

    def render_weather(self, mvp, dt):
        self._rain_time += dt
        self.puddle_strength = float(np.clip(self.wetness * self.rain_intensity, 0.0, 1.0))
        if dt > 0.0:
            self.camera_velocity = (self.camera_pos - self._last_camera_pos) / max(dt, 1e-3)
        else:
            self.camera_velocity = np.zeros(3, dtype='f4')
        self._last_camera_pos = self.camera_pos.copy()
        self._update_fog_sheets(dt)
        self._update_rain(dt)

        if (
            self.fog_vao is not None
            and self.fog_sheet_count > 0
            and (self.fog_density > 0.0 or self.rain_intensity > 0.01)
        ):
            was_wire = self.ctx.wireframe
            self.ctx.wireframe = False
            self.ctx.disable(moderngl.DEPTH_TEST)
            self.prog_fog_sheet['mvp'].write(mvp.T.tobytes())
            self._apply_common_uniforms(self.prog_fog_sheet)
            self.prog_fog_sheet['time'].value = float(self._rain_time + self._fog_field_clock)
            self.fog_vao.render(moderngl.TRIANGLES)
            self.ctx.enable(moderngl.DEPTH_TEST)
            self.ctx.wireframe = was_wire

        if self._rain_vertex_count > 0:
            was_wire = self.ctx.wireframe
            self.ctx.wireframe = False
            self.ctx.line_width = 1.2
            self.prog_rain['mvp'].write(mvp.T.tobytes())
            self._apply_common_uniforms(self.prog_rain)
            if 'fade_distance' in self.prog_rain:
                self.prog_rain['fade_distance'].value = float(self.rain_spawn_radius * 1.75)
            self.rain_vao.render(moderngl.LINES, vertices=self._rain_vertex_count)
            self.ctx.line_width = 1.0
            self.ctx.wireframe = was_wire

            self.ctx.line_width = 1.0
            self.ctx.wireframe = was_wire

        if self.puddle_vao is not None and self.puddle_strength > 0.02:
            was_wire = self.ctx.wireframe
            self.ctx.wireframe = False
            self.prog_puddle['mvp'].write(mvp.T.tobytes())
            self._apply_common_uniforms(self.prog_puddle)
            self.prog_puddle['time'].value = self._rain_time
            self.prog_puddle['rain_strength'].value = float(self.rain_intensity)
            self.prog_puddle['wetness'].value = float(self.wetness)
            self.puddle_vao.render(moderngl.TRIANGLES)
            self.ctx.wireframe = was_wire

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
        if 'light_dir' in prog:
            prog['light_dir'].value = tuple(self.light_dir)
        if 'light_color' in prog:
            prog['light_color'].value = tuple(self.light_color)
        if 'ambient_k' in prog:
            # ambient tracks sun intensity (dimmer at night)
            sun_up = float(getattr(self, "sun_intensity", 0.0))
            ambient = 0.002 + 0.18 * sun_up
            prog['ambient_k'].value = float(ambient)
        if 'sky_brightness' in prog:
            prog['sky_brightness'].value = float(getattr(self, 'sky_brightness', 1.0))
        if 'rain_strength' in prog:
            prog['rain_strength'].value = float(self.rain_intensity)
        if 'rain_time' in prog:
            prog['rain_time'].value = float(self._rain_time)
        # headlights
        if 'headlight_dir' in prog:
            prog['headlight_dir'].value = tuple(self.headlight_dir)
        if 'headlight_pos0' in prog:
            prog['headlight_pos0'].value = tuple(self.headlight_pos[0])
        if 'headlight_pos1' in prog:
            prog['headlight_pos1'].value = tuple(self.headlight_pos[1])
        if 'headlight_intensity' in prog:
            prog['headlight_intensity'].value = float(self.headlight_intensity)
        if 'headlight_range' in prog:
            prog['headlight_range'].value = float(self.headlight_range)

    def set_headlights(self, pos_left, pos_right, direction, intensity=1.8, range_m=30.0):
        self.headlight_pos[0] = np.array(pos_left, dtype="f4")
        self.headlight_pos[1] = np.array(pos_right, dtype="f4")
        d = np.array(direction, dtype="f4")
        n = max(np.linalg.norm(d), 1e-6)
        self.headlight_dir = d / n
        self.headlight_intensity = float(intensity)
        self.headlight_range = float(range_m)

    def render_terrain(self, terrain_vao, mvp, noise_scale=0.0, terrain_mode=None):
        self._current_mvp = mvp
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

    def render_lit_mesh(self, vao, mvp, noise_scale=0.0, terrain_mode=0):
        prog = vao.program
        prog['mvp'].write(mvp.T.tobytes())
        self._apply_common_uniforms(prog)
        if 'noise_scale' in prog:
            prog['noise_scale'].value = noise_scale
        if 'terrain_mode' in prog:
            prog['terrain_mode'].value = terrain_mode
        vao.render(moderngl.TRIANGLES)

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

    def render_skid_marks(self, vertices, mvp):
        if vertices.size == 0:
            return
        data = vertices.astype('f4').tobytes()
        if self.skid_vbo is None or len(data) > self.skid_vbo.size:
            if self.skid_vbo is not None:
                self.skid_vbo.release()
            self.skid_vbo = self.ctx.buffer(data)
            self.skid_vao = self.ctx.vertex_array(
                self.prog,
                [(self.skid_vbo, '3f 4f', 'in_vert', 'in_color')],
            )
        else:
            self.skid_vbo.write(data)
        was_wire = self.ctx.wireframe
        self.ctx.wireframe = False
        self.prog['mvp'].write(mvp.T.tobytes())
        self._apply_common_uniforms(self.prog)
        if 'noise_scale' in self.prog:
            self.prog['noise_scale'].value = 0.0
        if 'terrain_mode' in self.prog:
            self.prog['terrain_mode'].value = 0
        self.skid_vao.render(moderngl.TRIANGLES)
        self.ctx.wireframe = was_wire

    def render_car(self, vertices, mvp):
        main_vertices, shock_vertices = vertices
        self.prog['mvp'].write(mvp.T.tobytes())
        self._apply_common_uniforms(self.prog)
        # Force wireframe to render bright white regardless of sun lighting
        if 'ambient_k' in self.prog:
            self.prog['ambient_k'].value = 1.0
        if 'light_color' in self.prog:
            self.prog['light_color'].value = (1.0, 1.0, 1.0)
        if 'light_dir' in self.prog:
            self.prog['light_dir'].value = (0.0, 1.0, 0.0)
        if main_vertices:
            main_array = np.array(main_vertices, dtype='f4')
            main_array = main_array.reshape(-1, 7)
            main_array[:, 3:7] = 1.0
            main_array = main_array.ravel()
            main_data = main_array.tobytes()
            if self.main_vbo is None or len(main_data) > self.main_vbo.size:
                if self.main_vbo:
                    self.main_vbo.release()
                self.main_vbo = self.ctx.buffer(main_data)
                self.main_vao = self.ctx.vertex_array(self.prog, self.main_vbo, 'in_vert', 'in_color')
            else:
                self.main_vbo.write(main_data, offset=0)
            self.ctx.line_width = 1.0
            vertex_count = int(main_array.size // 7)
            self.main_vao.render(moderngl.LINES, vertices=vertex_count)
        if shock_vertices:
            shock_array = np.array(shock_vertices, dtype='f4')
            shock_array = shock_array.reshape(-1, 7)
            shock_array[:, 3:7] = 1.0
            shock_array = shock_array.ravel()
            shock_data = shock_array.tobytes()
            if self.shock_vbo is None or len(shock_data) > self.shock_vbo.size:
                if self.shock_vbo:
                    self.shock_vbo.release()
                self.shock_vbo = self.ctx.buffer(shock_data)
                self.shock_vao = self.ctx.vertex_array(self.prog, self.shock_vbo, 'in_vert', 'in_color')
            else:
                self.shock_vbo.write(shock_data, offset=0)
            self.ctx.line_width = 3.0
            shock_count = int(shock_array.size // 7)
            self.shock_vao.render(moderngl.LINES, vertices=shock_count)

    def render_car_model(self, vertices, mvp):
        """Draw either wireframe or textured model using explicit VAO layouts."""
        tri_vertices, edge_vertices = vertices

        if self.ctx.wireframe:
            if not edge_vertices:
                return
            edge_array = np.asarray(edge_vertices, 'f4')
            edge_array = edge_array.reshape(-1, 7)
            edge_array[:, 3:7] = 1.0
            edge_array = edge_array.ravel()
            data = edge_array.tobytes()
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
            if 'ambient_k' in self.prog:
                self.prog['ambient_k'].value = 1.0
            if 'light_color' in self.prog:
                self.prog['light_color'].value = (1.0, 1.0, 1.0)
            if 'light_dir' in self.prog:
                self.prog['light_dir'].value = (0.0, 1.0, 0.0)
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
        self.ctx.wireframe = was_wire

    def render_debug_lines(self, vertices, mvp):
        """Render simple debug lines (pos/color) using the generic program."""
        if vertices is None:
            return
        arr = np.asarray(vertices, dtype="f4")
        if arr.size == 0:
            return
        data = arr.tobytes()
        if self.debug_vbo is None or len(data) > self.debug_vbo.size:
            if self.debug_vbo:
                self.debug_vbo.release()
            self.debug_vbo = self.ctx.buffer(data)
            self.debug_vao = self.ctx.vertex_array(
                self.prog, self.debug_vbo, 'in_vert', 'in_color'
            )
        else:
            self.debug_vbo.write(data)
        self.prog['mvp'].write(mvp.T.tobytes())
        # Force unlit bright lines
        if 'ambient_k' in self.prog:
            self.prog['ambient_k'].value = 1.0
        if 'light_color' in self.prog:
            self.prog['light_color'].value = (1.0, 1.0, 1.0)
        if 'light_dir' in self.prog:
            self.prog['light_dir'].value = (0.0, 1.0, 0.0)
        self.ctx.line_width = 2.0
        count = int(arr.size // 7)
        self.debug_vao.render(moderngl.LINES, vertices=count)

    def render_skybox(self):
        # Always render the skybox filled even if the main scene uses wireframe
        was_wireframe = self.ctx.wireframe
        self.ctx.wireframe = False
        self.ctx.disable(moderngl.DEPTH_TEST)
        # For sky, use view rotation only (no translation)
        pv_rot = self.projection @ self.view_rot
        self.prog_sky['mvp'].write(pv_rot.T.tobytes())
        self.prog_sky['light_dir'].value = tuple(self.light_dir)
        self.prog_sky['light_color'].value = tuple(self.light_color)
        self.prog_sky['ambient_k'].value = 0.002 + 0.18 * self.sun_intensity
        self.prog_sky['time'].value = self._rain_time
        self.sky_vao.render(moderngl.TRIANGLES)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.wireframe = was_wireframe
        self._render_sun_marker()

    def _render_sun_marker(self):
        """Draw a small white circle representing the sun in world space, anchored to the scene."""
        if self._current_mvp is None:
            return
        sun_pos = getattr(self, "sun_world", None)
        if sun_pos is None or np.linalg.norm(sun_pos) < 1e-4:
            return
        pos4 = np.array([sun_pos[0], sun_pos[1], sun_pos[2], 1.0], dtype="f4")
        clip = self._current_mvp @ pos4
        if abs(clip[3]) < 1e-5:
            return
        ndc = clip[:3] / clip[3]
        x, y = float(ndc[0]), float(ndc[1])
        if abs(x) > 1.2 or abs(y) > 1.2:
            return
        radius = 0.04
        segments = 24
        verts = [x, y, 0.0, 1.0, 1.0, 1.0, 1.0]
        for i in range(segments + 1):
            ang = 2.0 * math.pi * (i / segments)
            vx = x + math.cos(ang) * radius
            vy = y + math.sin(ang) * radius
            verts += [vx, vy, 0.0, 1.0, 1.0, 1.0, 1.0]
        verts = np.array(verts, dtype="f4")
        if self.sun_vbo is None:
            self.sun_vbo = self.ctx.buffer(verts.tobytes())
            self.sun_vao = self.ctx.vertex_array(self.prog, self.sun_vbo, "in_vert", "in_color")
        else:
            self.sun_vbo.write(verts.tobytes())
        was_wire = self.ctx.wireframe
        self.ctx.wireframe = False
        self.sun_vao.render(moderngl.TRIANGLE_FAN)
        self.ctx.wireframe = was_wire

    def render_hud(self, hud_surf, dirty_rects=None):
        self.ctx.viewport = (0, 0, self.width, self.height)

        surf_w, surf_h = hud_surf.get_size()
        if self.hud_tex is None or self.hud_tex_size != (surf_w, surf_h):
            self.hud_tex = self.ctx.texture((surf_w, surf_h), 4)
            self.hud_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            blank = b"\x00" * (surf_w * surf_h * 4)
            self.hud_tex.write(blank)
            self.hud_tex_size = (surf_w, surf_h)

        rects = dirty_rects or [pygame.Rect(0, 0, surf_w, surf_h)]
        bounds = pygame.Rect(0, 0, surf_w, surf_h)
        for rect in rects:
            r = pygame.Rect(rect).clip(bounds)
            if r.width == 0 or r.height == 0:
                continue
            region = hud_surf.subsurface(r)
            hud_data = pygame.image.tostring(region, 'RGBA', True)
            gl_y = surf_h - r.y - r.height
            self.hud_tex.write(
                hud_data,
                viewport=(r.x, gl_y, r.width, r.height),
            )

        was_wireframe = self.ctx.wireframe
        self.ctx.wireframe = False
        self.hud_tex.use(0)
        self.prog2d['tex'] = 0
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.hud_vao.render(moderngl.TRIANGLES)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.wireframe = was_wireframe
