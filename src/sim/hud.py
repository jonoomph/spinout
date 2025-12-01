import math
import weakref
from typing import Dict, List, Optional, Tuple

import pygame

_HUD_STATE: "weakref.WeakKeyDictionary[pygame.Surface, Dict[str, object]]" = (
    weakref.WeakKeyDictionary()
)


def _build_static_layer(size: Tuple[int, int]) -> Tuple[pygame.Surface, pygame.Rect]:
    base = pygame.Surface(size, pygame.SRCALPHA)
    w, h = size
    bar_h = min(80, h)
    max_alpha = 180
    grad = pygame.Surface((w, bar_h), pygame.SRCALPHA)
    for y in range(bar_h):
        alpha = int(max_alpha * (1 - (y / bar_h))) if y < bar_h else 0
        grad.fill((50, 50, 50, alpha), (0, y, w, 1))
    base.blit(grad, (0, 0))
    bar_rect = pygame.Rect(0, 0, w, bar_h)
    return base, bar_rect


def _ensure_state(hud_surf: pygame.Surface) -> Dict[str, object]:
    state = _HUD_STATE.get(hud_surf)
    size = hud_surf.get_size()
    if state is None or state.get("size") != size or "bar_rect" not in state:
        base, bar_rect = _build_static_layer(size)
        hud_surf.blit(base, (0, 0))
        state = {
            "size": size,
            "base": base,
            "bar_rect": bar_rect,
            "areas": {},
            "full_dirty": True,
            "left_lines": 0,
        }
        _HUD_STATE[hud_surf] = state
    return state


def render_hud(
    hud_surf: pygame.Surface,
    font_small: pygame.font.Font,
    font_big: pygame.font.Font,
    speed_mph: float,
    render_fps: float,
    physics_fps: float,
    steer_angle: float,
    car_info: str = "",
    rpm: Optional[float] = None,
    gear: Optional[int] = None,
    surface_info: Optional[str] = None,
    render_mode: int = 0,
    camera_mode: int = 0,
    wind_speed_mph: float = 0.0,
    wind_direction_deg: float = 0.0,
    wind_label: str = "Calm",
    wind_vectors_enabled: bool = False,
    controller_name: str = "No Controller",
    steer_label: str = "Manual Steer",
    sun_time_hours: Optional[float] = None,
    sun_cardinal: str = "",
    scene_top_cardinal: str = "",
    sun_azimuth_deg: Optional[float] = None,
    heading_deg: Optional[float] = None,
) -> List[pygame.Rect]:
    """
    Draw HUD content and report the rectangles that changed.

    The returned rectangles are used by the renderer to upload
    only the modified parts of the HUD texture each frame.
    """
    w, h = hud_surf.get_size()
    bar_h = 80

    state = _ensure_state(hud_surf)
    base = state["base"]
    bar_rect = state["bar_rect"]
    bounds = pygame.Rect(0, 0, w, h)
    areas: Dict[str, pygame.Rect] = state["areas"]  # type: ignore[assignment]
    dirty: List[pygame.Rect] = []

    hud_surf.fill((0, 0, 0, 0), bar_rect)
    hud_surf.blit(base, bar_rect, bar_rect)

    if state.get("full_dirty"):
        dirty.append(bar_rect.copy())
        state["full_dirty"] = False

    def _register_area(
        name: str, target: pygame.Rect, pad: int = 2, clear_base: bool = False
    ) -> bool:
        padded = target.inflate(pad * 2, pad * 2).clip(bounds)
        if padded.width <= 0 or padded.height <= 0:
            _clear_area(name)
            return False
        if clear_base:
            hud_surf.blit(base, padded, padded)
        prev = areas.get(name)
        combined = padded if prev is None else padded.union(prev).clip(bounds)
        areas[name] = padded
        dirty.append(combined.copy())
        return True

    def _clear_area(name: str) -> None:
        rect = areas.pop(name, None)
        if rect:
            dirty.append(rect.copy())

    def _blit_text(name: str, surf: pygame.Surface, rect: pygame.Rect, pad: int = 2):
        if _register_area(name, rect, pad=pad, clear_base=True):
            hud_surf.blit(surf, rect)

    # small-text clusters
    mode_names = {0: "Wireframe", 1: "Textured"}
    cam_names = {0: "Follow Far", 1: "Follow Near", 2: "Driver", 3: "Free Fly"}
    line_h = font_small.get_height() + 3

    def _ipoint(pt):
        return int(round(pt[0])), int(round(pt[1]))

    # Remove unused sun overlay/time on the left, but keep central compass + time
    _clear_area("sun_info")
    # Simplified compass strip with flowing labels + time box beside it
    compass_h = 26
    compass_w = 200
    compass_rect = pygame.Rect(
        w // 2 - compass_w // 2, 8, compass_w, compass_h
    )
    time_rect = pygame.Rect(
        compass_rect.right + 12, compass_rect.y, 86, compass_h
    )

    if heading_deg is not None:
        _register_area("compass_strip", compass_rect, pad=0, clear_base=True)
        hud_surf.fill((0, 0, 0, 0), compass_rect)
        pygame.draw.rect(hud_surf, (70, 70, 70), compass_rect, border_radius=4)
        pygame.draw.rect(hud_surf, (140, 140, 140), compass_rect, 1, border_radius=4)

        labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        label_angles = [i * 45.0 for i in range(len(labels))]  # 0=N
        used_labels = []

        def _angle_diff(a, b):
            d = (a - b + 180.0) % 360.0 - 180.0
            return d

        pad = 8
        span_px = max(1, compass_w - pad * 2)
        for lbl, ang in zip(labels, label_angles):
            delta = _angle_diff(ang, heading_deg)
            # Show quarter of the compass: +/-45 degrees
            if abs(delta) > 60.0:  # soft cull
                continue
            clamped = max(-45.0, min(45.0, delta))
            norm = (clamped + 45.0) / 90.0
            x = compass_rect.left + pad + norm * span_px
            t = 1.0 - min(abs(clamped) / 45.0, 1.0)
            scale = 0.65 + 1.00 * t
            alpha = int(80 + 175 * t)
            surf = font_small.render(lbl, True, (235, 235, 235))
            size = surf.get_size()
            surf = pygame.transform.smoothscale(
                surf,
                (
                    max(1, int(size[0] * scale)),
                    max(1, int(size[1] * scale)),
                ),
            )
            surf.set_alpha(alpha)
            rect = surf.get_rect(center=(int(x), compass_rect.centery))
            name = f"compass_lbl_{lbl}"
            _blit_text(name, surf, rect, pad=0)
            used_labels.append(name)
        prev_labels = state.get("compass_labels", [])
        for name in prev_labels:
            if name not in used_labels:
                _clear_area(name)
        state["compass_labels"] = used_labels
    else:
        _clear_area("compass_strip")
        for name in state.get("compass_labels", []):
            _clear_area(name)
        state["compass_labels"] = []

    if sun_time_hours is not None:
        hh = int(sun_time_hours) % 24
        mm = int((sun_time_hours * 60.0) % 60)
        time_txt = f"{hh:02d}:{mm:02d}"
        surf = font_small.render(time_txt, True, (255, 190, 90))
        rect = surf.get_rect(center=time_rect.center)
        _register_area("sun_clock", time_rect, pad=0, clear_base=True)
        hud_surf.fill((0, 0, 0, 0), time_rect)
        pygame.draw.rect(hud_surf, (60, 60, 60), time_rect, border_radius=4)
        pygame.draw.rect(hud_surf, (140, 140, 140), time_rect, 1, border_radius=4)
        hud_surf.blit(surf, rect)
    else:
        _clear_area("sun_clock")

    # vertical offset for other HUD text (keep left/right rows high even with compass present)
    cluster_top_y = 10

    # big centered speed, lower to leave room for compass
    speed_txt = f"{speed_mph:.1f} mph"
    surf_speed = font_big.render(speed_txt, True, (255, 255, 255))
    y_speed = int(bar_h * 0.65)
    rect_speed = surf_speed.get_rect(center=(w // 2, y_speed))
    _blit_text("speed", surf_speed, rect_speed, pad=4)

    # gear & RPM below speed (with '|')
    if rpm is not None and gear is not None:
        gear_txt = f"{gear} | {int(rpm)} RPM"
        surf_gear = font_small.render(gear_txt, True, (255, 255, 255))
        rect_gear = surf_gear.get_rect(
            center=(w // 2, rect_speed.bottom + font_small.get_height() // 2)
        )
        _blit_text("gear", surf_gear, rect_gear)
    else:
        _clear_area("gear")

    # left cluster: FPS, car, surface, wind
    x_left, y = 10, cluster_top_y
    fps_txt = f"FPS [draw/physics]: {render_fps:.1f} | {physics_fps:.1f}"
    surf_fps = font_small.render(fps_txt, True, (255, 255, 255))
    rect = surf_fps.get_rect(topleft=(x_left, y))
    _blit_text("left_0", surf_fps, rect)
    y += line_h
    line_idx = 1
    if car_info:
        info_txt = f"{car_info} [#1–6]"
        surf_info = font_small.render(info_txt, True, (255, 255, 255))
        rect_info = surf_info.get_rect(topleft=(x_left, y))
        _blit_text(f"left_{line_idx}", surf_info, rect_info)
        y += line_h
        line_idx += 1
    if surface_info:
        surf_surface = font_small.render(f"{surface_info} (T)", True, (255, 255, 255))
        rect_surface = surf_surface.get_rect(topleft=(x_left, y))
        _blit_text(f"left_{line_idx}", surf_surface, rect_surface)
        y += line_h
        line_idx += 1
    wind_state = "Calm" if wind_speed_mph < 0.15 else f"{wind_label} {wind_speed_mph:.1f} mph"
    indicator_state = "ON" if wind_vectors_enabled else "OFF"
    wind_txt = f"Wind [W]: {wind_state} | Indicators {indicator_state}"
    surf_wind = font_small.render(wind_txt, True, (255, 255, 255))
    rect_wind = surf_wind.get_rect(topleft=(x_left, y))
    _blit_text(f"left_{line_idx}", surf_wind, rect_wind)
    line_idx += 1

    prev_left = state.get("left_lines", 0)
    for idx in range(line_idx, prev_left):
        _clear_area(f"left_{idx}")
    state["left_lines"] = line_idx

    # right cluster: steer, mode, camera
    x_right, y = w - 10, cluster_top_y
    steer_txt = (
        f"{controller_name}: {steer_label}: {math.degrees(steer_angle):.1f}°"
    )
    surf_steer = font_small.render(steer_txt, True, (255, 255, 255))
    rect_steer = surf_steer.get_rect(topright=(x_right, y))
    _blit_text("right_0", surf_steer, rect_steer)

    y += line_h
    mode_txt = f"Mode: {mode_names.get(render_mode, '?')} (F1/F2)"
    surf_mode = font_small.render(mode_txt, True, (255, 255, 255))
    rect_mode = surf_mode.get_rect(topright=(x_right, y))
    _blit_text("right_1", surf_mode, rect_mode)

    y += line_h
    cam_txt = f"Camera: {cam_names.get(camera_mode, '?')} (C/F)"
    surf_cam = font_small.render(cam_txt, True, (255, 255, 255))
    rect_cam = surf_cam.get_rect(topright=(x_right, y))
    _blit_text("right_2", surf_cam, rect_cam)

    return dirty
