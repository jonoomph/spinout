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

    vane_center = (w // 2, int(bar_h * 0.22))
    vane_radius = 22
    ring_color = (120, 120, 120)
    pygame.draw.circle(base, ring_color, vane_center, vane_radius, 1)
    pygame.draw.circle(base, (70, 70, 70), vane_center, max(vane_radius - 6, 2), 1)
    def _ipoint(pt):
        return int(round(pt[0])), int(round(pt[1]))

    for ang in (0.0, 90.0, 180.0, 270.0):
        rad = math.radians(ang)
        inner = (
            vane_center[0] + math.sin(rad) * (vane_radius - 6),
            vane_center[1] - math.cos(rad) * (vane_radius - 6),
        )
        outer = (
            vane_center[0] + math.sin(rad) * (vane_radius - 2),
            vane_center[1] - math.cos(rad) * (vane_radius - 2),
        )
        pygame.draw.line(base, ring_color, _ipoint(inner), _ipoint(outer), 1)
    pygame.draw.circle(
        base,
        (200, 200, 200),
        (vane_center[0], vane_center[1] - vane_radius + 3),
        2,
    )
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

    def _register_area(name: str, target: pygame.Rect, pad: int = 2) -> bool:
        padded = target.inflate(pad * 2, pad * 2).clip(bounds)
        if padded.width <= 0 or padded.height <= 0:
            _clear_area(name)
            return False
        prev = areas.get(name)
        combined = padded if prev is None else padded.union(prev).clip(bounds)
        areas[name] = padded
        dirty.append(combined.copy())
        return True

    def _clear_area(name: str) -> None:
        rect = areas.pop(name, None)
        if rect:
            dirty.append(rect.copy())

    # small-text clusters
    mode_names = {0: "Wireframe", 1: "Textured"}
    cam_names = {0: "Follow Far", 1: "Follow Near", 2: "Driver", 3: "Free Fly"}
    line_h = font_small.get_height() + 3

    # helper for vane drawing
    def _ipoint(pt):
        return int(round(pt[0])), int(round(pt[1]))

    vane_center = (w // 2, int(bar_h * 0.22))
    vane_radius = 22
    arrow_color = (235, 235, 235)
    vane_bounds = pygame.Rect(
        vane_center[0] - vane_radius - 8,
        vane_center[1] - vane_radius - 8,
        (vane_radius + 8) * 2,
        (vane_radius + 8) * 2,
    )
    _register_area("wind_indicator", vane_bounds, pad=0)
    if wind_speed_mph < 0.15:
        pygame.draw.circle(hud_surf, arrow_color, vane_center, 4)
    else:
        rad = math.radians(wind_direction_deg % 360.0)
        dx = math.sin(rad)
        dy = -math.cos(rad)
        norm = max(0.0, min(wind_speed_mph / 32.0, 1.0))
        tail_len = vane_radius * (0.28 + 0.22 * norm)
        arrow_len = vane_radius * (0.6 + 0.4 * norm)
        tail = (
            vane_center[0] - dx * tail_len,
            vane_center[1] - dy * tail_len,
        )
        tip = (
            vane_center[0] + dx * arrow_len,
            vane_center[1] + dy * arrow_len,
        )
        pygame.draw.line(hud_surf, arrow_color, _ipoint(tail), _ipoint(tip), 3)
        perp = (-dy, dx)
        head = 6 + norm * 5
        left = (
            tip[0] - dx * 8 - perp[0] * head,
            tip[1] - dy * 8 - perp[1] * head,
        )
        right = (
            tip[0] - dx * 8 + perp[0] * head,
            tip[1] - dy * 8 + perp[1] * head,
        )
        pygame.draw.polygon(
            hud_surf,
            arrow_color,
            [_ipoint(tip), _ipoint(left), _ipoint(right)],
        )

    # big centered speed, raised a bit (40% down instead of 50%)
    speed_txt = f"{speed_mph:.1f} mph"
    surf_speed = font_big.render(speed_txt, True, (255, 255, 255))
    y_speed = int(bar_h * 0.45)
    rect_speed = surf_speed.get_rect(center=(w // 2, y_speed))
    if _register_area("speed", rect_speed, pad=4):
        hud_surf.blit(surf_speed, rect_speed)

    # gear & RPM below speed (with '|')
    if rpm is not None and gear is not None:
        gear_txt = f"{gear} | {int(rpm)} RPM"
        surf_gear = font_small.render(gear_txt, True, (255, 255, 255))
        rect_gear = surf_gear.get_rect(
            center=(w // 2, rect_speed.bottom + font_small.get_height() // 2)
        )
        if _register_area("gear", rect_gear):
            hud_surf.blit(surf_gear, rect_gear)
    else:
        _clear_area("gear")

    # left cluster: FPS, car, surface, wind
    x_left, y = 10, 10
    fps_txt = f"FPS [draw/physics]: {render_fps:.1f} | {physics_fps:.1f}"
    surf_fps = font_small.render(fps_txt, True, (255, 255, 255))
    rect = surf_fps.get_rect(topleft=(x_left, y))
    if _register_area("left_0", rect):
        hud_surf.blit(surf_fps, rect)
    y += line_h
    line_idx = 1
    if car_info:
        info_txt = f"{car_info} [#1–6]"
        surf_info = font_small.render(info_txt, True, (255, 255, 255))
        rect_info = surf_info.get_rect(topleft=(x_left, y))
        if _register_area(f"left_{line_idx}", rect_info):
            hud_surf.blit(surf_info, rect_info)
        y += line_h
        line_idx += 1
    if surface_info:
        surf_surface = font_small.render(f"{surface_info} (T)", True, (255, 255, 255))
        rect_surface = surf_surface.get_rect(topleft=(x_left, y))
        if _register_area(f"left_{line_idx}", rect_surface):
            hud_surf.blit(surf_surface, rect_surface)
        y += line_h
        line_idx += 1
    wind_state = "Calm" if wind_speed_mph < 0.15 else f"{wind_label} {wind_speed_mph:.1f} mph"
    indicator_state = "ON" if wind_vectors_enabled else "OFF"
    wind_txt = f"Wind [W]: {wind_state} | Indicators {indicator_state}"
    surf_wind = font_small.render(wind_txt, True, (255, 255, 255))
    rect_wind = surf_wind.get_rect(topleft=(x_left, y))
    if _register_area(f"left_{line_idx}", rect_wind):
        hud_surf.blit(surf_wind, rect_wind)
    line_idx += 1

    prev_left = state.get("left_lines", 0)
    for idx in range(line_idx, prev_left):
        _clear_area(f"left_{idx}")
    state["left_lines"] = line_idx

    # right cluster: steer, mode, camera
    x_right, y = w - 10, 10
    steer_txt = (
        f"{controller_name}: {steer_label}: {math.degrees(steer_angle):.1f}°"
    )
    surf_steer = font_small.render(steer_txt, True, (255, 255, 255))
    rect_steer = surf_steer.get_rect(topright=(x_right, y))
    if _register_area("right_0", rect_steer):
        hud_surf.blit(surf_steer, rect_steer)

    y += line_h
    mode_txt = f"Mode: {mode_names.get(render_mode, '?')} (F1/F2)"
    surf_mode = font_small.render(mode_txt, True, (255, 255, 255))
    rect_mode = surf_mode.get_rect(topright=(x_right, y))
    if _register_area("right_1", rect_mode):
        hud_surf.blit(surf_mode, rect_mode)

    y += line_h
    cam_txt = f"Camera: {cam_names.get(camera_mode, '?')} (C/F)"
    surf_cam = font_small.render(cam_txt, True, (255, 255, 255))
    rect_cam = surf_cam.get_rect(topright=(x_right, y))
    if _register_area("right_2", rect_cam):
        hud_surf.blit(surf_cam, rect_cam)

    return dirty
