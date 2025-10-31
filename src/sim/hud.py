import pygame
import math
from typing import Optional

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
) -> None:
    """
    Draw a slim top-bar HUD onto hud_surf.

    - font_big: large centered speed
    - font_small: all other text
    - FPS shown as: FPS [draw/physics]: X | Y
    - '|' separator between gear and RPM
    - Gradient: solid gray at top, gently fading to transparent
      (fade spans 70% of bar height)
    - Speed text raised slightly for better visibility
    """
    # clear surface
    hud_surf.fill((0, 0, 0, 0))

    w, h = hud_surf.get_size()
    bar_h = 80
    max_alpha = 180

    # gradient: solid at top, fading over bar height
    fade_h = int(bar_h)
    grad = pygame.Surface((w, bar_h), pygame.SRCALPHA)
    for y in range(bar_h):
        if y < fade_h:
            alpha = int(max_alpha * (1 - (y / fade_h)))
        else:
            alpha = 0
        grad.fill((50, 50, 50, alpha), (0, y, w, 1))
    hud_surf.blit(grad, (0, 0))

    # big centered speed, raised a bit (40% down instead of 50%)
    speed_txt = f"{speed_mph:.1f} mph"
    surf_speed = font_big.render(speed_txt, True, (255, 255, 255))
    y_speed = int(bar_h * 0.45)
    rect_speed = surf_speed.get_rect(center=(w // 2, y_speed))
    hud_surf.blit(surf_speed, rect_speed)

    # gear & RPM below speed (with '|')
    if rpm is not None and gear is not None:
        gear_txt = f"{gear} | {int(rpm)} RPM"
        surf_gear = font_small.render(gear_txt, True, (255, 255, 255))
        rect_gear = surf_gear.get_rect(
            center=(w // 2, rect_speed.bottom + font_small.get_height() // 2)
        )
        hud_surf.blit(surf_gear, rect_gear)

    # small-text clusters
    mode_names = {0: "Wireframe", 1: "Textured"}
    cam_names = {0: "Follow Far", 1: "Follow Near", 2: "Driver", 3: "Free Fly"}
    line_h = font_small.get_height() + 3

    # helper for vane drawing
    def _ipoint(pt):
        return int(round(pt[0])), int(round(pt[1]))

    vane_center = (w // 2, int(bar_h * 0.22))
    vane_radius = 22
    ring_color = (120, 120, 120)
    arrow_color = (235, 235, 235)
    pygame.draw.circle(hud_surf, ring_color, vane_center, vane_radius, 1)
    pygame.draw.circle(hud_surf, (70, 70, 70), vane_center, max(vane_radius - 6, 2), 1)
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
        pygame.draw.line(hud_surf, ring_color, _ipoint(inner), _ipoint(outer), 1)
    pygame.draw.circle(
        hud_surf,
        (200, 200, 200),
        (vane_center[0], vane_center[1] - vane_radius + 3),
        2,
    )

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

    # left cluster: FPS, car, surface, wind
    x_left, y = 10, 10
    fps_txt = f"FPS [draw/physics]: {render_fps:.1f} | {physics_fps:.1f}"
    hud_surf.blit(
        font_small.render(fps_txt, True, (255, 255, 255)),
        (x_left, y),
    )
    y += line_h
    if car_info:
        info_txt = f"{car_info} [#1–6]"
        hud_surf.blit(
            font_small.render(info_txt, True, (255, 255, 255)),
            (x_left, y),
        )
        y += line_h
    if surface_info:
        hud_surf.blit(
            font_small.render(f"{surface_info} (T)", True, (255, 255, 255)),
            (x_left, y),
        )
        y += line_h
    wind_state = "Calm" if wind_speed_mph < 0.15 else f"{wind_label} {wind_speed_mph:.1f} mph"
    indicator_state = "ON" if wind_vectors_enabled else "OFF"
    wind_txt = f"Wind [W]: {wind_state} | Indicators {indicator_state}"
    hud_surf.blit(
        font_small.render(wind_txt, True, (255, 255, 255)),
        (x_left, y),
    )

    # right cluster: steer, mode, camera
    x_right, y = w - 10, 10
    steer_txt = (
        f"{controller_name}: {steer_label}: {math.degrees(steer_angle):.1f}°"
    )
    surf_steer = font_small.render(steer_txt, True, (255, 255, 255))
    rect_steer = surf_steer.get_rect(topright=(x_right, y))
    hud_surf.blit(surf_steer, rect_steer)

    y += line_h
    mode_txt = f"Mode: {mode_names.get(render_mode, '?')} (F1/F2)"
    surf_mode = font_small.render(mode_txt, True, (255, 255, 255))
    rect_mode = surf_mode.get_rect(topright=(x_right, y))
    hud_surf.blit(surf_mode, rect_mode)

    y += line_h
    cam_txt = f"Camera: {cam_names.get(camera_mode, '?')} (C/F)"
    surf_cam = font_small.render(cam_txt, True, (255, 255, 255))
    rect_cam = surf_cam.get_rect(topright=(x_right, y))
    hud_surf.blit(surf_cam, rect_cam)
