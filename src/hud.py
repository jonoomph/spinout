import pygame
import math

def render_hud(
    hud_surf: pygame.Surface,
    font_small: pygame.font.Font,
    font_big: pygame.font.Font,
    speed_mph: float,
    render_fps: float,
    physics_fps: float,
    steer_angle: float,
    car_info: str = "",
    rpm: float | None = None,
    gear: int | None = None,
    surface_info: str | None = None,
    render_mode: int = 0,
    camera_mode: int = 0,
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

    w, bar_h = hud_surf.get_size()
    max_alpha = 180

    # gradient: solid at top, fading over 70% of height
    fade_h = int(bar_h * 1.25)
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
    y_speed = int(bar_h * 0.4)
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
    cam_names = {0: "Follow Far", 1: "Follow Near", 2: "Driver"}
    line_h = font_small.get_height() + 3

    # left cluster: FPS, car, surface
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

    # right cluster: steer, mode, camera
    x_right, y = w - 10, 10
    steer_txt = f"Steer: {math.degrees(steer_angle):.1f}°"
    surf_steer = font_small.render(steer_txt, True, (255, 255, 255))
    rect_steer = surf_steer.get_rect(topright=(x_right, y))
    hud_surf.blit(surf_steer, rect_steer)

    y += line_h
    mode_txt = f"Mode: {mode_names.get(render_mode, '?')} (F1/F2)"
    surf_mode = font_small.render(mode_txt, True, (255, 255, 255))
    rect_mode = surf_mode.get_rect(topright=(x_right, y))
    hud_surf.blit(surf_mode, rect_mode)

    y += line_h
    cam_txt = f"Camera: {cam_names.get(camera_mode, '?')} (C)"
    surf_cam = font_small.render(cam_txt, True, (255, 255, 255))
    rect_cam = surf_cam.get_rect(topright=(x_right, y))
    hud_surf.blit(surf_cam, rect_cam)
