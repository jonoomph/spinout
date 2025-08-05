# hud.py
import pygame
import math

def render_hud(
    hud_surf,
    font,
    speed_mph,
    render_fps,
    physics_fps,
    steer_angle,
    car_info="",
    rpm=None,
    gear=None,
    surface_info=None,
    render_mode=0,
):
    hud_surf.fill((0, 0, 0, 0))
    if rpm is not None and gear is not None:
        text_speed = font.render(
            f"Speed: {speed_mph:.1f} mph [gear {gear} @ {int(rpm)} RPM]",
            True,
            (255, 255, 255, 255),
        )
    else:
        text_speed = font.render(
            f"Speed: {speed_mph:.1f} mph",
            True,
            (255, 255, 255, 255),
        )
    text_render_fps = font.render(f"Render FPS: {render_fps:.1f}", True, (255, 255, 255, 255))
    text_physics_fps = font.render(f"Physics FPS: {physics_fps:.1f}", True, (255, 255, 255, 255))
    text_steer = font.render(f"Steer Angle: {math.degrees(steer_angle):.1f}°", True, (255, 255, 255, 255))
    text_car = font.render(car_info, True, (255, 255, 255, 255))
    hud_surf.blit(text_speed, (10, 10))
    hud_surf.blit(text_render_fps, (10, 40))
    hud_surf.blit(text_physics_fps, (10, 70))
    hud_surf.blit(text_steer, (10, 100))
    hud_surf.blit(text_car, (10, 130))
    if surface_info:
        text_surface = font.render(surface_info, True, (255, 255, 255, 255))
        hud_surf.blit(text_surface, (10, 160))
    mode_names = {0: "Wireframe", 1: "Solid", 2: "Sunlit"}
    text_mode = font.render(
        f"Render: {mode_names.get(render_mode, '?')} (F1/F2/F3)",
        True,
        (255, 255, 255, 255),
    )
    hud_surf.blit(text_mode, (10, 190))
