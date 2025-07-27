# hud.py
import pygame
import math

def render_hud(screen, font, speed_mph, render_fps, physics_fps, steer_angle):
    text_speed = font.render(f"Speed: {speed_mph:.1f} mph", True, (0, 0, 0))
    text_render_fps = font.render(f"Render FPS: {render_fps:.1f}", True, (0, 0, 0))
    text_physics_fps = font.render(f"Physics FPS: {physics_fps:.1f}", True, (0, 0, 0))
    text_steer = font.render(f"Steer Angle: {math.degrees(steer_angle):.1f}°", True, (0, 0, 0))
    screen.blit(text_speed, (10, 10))
    screen.blit(text_render_fps, (10, 40))
    screen.blit(text_physics_fps, (10, 70))
    screen.blit(text_steer, (10, 100))