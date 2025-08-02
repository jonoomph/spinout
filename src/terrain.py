# terrain.py
import numpy as np

def build_terrain_vertices(terrain):
    vertices = []
    terrain_color = terrain.color
    for i in range(terrain.res - 1):
        for j in range(terrain.res - 1):
            x1 = i * terrain.cell_size
            z1 = j * terrain.cell_size
            x2 = (i + 1) * terrain.cell_size
            z2 = (j + 1) * terrain.cell_size
            p1 = [x1, terrain.heights[i, j], z1]
            p2 = [x2, terrain.heights[i + 1, j], z1]
            p3 = [x2, terrain.heights[i + 1, j + 1], z2]
            p4 = [x1, terrain.heights[i, j + 1], z2]
            vertices.extend(p1 + terrain_color + p2 + terrain_color)
            vertices.extend(p2 + terrain_color + p3 + terrain_color)
            vertices.extend(p3 + terrain_color + p4 + terrain_color)
            vertices.extend(p4 + terrain_color + p1 + terrain_color)
    return np.array(vertices, dtype='f4')
