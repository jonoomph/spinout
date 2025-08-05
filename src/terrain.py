# terrain.py
import numpy as np

def build_terrain_vertices(terrain):
    res = terrain.res
    cs = terrain.cell_size
    color = np.asarray(terrain.color, dtype='f4')
    num_cells = (res - 1) * (res - 1)

    # Generate all cell indices
    I, J = np.meshgrid(np.arange(res - 1), np.arange(res - 1), indexing='ij')
    I = I.ravel()
    J = J.ravel()

    # Get all quad corners
    x1 = I * cs
    z1 = J * cs
    x2 = (I + 1) * cs
    z2 = (J + 1) * cs
    h11 = terrain.heights[I, J]
    h21 = terrain.heights[I + 1, J]
    h22 = terrain.heights[I + 1, J + 1]
    h12 = terrain.heights[I, J + 1]

    # Vertices for each quad (p1, p2, p3, p4)
    p1 = np.stack([x1, h11, z1], axis=-1)
    p2 = np.stack([x2, h21, z1], axis=-1)
    p3 = np.stack([x2, h22, z2], axis=-1)
    p4 = np.stack([x1, h12, z2], axis=-1)

    # For each quad, generate four triangle strips:
    # (p1->p2), (p2->p3), (p3->p4), (p4->p1)
    verts = np.concatenate([
        np.concatenate([p1, p2], axis=-1),
        np.concatenate([p2, p3], axis=-1),
        np.concatenate([p3, p4], axis=-1),
        np.concatenate([p4, p1], axis=-1)
    ], axis=0)
    # verts shape: (4 * num_cells, 6) -- (p, p)

    # Interleave color for each vertex (one color per vertex triplet)
    # Each "edge" above produces two vertices, so duplicate color for both
    colors = np.tile(color, (verts.shape[0], 1))
    # Flatten into (N, xyzrgb)
    vertices = np.concatenate([verts[:, :3], colors, verts[:, 3:], colors], axis=1)
    # Now, split into triangles (as in your original function)
    # Each quad produces four lines (eight vertices, to be interpreted as lines, not triangles)
    # If you want triangles (not lines), switch to below approach!

    # Generate triangles from quad (2 triangles per quad)
    tri_verts = np.concatenate([
        # First triangle: p1, p2, p3
        np.concatenate([p1, p2, p3], axis=0),
        # Second triangle: p1, p3, p4
        np.concatenate([p1, p3, p4], axis=0)
    ], axis=0)
    tri_colors = np.tile(color, (tri_verts.shape[0], 1))
    vertices = np.concatenate([tri_verts, tri_colors], axis=1)

    return vertices.astype('f4')


def build_terrain_triangles(terrain):
    """Vectorized and efficient mesh builder, matching original output shapes."""
    res = terrain.res
    cs = terrain.cell_size
    color = np.asarray(terrain.color, dtype='f4')
    num_cells = (res - 1) * (res - 1)

    # Grid indices for each cell
    I, J = np.meshgrid(np.arange(res - 1), np.arange(res - 1), indexing='ij')
    I = I.ravel()
    J = J.ravel()

    # Get all quad corners
    x1 = I * cs
    z1 = J * cs
    x2 = (I + 1) * cs
    z2 = (J + 1) * cs
    h11 = terrain.heights[I, J]
    h21 = terrain.heights[I + 1, J]
    h22 = terrain.heights[I + 1, J + 1]
    h12 = terrain.heights[I, J + 1]

    # Vertices for each quad
    p1 = np.stack([x1, h11, z1], axis=-1)
    p2 = np.stack([x2, h21, z1], axis=-1)
    p3 = np.stack([x2, h22, z2], axis=-1)
    p4 = np.stack([x1, h12, z2], axis=-1)

    # Two triangles per quad: [p1, p2, p3], [p1, p3, p4]
    tri1 = np.stack([p1, p2, p3], axis=1)  # shape: (num_cells, 3, 3)
    tri2 = np.stack([p1, p3, p4], axis=1)
    all_tris = np.concatenate([tri1, tri2], axis=0).reshape(-1, 3)

    # Unlit (basic) mesh: position + color
    basic = np.concatenate([
        all_tris,
        np.tile(color, (all_tris.shape[0], 1))
    ], axis=1).astype('f4')

    # Compute flat normals for each triangle (one normal per triangle, shared by its 3 vertices)
    v0 = all_tris[0::3]
    v1 = all_tris[1::3]
    v2 = all_tris[2::3]
    normals = np.cross(v2 - v0, v1 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / (norms + 1e-8)

    # Each triangle: 3 vertices, so repeat each normal 3x
    normals_repeated = np.repeat(normals, 3, axis=0)

    # Lit mesh: position + normal + color
    lit = np.concatenate([
        all_tris,
        normals_repeated,
        np.tile(color, (all_tris.shape[0], 1))
    ], axis=1).astype('f4')

    return basic, lit
