import base64
import json
from io import BytesIO
from typing import Dict, List
import numpy as np
from PIL import Image
from .colors import WIRE_COLOR


def load_bbmodel(path: str) -> Dict:
    """Load a Blockbench .bbmodel and return positions, uvs, edges,
    model size, raw texture bytes, and texture dimensions."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Decode texture (data URI or external file)
    tex_info = data['textures'][0]
    src = tex_info.get('source', '')
    if src.startswith('data:'):
        b64 = src.split(',', 1)[1]
        tex_img = Image.open(BytesIO(base64.b64decode(b64))).convert('RGBA')
    else:
        tex_path = tex_info.get('relative_path') or tex_info.get('path')
        tex_img = Image.open(tex_path).convert('RGBA')

    # Optional: Flip the texture vertically if needed (uncomment to test)
    # tex_img = tex_img.transpose(Image.FLIP_TOP_BOTTOM)

    # Optional: Flip horizontally if the model appears mirrored (uncomment to test)
    # tex_img = tex_img.transpose(Image.FLIP_LEFT_RIGHT)

    tex_w, tex_h = tex_img.size
    uv_w = float(tex_info.get('uv_width', tex_w))
    uv_h = float(tex_info.get('uv_height', tex_h))

    elem = data['elements'][0]
    verts_raw = elem['vertices']
    pts = np.array(list(verts_raw.values()), dtype='f4')
    min_v, max_v = pts.min(axis=0), pts.max(axis=0)
    center = (min_v + max_v) / 2

    # center each vertex
    verts = {k: np.array(v, 'f4') - center for k, v in verts_raw.items()}

    positions: List[np.ndarray] = []
    uvs: List[np.ndarray] = []
    edges: List[np.ndarray] = []
    edge_set = set()

    rot_flip = np.array([-1, 1, -1], dtype='f4')  # Rotate 180 around Y to fix facing and horizontal flip

    for face in elem['faces'].values():
        keys   = face['vertices']
        uv_map = face.get('uv', {})

        # triangulate dynamically for quads based on shorter diagonal
        if len(keys) == 3:
            tri_idx = [0, 1, 2]
        elif len(keys) == 4:
            p0 = verts[keys[0]]
            p1 = verts[keys[1]]
            p2 = verts[keys[2]]
            p3 = verts[keys[3]]
            d02 = np.linalg.norm(p0 - p2)
            d13 = np.linalg.norm(p1 - p3)
            if d02 <= d13:
                tri_idx = [0, 1, 2, 0, 2, 3]
            else:
                tri_idx = [0, 1, 3, 1, 2, 3]
        else:
            tri_idx = []
            for i in range(1, len(keys) - 1):
                tri_idx.extend((0, i, i+1))

        # build tris
        for idx in tri_idx:
            vid = keys[idx]
            pos = verts[vid] * rot_flip
            positions.append(pos)

            # pick a UV-transform variant below by uncommenting it
            if isinstance(uv_map, dict) and vid in uv_map:
                u_px, v_px = uv_map[vid]
            else:
                # fallback box-UV [u0,v0,u1,v1]
                u0, v0, u1, v1 = face.get('uv', [0,0,0,0])
                rect = [(u0,v0),(u1,v0),(u1,v1),(u0,v1)]
                u_px, v_px = rect[idx % 4]

            # Using no vertical flip (bottom-left origin)
            uv = [ u_px/uv_w, v_px/uv_h ]

            # If texture still appears horizontally flipped after coord flip, try:
            # uv = [ 1.0 - (u_px/uv_w), v_px/uv_h ]

            uvs.append(uv)

        # collect unique edges
        for i in range(len(keys)):
            a, b = keys[i], keys[(i+1) % len(keys)]
            edge = tuple(sorted((a, b)))
            if edge not in edge_set:
                edge_set.add(edge)
                edges.extend([verts[a] * rot_flip, verts[b] * rot_flip])

    return {
        'positions':      np.asarray(positions, 'f4'),
        'uvs':            np.asarray(uvs,       'f4'),
        'edges':          np.asarray(edges,     'f4'),
        'size':           (max_v - min_v).astype('f4'),
        'texture_bytes':  tex_img.tobytes(),
        'texture_size':   tex_img.size,
    }



def collect_car_model_vertices(car, model_data: Dict):
    """Return triangle and edge vertices for the car ``car``.

    ``model_data`` should be the dictionary returned by :func:`load_bbmodel`.
    The car's bounding-box dimensions are used to uniformly scale the model so
    it touches all sides of the box.  The car's ``body_offset`` keeps the model
    resting on the wheels.
    Returns a tuple ``(tri_vertices, edge_vertices)`` suitable for
    :func:`RenderContext.render_car_model`.
    """
    base_pos = model_data["positions"]
    base_uvs = model_data["uvs"]
    base_edges = model_data["edges"]
    size = model_data["size"]
    dims = car.dimensions
    scale = np.array(
        [
            dims["width"] / size[0],
            dims["height"] / size[1],
            dims["length"] / size[2],
        ],
        dtype="f4",
    )
    offset = np.array([0.0, car.body_offset, 0.0], dtype="f4")

    tri_verts: List[float] = []
    for p, uv in zip(base_pos, base_uvs):
        scaled = p * scale + offset
        world = car.body.pos + car.body.rot.rotate(scaled)
        tri_verts.extend(world.tolist() + uv.tolist())

    edge_color = list(WIRE_COLOR)
    edge_verts: List[float] = []
    for p in base_edges:
        scaled = p * scale + offset
        world = car.body.pos + car.body.rot.rotate(scaled)
        edge_verts.extend(world.tolist() + edge_color)

    return tri_verts, edge_verts