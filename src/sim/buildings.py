from __future__ import annotations

import colorsys
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Utility helpers -----------------------------------------------------------


def _hsl_color(h: float, s: float, l: float) -> np.ndarray:
    r, g, b = colorsys.hls_to_rgb(h % 1.0, np.clip(l, 0.0, 1.0), np.clip(s, 0.0, 1.0))
    return np.array([r, g, b, 1.0], dtype="f4")


def _jitter(color: np.ndarray, rng: np.random.Generator, amount: float = 0.04) -> np.ndarray:
    rgb = np.array(color[:3], dtype="f4")
    rgb = np.clip(rgb + rng.normal(0.0, amount, size=3), 0.0, 1.0)
    return np.array([rgb[0], rgb[1], rgb[2], color[3] if len(color) > 3 else 1.0], dtype="f4")


def _mix(c1: np.ndarray, c2: np.ndarray, t: float) -> np.ndarray:
    return np.array(c1 * (1.0 - t) + c2 * t, dtype="f4")


def _rotation_matrix_y(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype="f4")


def _rotate_xz(x: float, z: float, theta: float) -> Tuple[float, float]:
    c = math.cos(theta)
    s = math.sin(theta)
    return (x * c + z * s, -x * s + z * c)


def _add_triangle(
    out: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    p0: Sequence[float],
    p1: Sequence[float],
    p2: Sequence[float],
    colors: Sequence[np.ndarray],
) -> None:
    v0 = np.array(p0, dtype="f4")
    v1 = np.array(p1, dtype="f4")
    v2 = np.array(p2, dtype="f4")
    normal = np.cross(v1 - v0, v2 - v0)
    n = float(np.linalg.norm(normal))
    if n < 1e-6:
        return
    normal /= n
    for pos, col in zip((v0, v1, v2), colors):
        out.append((pos, normal, np.array(col, dtype="f4")))


def _add_quad(
    out: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    vertices: Sequence[Sequence[float]],
    colors: Optional[Sequence[np.ndarray]] = None,
) -> None:
    if colors is None:
        col = np.array([1.0, 1.0, 1.0, 1.0], dtype="f4")
        colors = (col, col, col, col)
    if len(colors) == 3:
        c0, c1, c2 = colors
        c3 = colors[2]
    else:
        c0, c1, c2, c3 = colors
    p0, p1, p2, p3 = vertices
    _add_triangle(out, p0, p1, p2, (c0, c1, c2))
    _add_triangle(out, p0, p2, p3, (c0, c2, c3))


def _face_vertices(
    face: str, width: float, depth: float, y0: float, y1: float
) -> List[Tuple[float, float, float]]:
    if face == "front":
        z = depth / 2.0
        return [
            (-width / 2.0, y0, z),
            (width / 2.0, y0, z),
            (width / 2.0, y1, z),
            (-width / 2.0, y1, z),
        ]
    if face == "back":
        z = -depth / 2.0
        return [
            (width / 2.0, y0, z),
            (-width / 2.0, y0, z),
            (-width / 2.0, y1, z),
            (width / 2.0, y1, z),
        ]
    if face == "right":
        x = width / 2.0
        return [
            (x, y0, depth / 2.0),
            (x, y0, -depth / 2.0),
            (x, y1, -depth / 2.0),
            (x, y1, depth / 2.0),
        ]
    if face == "left":
        x = -width / 2.0
        return [
            (x, y0, -depth / 2.0),
            (x, y0, depth / 2.0),
            (x, y1, depth / 2.0),
            (x, y1, -depth / 2.0),
        ]
    raise ValueError(face)


def _panel_vertices(
    face: str,
    u0: float,
    u1: float,
    y0: float,
    y1: float,
    width: float,
    depth: float,
    inset: float,
) -> List[Tuple[float, float, float]]:
    if face == "front":
        z = depth / 2.0 - inset
        return [(u0, y0, z), (u1, y0, z), (u1, y1, z), (u0, y1, z)]
    if face == "back":
        z = -depth / 2.0 + inset
        return [(u1, y0, z), (u0, y0, z), (u0, y1, z), (u1, y1, z)]
    if face == "right":
        x = width / 2.0 - inset
        return [(x, y0, u1), (x, y0, u0), (x, y1, u0), (x, y1, u1)]
    if face == "left":
        x = -width / 2.0 + inset
        return [(x, y0, u0), (x, y0, u1), (x, y1, u1), (x, y1, u0)]
    raise ValueError(face)


def _top_vertices(width: float, depth: float, y: float) -> List[Tuple[float, float, float]]:
    return [
        (-width / 2.0, y, -depth / 2.0),
        (width / 2.0, y, -depth / 2.0),
        (width / 2.0, y, depth / 2.0),
        (-width / 2.0, y, depth / 2.0),
    ]


def _ensure_list(color: np.ndarray, count: int = 4) -> List[np.ndarray]:
    return [np.array(color, dtype="f4") for _ in range(count)]


# ---------------------------------------------------------------------------
# Palette -------------------------------------------------------------------


def _random_palette(rng: np.random.Generator) -> Dict[str, List[np.ndarray]]:
    base_h = float(rng.uniform(0.0, 1.0))
    base_s = float(rng.uniform(0.18, 0.42))
    base_l = float(rng.uniform(0.32, 0.55))
    body = []
    for offset in (-0.04, 0.0, 0.05):
        h = (base_h + offset + rng.uniform(-0.03, 0.03)) % 1.0
        s = np.clip(base_s + rng.uniform(-0.06, 0.08), 0.12, 0.55)
        l = np.clip(base_l + rng.uniform(-0.10, 0.08), 0.25, 0.68)
        body.append(_hsl_color(h, s, l))
    accent = []
    accent_base = (base_h + 0.08 + rng.uniform(-0.12, 0.12)) % 1.0
    for _ in range(3):
        s = np.clip(base_s + rng.uniform(0.05, 0.18), 0.20, 0.65)
        l = np.clip(base_l + rng.uniform(-0.15, 0.05), 0.25, 0.60)
        accent.append(_hsl_color(accent_base + rng.uniform(-0.05, 0.05), s, l))
    roof = []
    roof_h = (base_h + 0.5 + rng.uniform(-0.08, 0.08)) % 1.0
    for _ in range(3):
        s = np.clip(base_s * rng.uniform(0.4, 0.8), 0.05, 0.35)
        l = np.clip(base_l * rng.uniform(0.55, 0.85), 0.18, 0.45)
        roof.append(_hsl_color(roof_h + rng.uniform(-0.04, 0.04), s, l))
    trim = []
    for _ in range(3):
        s = np.clip(base_s + rng.uniform(-0.08, 0.08), 0.10, 0.45)
        l = np.clip(base_l + rng.uniform(-0.18, 0.18), 0.20, 0.75)
        trim.append(_hsl_color(base_h + rng.uniform(-0.06, 0.06), s, l))
    glass = []
    glass_base = _hsl_color(
        0.55 + rng.uniform(-0.05, 0.05),
        0.24 + rng.uniform(-0.05, 0.05),
        0.52 + rng.uniform(-0.06, 0.06),
    )
    glass.append(glass_base)
    glass.append(_mix(glass_base, np.array([0.65, 0.70, 0.75, 1.0], dtype="f4"), 0.4))
    glass.append(_mix(glass_base, np.array([0.40, 0.45, 0.50, 1.0], dtype="f4"), 0.5))
    return {"body": body, "accent": accent, "roof": roof, "trim": trim, "glass": glass}


# ---------------------------------------------------------------------------
# Data classes ---------------------------------------------------------------


@dataclass
class BuildingSpec:
    type: str
    width: float
    depth: float
    floors: int
    floor_height: float
    foundation: float
    roof: str
    roof_height: float
    facades: Dict[str, Dict[str, float | int | bool | str]]
    features: Dict[str, float | int | bool | str] = field(default_factory=dict)
    max_slope: float = 2.5
    base_offset: float = 0.05
    noise_scale: float = 0.02


@dataclass
class BuildingInstance:
    spec: BuildingSpec
    center: Tuple[float, float]
    rotation: float
    base_height: float


# ---------------------------------------------------------------------------
# Specification factories ----------------------------------------------------


def _spec_house(rng: np.random.Generator) -> BuildingSpec:
    floors = int(rng.integers(1, 4))
    floor_height = float(rng.uniform(2.6, 3.15))
    width = float(rng.uniform(7.0, 13.0))
    depth = float(rng.uniform(6.0, 10.5))
    foundation = float(rng.uniform(0.25, 0.45))
    roof_type = rng.choice(["gable", "hip", "pyramid", "flat"], p=[0.45, 0.25, 0.2, 0.1])
    roof_height = float(rng.uniform(1.4, 2.5)) if roof_type != "flat" else float(rng.uniform(0.4, 0.8))
    facades = {
        "front": {"pattern": "house_front", "cols": int(rng.integers(2, 5)), "door": True},
        "back": {"pattern": "house_back", "cols": int(rng.integers(2, 4))},
        "left": {"pattern": "house_side"},
        "right": {"pattern": "house_side"},
    }
    features = {
        "porch": rng.random() < 0.35,
        "chimney": rng.random() < 0.25,
    }
    return BuildingSpec(
        type="house",
        width=width,
        depth=depth,
        floors=floors,
        floor_height=floor_height,
        foundation=foundation,
        roof=roof_type,
        roof_height=roof_height,
        facades=facades,
        features=features,
        max_slope=2.8,
        base_offset=0.03,
        noise_scale=0.015,
    )


def _spec_strip_mall(rng: np.random.Generator) -> BuildingSpec:
    floors = 1 if rng.random() < 0.8 else 2
    floor_height = float(rng.uniform(3.3, 3.8))
    width = float(rng.uniform(28.0, 46.0))
    depth = float(rng.uniform(16.0, 22.0))
    foundation = float(rng.uniform(0.2, 0.35))
    facades = {
        "front": {
            "pattern": "storefront",
            "cols": int(rng.integers(5, 9)),
            "door": True,
            "band": True,
        },
        "back": {"pattern": "panels", "cols": int(rng.integers(3, 6))},
        "left": {"pattern": "panels", "cols": int(rng.integers(2, 4))},
        "right": {"pattern": "panels", "cols": int(rng.integers(2, 4))},
    }
    features = {
        "service_doors": True,
        "roof_units": rng.random() < 0.65,
    }
    return BuildingSpec(
        type="strip_mall",
        width=width,
        depth=depth,
        floors=floors,
        floor_height=floor_height,
        foundation=foundation,
        roof="flat",
        roof_height=float(rng.uniform(0.6, 1.1)),
        facades=facades,
        features=features,
        max_slope=1.8,
        noise_scale=0.02,
    )


def _spec_store(rng: np.random.Generator) -> BuildingSpec:
    floors = 1
    floor_height = float(rng.uniform(3.2, 3.6))
    width = float(rng.uniform(15.0, 24.0))
    depth = float(rng.uniform(12.0, 18.0))
    foundation = float(rng.uniform(0.18, 0.35))
    facades = {
        "front": {"pattern": "windows", "cols": int(rng.integers(3, 6)), "door": True, "tall": True},
        "back": {"pattern": "panels", "cols": int(rng.integers(2, 4))},
        "left": {"pattern": "panels"},
        "right": {"pattern": "panels"},
    }
    features = {
        "band": rng.random() < 0.5,
        "front_canopy": rng.random() < 0.45,
    }
    return BuildingSpec(
        type="store",
        width=width,
        depth=depth,
        floors=floors,
        floor_height=floor_height,
        foundation=foundation,
        roof="flat",
        roof_height=float(rng.uniform(0.5, 0.9)),
        facades=facades,
        features=features,
        max_slope=2.0,
        noise_scale=0.018,
    )


def _spec_fast_food(rng: np.random.Generator) -> BuildingSpec:
    floors = 1
    floor_height = float(rng.uniform(3.0, 3.4))
    width = float(rng.uniform(12.0, 18.0))
    depth = float(rng.uniform(10.0, 14.0))
    foundation = float(rng.uniform(0.18, 0.30))
    roof_type = rng.choice(["pyramid", "gable", "flat"], p=[0.45, 0.25, 0.30])
    roof_height = float(rng.uniform(1.6, 2.4)) if roof_type != "flat" else float(rng.uniform(0.4, 0.7))
    facades = {
        "front": {"pattern": "storefront", "cols": int(rng.integers(3, 5)), "door": True, "band": True},
        "back": {"pattern": "panels", "cols": int(rng.integers(2, 4))},
        "left": {"pattern": "panels", "cols": int(rng.integers(2, 3))},
        "right": {"pattern": "panels", "cols": int(rng.integers(2, 3))},
    }
    features = {
        "drive_thru": rng.random() < 0.6,
        "sign_pylon": rng.random() < 0.3,
    }
    return BuildingSpec(
        type="fast_food",
        width=width,
        depth=depth,
        floors=floors,
        floor_height=floor_height,
        foundation=foundation,
        roof=roof_type,
        roof_height=roof_height,
        facades=facades,
        features=features,
        max_slope=2.2,
        noise_scale=0.018,
    )


def _spec_industrial(rng: np.random.Generator) -> BuildingSpec:
    floors = 1 if rng.random() < 0.8 else 2
    floor_height = float(rng.uniform(5.0, 6.5)) if floors == 1 else float(rng.uniform(4.2, 5.2))
    width = float(rng.uniform(34.0, 55.0))
    depth = float(rng.uniform(28.0, 44.0))
    foundation = float(rng.uniform(0.20, 0.35))
    facades = {
        "front": {"pattern": "industrial", "cols": int(rng.integers(3, 6)), "doors": int(rng.integers(1, 3))},
        "back": {"pattern": "panels", "cols": int(rng.integers(3, 6))},
        "left": {"pattern": "panels", "cols": int(rng.integers(2, 4))},
        "right": {"pattern": "panels", "cols": int(rng.integers(2, 4))},
    }
    features = {
        "vents": rng.random() < 0.7,
        "roof_units": True,
    }
    return BuildingSpec(
        type="industrial",
        width=width,
        depth=depth,
        floors=floors,
        floor_height=floor_height,
        foundation=foundation,
        roof="flat",
        roof_height=float(rng.uniform(0.8, 1.4)),
        facades=facades,
        features=features,
        max_slope=1.6,
        noise_scale=0.025,
    )


def _spec_office(rng: np.random.Generator) -> BuildingSpec:
    floors = int(rng.integers(4, 12))
    floor_height = float(rng.uniform(3.0, 3.4))
    width = float(rng.uniform(18.0, 30.0))
    depth = float(rng.uniform(16.0, 26.0))
    foundation = float(rng.uniform(0.18, 0.30))
    facades = {
        "front": {"pattern": "glass_grid", "cols": int(rng.integers(4, 7))},
        "back": {"pattern": "glass_grid", "cols": int(rng.integers(4, 7))},
        "left": {"pattern": "glass_grid", "cols": int(rng.integers(3, 6))},
        "right": {"pattern": "glass_grid", "cols": int(rng.integers(3, 6))},
    }
    features = {
        "crown": rng.random() < 0.55,
        "lobby": True,
    }
    return BuildingSpec(
        type="office",
        width=width,
        depth=depth,
        floors=floors,
        floor_height=floor_height,
        foundation=foundation,
        roof="flat",
        roof_height=float(rng.uniform(0.8, 1.6)),
        facades=facades,
        features=features,
        max_slope=1.8,
        noise_scale=0.02,
    )


def _spec_skyscraper(rng: np.random.Generator) -> BuildingSpec:
    floors = int(rng.integers(22, 61))
    floor_height = float(rng.uniform(3.0, 3.4))
    width = float(rng.uniform(14.0, 24.0))
    depth = float(rng.uniform(14.0, 24.0))
    foundation = float(rng.uniform(0.3, 0.45))
    facades = {
        "front": {"pattern": "glass_grid", "cols": int(rng.integers(4, 8)), "dense": True},
        "back": {"pattern": "glass_grid", "cols": int(rng.integers(4, 8)), "dense": True},
        "left": {"pattern": "glass_grid", "cols": int(rng.integers(3, 6)), "dense": True},
        "right": {"pattern": "glass_grid", "cols": int(rng.integers(3, 6)), "dense": True},
    }
    features = {
        "crown": True,
        "mechanical": True,
    }
    return BuildingSpec(
        type="skyscraper",
        width=width,
        depth=depth,
        floors=floors,
        floor_height=floor_height,
        foundation=foundation,
        roof="flat",
        roof_height=float(rng.uniform(1.8, 3.5)),
        facades=facades,
        features=features,
        max_slope=1.4,
        noise_scale=0.018,
    )


def _spec_pyramid(rng: np.random.Generator) -> BuildingSpec:
    base = float(rng.uniform(36.0, 64.0))
    height = float(rng.uniform(18.0, 34.0))
    steps = int(rng.integers(4, 8))
    top_ratio = float(rng.uniform(0.12, 0.22))
    facades: Dict[str, Dict[str, float | int | bool | str]] = {}
    features = {
        "pyramid_height": height,
        "steps": steps,
        "top_ratio": top_ratio,
        "temple": rng.random() < 0.45,
    }
    return BuildingSpec(
        type="pyramid",
        width=base,
        depth=base * float(rng.uniform(0.85, 1.15)),
        floors=0,
        floor_height=0.0,
        foundation=0.0,
        roof="pyramid",
        roof_height=height,
        facades=facades,
        features=features,
        max_slope=1.2,
        base_offset=0.0,
        noise_scale=0.03,
    )


def _choose_spec(category: str, rng: np.random.Generator) -> BuildingSpec:
    if category == "residential":
        return _spec_house(rng)
    if category == "roadside":
        options = [
            _spec_strip_mall,
            _spec_store,
            _spec_fast_food,
            _spec_office,
            _spec_industrial,
        ]
        weights = np.array([0.18, 0.24, 0.18, 0.18, 0.22], dtype=float)
        weights /= weights.sum()
        choice = rng.choice(len(options), p=weights)
        return options[choice](rng)
    if category == "tall":
        return _spec_skyscraper(rng)
    if category == "landmark":
        return _spec_pyramid(rng)
    if category == "industrial":
        return _spec_industrial(rng)
    return _spec_store(rng)


# ---------------------------------------------------------------------------
# Geometry builders ----------------------------------------------------------


def _select_colors(
    palette: Dict[str, List[np.ndarray]], spec: BuildingSpec, rng: np.random.Generator
) -> Dict[str, np.ndarray]:
    body = _jitter(rng.choice(palette["body"]), rng, 0.05)
    accent = _jitter(rng.choice(palette["accent"]), rng, 0.05)
    roof = _jitter(rng.choice(palette["roof"]), rng, 0.04)
    trim = _jitter(rng.choice(palette["trim"]), rng, 0.04)
    window = _jitter(rng.choice(palette["glass"]), rng, 0.03)
    if spec.type in {"office", "skyscraper"}:
        window = _mix(window, np.array([0.55, 0.62, 0.72, 1.0], dtype="f4"), 0.4)
        body = _mix(body, window, 0.25 if spec.type == "office" else 0.45)
        accent = _mix(accent, window, 0.2)
    foundation = _jitter(_mix(body, trim, 0.45), rng, 0.03)
    door = _jitter(_mix(accent, trim, 0.5), rng, 0.04)
    metal = _jitter(_mix(accent, np.array([0.6, 0.6, 0.6, 1.0], dtype="f4"), 0.6), rng, 0.03)
    return {
        "body": body,
        "accent": accent,
        "roof": roof,
        "trim": trim,
        "window": window,
        "foundation": foundation,
        "door": door,
        "metal": metal,
    }


def _facade_add_band(
    tris: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    face: str,
    dims: Dict[str, float],
    color: np.ndarray,
    y0: float,
    y1: float,
    inset: float = 0.02,
) -> None:
    width = dims["width"]
    depth = dims["depth"]
    verts = _panel_vertices(
        face,
        - (width if face in {"front", "back"} else depth) / 2.0,
        (width if face in {"front", "back"} else depth) / 2.0,
        y0,
        y1,
        width,
        depth,
        inset,
    )
    _add_quad(tris, verts, _ensure_list(color))


def _facade_add_columns(
    tris: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    face: str,
    dims: Dict[str, float],
    color: np.ndarray,
    count: int,
    width_ratio: float,
    inset: float,
    base: float,
    top: float,
    rng: np.random.Generator,
) -> None:
    if count <= 0:
        return
    span = dims["width"] if face in {"front", "back"} else dims["depth"]
    usable = span * 0.9
    offset = -usable / 2.0
    step = usable / max(count, 1)
    col_w = step * width_ratio
    for i in range(count):
        u0 = offset + i * step
        u1 = u0 + col_w
        verts = _panel_vertices(face, u0, u1, base, top, dims["width"], dims["depth"], inset)
        _add_quad(tris, verts, _ensure_list(_jitter(color, rng, 0.02)))


def _facade_windows(
    tris: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    face: str,
    spec: BuildingSpec,
    dims: Dict[str, float],
    colors: Dict[str, np.ndarray],
    style: Dict[str, float | int | bool | str],
    rng: np.random.Generator,
    start_floor: int = 0,
) -> None:
    floors = spec.floors
    if floors <= 0:
        return
    cols = int(style.get("cols", max(2, int(dims["width"] / 3.0))))
    margin_ratio = float(style.get("margin", 0.08))
    vertical_margin = float(style.get("vertical_margin", 0.12))
    window_ratio = float(style.get("window_ratio", 0.68))
    inset = float(style.get("inset", 0.06 if face in {"front", "back"} else 0.08))
    span = dims["width"] if face in {"front", "back"} else dims["depth"]
    margin = span * margin_ratio
    usable = span - 2.0 * margin
    if usable <= 0:
        usable = span * 0.8
    col_width = usable / max(cols, 1)
    gap = col_width * float(style.get("gap_ratio", 0.18))
    panel_width = col_width - gap
    door_region = None
    if bool(style.get("door")) and face == "front":
        door_width = float(style.get("door_width", min(1.6, span * 0.22)))
        door_center = float(style.get("door_center", 0.0))
        door_region = (door_center - door_width * 0.55, door_center + door_width * 0.55)
        door_height = dims["foundation"] + spec.floor_height * 0.85
        verts = _panel_vertices(
            face,
            door_center - door_width / 2.0,
            door_center + door_width / 2.0,
            dims["foundation"],
            door_height,
            dims["width"],
            dims["depth"],
            inset * 0.35,
        )
        _add_quad(tris, verts, _ensure_list(_jitter(colors["door"], rng, 0.03)))
    for row in range(start_floor, floors):
        floor_base = dims["foundation"] + row * spec.floor_height
        floor_top = floor_base + spec.floor_height
        y0 = floor_base + spec.floor_height * vertical_margin
        y1 = min(floor_top - spec.floor_height * 0.12, y0 + spec.floor_height * window_ratio)
        for col in range(cols):
            u0 = -span / 2.0 + margin + col * col_width + gap * 0.5
            u1 = u0 + panel_width
            if door_region and row == 0:
                if not (u1 < door_region[0] or u0 > door_region[1]):
                    continue
            verts = _panel_vertices(face, u0, u1, y0, y1, dims["width"], dims["depth"], inset)
            color = _jitter(colors["window"], rng, 0.02)
            if style.get("warm_glow") and row == 0:
                glow = np.array([1.0, 0.82, 0.55, 1.0], dtype="f4")
                color = _mix(color, glow, 0.35)
            _add_quad(tris, verts, _ensure_list(color))


def _facade_house_front(
    tris: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    face: str,
    spec: BuildingSpec,
    dims: Dict[str, float],
    colors: Dict[str, np.ndarray],
    style: Dict[str, float | int | bool | str],
    rng: np.random.Generator,
) -> None:
    stripes = int(spec.floors * 6 + rng.integers(2, 5))
    span = dims["width"]
    for i in range(stripes):
        t0 = i / stripes
        t1 = (i + 1) / stripes
        y0 = dims["foundation"] + (dims["height"] - dims["foundation"]) * t0
        y1 = dims["foundation"] + (dims["height"] - dims["foundation"]) * t1
        inset = -0.015 if i % 2 == 0 else -0.01
        verts = _panel_vertices(
            face,
            -span / 2.0,
            span / 2.0,
            y0,
            y1,
            dims["width"],
            dims["depth"],
            inset,
        )
        color = _jitter(_mix(colors["body"], colors["trim"], 0.15 + 0.2 * (i % 2)), rng, 0.015)
        _add_quad(tris, verts, _ensure_list(color))
    _facade_windows(tris, face, spec, dims, colors, style, rng)
    if style.get("door"):
        porch_height = dims["foundation"] + spec.floor_height * 0.25
        porch_depth = float(style.get("porch_depth", 1.2))
        verts = [
            (-1.2, dims["foundation"], dims["depth"] / 2.0 + porch_depth),
            (1.2, dims["foundation"], dims["depth"] / 2.0 + porch_depth),
            (1.2, porch_height, dims["depth"] / 2.0 + porch_depth),
            (-1.2, porch_height, dims["depth"] / 2.0 + porch_depth),
        ]
        _add_quad(tris, verts, _ensure_list(_mix(colors["foundation"], colors["trim"], 0.3)))


def _facade_house_back(
    tris: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    face: str,
    spec: BuildingSpec,
    dims: Dict[str, float],
    colors: Dict[str, np.ndarray],
    style: Dict[str, float | int | bool | str],
    rng: np.random.Generator,
) -> None:
    stripes = int(spec.floors * 6 + rng.integers(2, 5))
    span = dims["width"] if face in {"front", "back"} else dims["depth"]
    for i in range(stripes):
        y0 = dims["foundation"] + (dims["height"] - dims["foundation"]) * (i / stripes)
        y1 = dims["foundation"] + (dims["height"] - dims["foundation"]) * ((i + 1) / stripes)
        inset = -0.012 if i % 2 == 0 else -0.007
        verts = _panel_vertices(face, -span / 2.0, span / 2.0, y0, y1, dims["width"], dims["depth"], inset)
        color = _jitter(_mix(colors["body"], colors["trim"], 0.1 + 0.25 * (i % 2)), rng, 0.012)
        _add_quad(tris, verts, _ensure_list(color))
    _facade_windows(tris, face, spec, dims, colors, style, rng)


def _facade_house_side(
    tris: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    face: str,
    spec: BuildingSpec,
    dims: Dict[str, float],
    colors: Dict[str, np.ndarray],
    style: Dict[str, float | int | bool | str],
    rng: np.random.Generator,
) -> None:
    stripes = int(spec.floors * 5 + rng.integers(1, 3))
    span = dims["depth"]
    for i in range(stripes):
        y0 = dims["foundation"] + (dims["height"] - dims["foundation"]) * (i / stripes)
        y1 = dims["foundation"] + (dims["height"] - dims["foundation"]) * ((i + 1) / stripes)
        inset = -0.01 if i % 2 == 0 else -0.006
        verts = _panel_vertices(face, -span / 2.0, span / 2.0, y0, y1, dims["width"], dims["depth"], inset)
        color = _jitter(_mix(colors["body"], colors["trim"], 0.12 + 0.25 * (i % 2)), rng, 0.01)
        _add_quad(tris, verts, _ensure_list(color))
    style_local = dict(style)
    style_local.setdefault("cols", 2)
    style_local.setdefault("inset", 0.05)
    _facade_windows(tris, face, spec, dims, colors, style_local, rng)


def _facade_panels(
    tris: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    face: str,
    spec: BuildingSpec,
    dims: Dict[str, float],
    colors: Dict[str, np.ndarray],
    style: Dict[str, float | int | bool | str],
    rng: np.random.Generator,
) -> None:
    cols = int(style.get("cols", max(2, int(dims["width"] / 4.5))))
    span = dims["width"] if face in {"front", "back"} else dims["depth"]
    margin = span * float(style.get("margin", 0.08))
    usable = span - 2.0 * margin
    if usable <= 0:
        usable = span * 0.85
    col_width = usable / max(cols, 1)
    gap = col_width * float(style.get("gap_ratio", 0.1))
    panel_width = col_width - gap
    base = dims["foundation"]
    top = dims["height"]
    inset = float(style.get("inset", 0.05 if face in {"front", "back"} else 0.06))
    for i in range(cols):
        u0 = -span / 2.0 + margin + i * col_width + gap * 0.5
        u1 = u0 + panel_width
        verts = _panel_vertices(face, u0, u1, base, top, dims["width"], dims["depth"], inset)
        t = rng.uniform(0.0, 1.0)
        color = _mix(colors["body"], colors["trim"], 0.15 + 0.35 * t)
        color = _jitter(color, rng, 0.025)
        _add_quad(tris, verts, _ensure_list(color))


def _facade_storefront(
    tris: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    face: str,
    spec: BuildingSpec,
    dims: Dict[str, float],
    colors: Dict[str, np.ndarray],
    style: Dict[str, float | int | bool | str],
    rng: np.random.Generator,
) -> None:
    cols = int(style.get("cols", 5))
    span = dims["width"]
    margin = span * 0.08
    usable = span - 2.0 * margin
    step = usable / max(cols, 1)
    column_color = _mix(colors["accent"], colors["trim"], 0.3)
    inset = float(style.get("inset", 0.05))
    for i in range(cols):
        u0 = -span / 2.0 + margin + i * step
        u1 = u0 + step * 0.88
        y0 = dims["foundation"] + 0.12
        y1 = dims["foundation"] + spec.floor_height * 0.92
        verts = _panel_vertices(face, u0, u1, y0, y1, dims["width"], dims["depth"], inset)
        color = _jitter(colors["window"], rng, 0.025)
        _add_quad(tris, verts, _ensure_list(color))
        if bool(style.get("door")):
            door_width = step * 0.32
            door_center = (u0 + u1) / 2.0
            door_bottom = dims["foundation"]
            door_top = door_bottom + spec.floor_height * 0.78
            verts = _panel_vertices(
                face,
                door_center - door_width / 2.0,
                door_center + door_width / 2.0,
                door_bottom,
                door_top,
                dims["width"],
                dims["depth"],
                inset * 0.4,
            )
            _add_quad(tris, verts, _ensure_list(_mix(colors["door"], colors["accent"], 0.5)))
        column_width = step * 0.12
        verts = _panel_vertices(
            face,
            u0 - column_width * 0.45,
            u0,
            dims["foundation"],
            dims["height"],
            dims["width"],
            dims["depth"],
            inset * 0.2,
        )
        _add_quad(tris, verts, _ensure_list(_jitter(column_color, rng, 0.02)))
    if bool(style.get("band")):
        band_y0 = dims["foundation"] + spec.floor_height * 0.95
        band_y1 = band_y0 + spec.floor_height * 0.15
        _facade_add_band(tris, face, dims, _mix(colors["accent"], colors["trim"], 0.5), band_y0, band_y1, 0.0)
    if bool(spec.features.get("front_canopy")):
        canopy_height = dims["foundation"] + spec.floor_height * 0.75
        verts = [
            (-span / 2.0, canopy_height, dims["depth"] / 2.0 + 1.5),
            (span / 2.0, canopy_height, dims["depth"] / 2.0 + 1.5),
            (span / 2.0, canopy_height, dims["depth"] / 2.0),
            (-span / 2.0, canopy_height, dims["depth"] / 2.0),
        ]
        _add_quad(tris, verts, _ensure_list(_mix(colors["accent"], colors["trim"], 0.3)))


def _facade_industrial(
    tris: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    face: str,
    spec: BuildingSpec,
    dims: Dict[str, float],
    colors: Dict[str, np.ndarray],
    style: Dict[str, float | int | bool | str],
    rng: np.random.Generator,
) -> None:
    cols = int(style.get("cols", 4))
    doors = int(style.get("doors", max(1, cols // 2)))
    door_indices = rng.choice(cols, size=min(doors, cols), replace=False)
    span = dims["width"]
    margin = span * 0.08
    usable = span - 2.0 * margin
    step = usable / max(cols, 1)
    inset = float(style.get("inset", 0.05))
    for i in range(cols):
        u0 = -span / 2.0 + margin + i * step
        u1 = u0 + step * 0.9
        base = dims["foundation"]
        top = dims["height"]
        verts = _panel_vertices(face, u0, u1, base, top, dims["width"], dims["depth"], inset)
        color = _jitter(_mix(colors["body"], colors["trim"], 0.25 + 0.35 * rng.random()), rng, 0.02)
        _add_quad(tris, verts, _ensure_list(color))
        if i in door_indices:
            door_height = dims["foundation"] + spec.floor_height * (1.1 if spec.floors == 1 else 0.85)
            verts = _panel_vertices(
                face,
                u0 + step * 0.18,
                u0 + step * 0.72,
                dims["foundation"],
                door_height,
                dims["width"],
                dims["depth"],
                inset * 0.5,
            )
            door_col = _mix(colors["metal"], colors["trim"], 0.4)
            _add_quad(tris, verts, _ensure_list(_jitter(door_col, rng, 0.02)))
    clerestory_base = dims["foundation"] + spec.floor_height * (1.05 if spec.floors == 1 else 1.8)
    clerestory_top = min(dims["height"], clerestory_base + spec.floor_height * 0.5)
    _facade_windows(
        tris,
        face,
        spec,
        dims,
        colors,
        {"cols": cols * 2, "inset": inset * 0.3},
        rng,
        start_floor=1 if spec.floors > 1 else 0,
    )
    if spec.features.get("vents"):
        _facade_add_columns(
            tris,
            face,
            dims,
            _mix(colors["metal"], colors["trim"], 0.6),
            max(1, cols // 2),
            0.06,
            inset * 0.1,
            clerestory_top + 0.2,
            clerestory_top + 0.8,
            rng,
        )


def _facade_glass_grid(
    tris: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    face: str,
    spec: BuildingSpec,
    dims: Dict[str, float],
    colors: Dict[str, np.ndarray],
    style: Dict[str, float | int | bool | str],
    rng: np.random.Generator,
) -> None:
    cols = int(style.get("cols", max(3, int(dims["width"] / 3.0))))
    dense = bool(style.get("dense"))
    rows = spec.floors * (2 if dense else 1)
    span = dims["width"] if face in {"front", "back"} else dims["depth"]
    inset = float(style.get("inset", 0.05))
    margin = span * 0.08
    usable = span - 2.0 * margin
    if usable <= 0:
        usable = span * 0.9
    col_width = usable / max(cols, 1)
    gap = col_width * 0.12
    panel_width = col_width - gap
    height_span = dims["height"] - dims["foundation"]
    row_height = height_span / max(rows, 1)
    for r in range(rows):
        y0 = dims["foundation"] + r * row_height + row_height * 0.08
        y1 = dims["foundation"] + (r + 1) * row_height - row_height * 0.08
        for c in range(cols):
            u0 = -span / 2.0 + margin + c * col_width + gap * 0.5
            u1 = u0 + panel_width
            verts = _panel_vertices(face, u0, u1, y0, y1, dims["width"], dims["depth"], inset)
            color = _jitter(_mix(colors["window"], colors["body"], 0.1 + 0.15 * rng.random()), rng, 0.015)
            _add_quad(tris, verts, _ensure_list(color))
    mullion_color = _mix(colors["body"], colors["trim"], 0.5)
    _facade_add_columns(
        tris,
        face,
        dims,
        mullion_color,
        cols + 1,
        0.015,
        inset * 0.1,
        dims["foundation"],
        dims["height"],
        rng,
    )


def _apply_facade(
    tris: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    face: str,
    spec: BuildingSpec,
    dims: Dict[str, float],
    colors: Dict[str, np.ndarray],
    style: Optional[Dict[str, float | int | bool | str]],
    rng: np.random.Generator,
) -> None:
    if not style:
        return
    pattern = style.get("pattern", "plain")
    if pattern == "house_front":
        _facade_house_front(tris, face, spec, dims, colors, style, rng)
    elif pattern == "house_back":
        _facade_house_back(tris, face, spec, dims, colors, style, rng)
    elif pattern == "house_side":
        _facade_house_side(tris, face, spec, dims, colors, style, rng)
    elif pattern == "windows":
        _facade_windows(tris, face, spec, dims, colors, style, rng)
    elif pattern == "storefront":
        _facade_storefront(tris, face, spec, dims, colors, style, rng)
    elif pattern == "panels":
        _facade_panels(tris, face, spec, dims, colors, style, rng)
    elif pattern == "industrial":
        _facade_industrial(tris, face, spec, dims, colors, style, rng)
    elif pattern == "glass_grid":
        _facade_glass_grid(tris, face, spec, dims, colors, style, rng)
    else:
        _facade_panels(tris, face, spec, dims, colors, {"cols": 2}, rng)


def _build_roof(
    tris: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    spec: BuildingSpec,
    dims: Dict[str, float],
    colors: Dict[str, np.ndarray],
    rng: np.random.Generator,
) -> None:
    height = dims["height"]
    width = dims["width"]
    depth = dims["depth"]
    roof_color = _jitter(colors["roof"], rng, 0.02)
    trim_color = _jitter(_mix(colors["roof"], colors["trim"], 0.4), rng, 0.02)
    if spec.type == "pyramid":
        return
    if spec.roof == "flat":
        top = _top_vertices(width, depth, height + 0.02)
        _add_quad(tris, top, _ensure_list(roof_color))
        parapet = min(spec.roof_height, 0.7)
        if parapet > 0.05:
            for face in ("front", "back", "left", "right"):
                verts = _face_vertices(face, width, depth, height, height + parapet)
                _add_quad(tris, verts, _ensure_list(trim_color))
        if spec.features.get("crown"):
            crown_width = width * 0.6
            crown_depth = depth * 0.6
            crown_height = height + parapet + min(spec.roof_height * 0.8, 1.6)
            base_height = height + parapet
            top = _top_vertices(crown_width, crown_depth, crown_height)
            _add_quad(tris, top, _ensure_list(_mix(roof_color, colors["trim"], 0.3)))
            for face in ("front", "back", "left", "right"):
                verts = _face_vertices(face, crown_width, crown_depth, base_height, crown_height)
                _add_quad(tris, verts, _ensure_list(_mix(trim_color, colors["window"], 0.3)))
        if spec.features.get("mechanical"):
            mech_width = width * 0.35
            mech_depth = depth * 0.35
            base = height + parapet + 0.2
            top_h = base + 1.2
            top = _top_vertices(mech_width, mech_depth, top_h)
            _add_quad(tris, top, _ensure_list(_mix(colors["metal"], colors["trim"], 0.5)))
            for face in ("front", "back", "left", "right"):
                verts = _face_vertices(face, mech_width, mech_depth, base, top_h)
                _add_quad(tris, verts, _ensure_list(_mix(colors["metal"], colors["trim"], 0.4)))
        return
    if spec.roof in {"pyramid", "hip"}:
        apex = np.array([0.0, height + spec.roof_height, 0.0], dtype="f4")
        corners = [
            (-width / 2.0, height, depth / 2.0),
            (width / 2.0, height, depth / 2.0),
            (width / 2.0, height, -depth / 2.0),
            (-width / 2.0, height, -depth / 2.0),
        ]
        for i in range(4):
            p0 = corners[i]
            p1 = corners[(i + 1) % 4]
            _add_triangle(tris, p0, p1, apex, (_ensure_list(roof_color, 1) * 3))
        return
    if spec.roof == "gable":
        peak_front = np.array([0.0, height + spec.roof_height, depth / 2.0], dtype="f4")
        peak_back = np.array([0.0, height + spec.roof_height, -depth / 2.0], dtype="f4")
        left_front = np.array([-width / 2.0, height, depth / 2.0], dtype="f4")
        left_back = np.array([-width / 2.0, height, -depth / 2.0], dtype="f4")
        right_front = np.array([width / 2.0, height, depth / 2.0], dtype="f4")
        right_back = np.array([width / 2.0, height, -depth / 2.0], dtype="f4")
        _add_triangle(tris, left_front, right_front, peak_front, _ensure_list(trim_color, 3))
        _add_triangle(tris, right_back, left_back, peak_back, _ensure_list(trim_color, 3))
        _add_triangle(tris, left_front, left_back, peak_back, _ensure_list(roof_color, 3))
        _add_triangle(tris, left_front, peak_back, peak_front, _ensure_list(roof_color, 3))
        _add_triangle(tris, right_front, peak_front, peak_back, _ensure_list(roof_color, 3))
        _add_triangle(tris, right_front, peak_back, right_back, _ensure_list(roof_color, 3))
        if spec.features.get("chimney"):
            chimney_width = min(0.8, width * 0.12)
            chimney_depth = min(0.8, depth * 0.12)
            base = height + spec.roof_height * 0.4
            top = base + 1.0
            x = width * 0.15 if rng.random() < 0.5 else -width * 0.15
            z = depth * -0.15
            verts = [
                (x - chimney_width / 2, base, z - chimney_depth / 2),
                (x + chimney_width / 2, base, z - chimney_depth / 2),
                (x + chimney_width / 2, top, z - chimney_depth / 2),
                (x - chimney_width / 2, top, z - chimney_depth / 2),
            ]
            _add_quad(tris, verts, _ensure_list(_mix(colors["trim"], colors["roof"], 0.4)))
        return


def _build_rectangular(
    spec: BuildingSpec,
    colors: Dict[str, np.ndarray],
    rng: np.random.Generator,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    tris: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    height = spec.foundation + spec.floors * spec.floor_height
    dims = {
        "width": spec.width,
        "depth": spec.depth,
        "foundation": spec.foundation,
        "height": height,
        "floors": spec.floors,
        "floor_height": spec.floor_height,
    }
    if spec.foundation > 0.01:
        for face in ("front", "back", "left", "right"):
            verts = _face_vertices(face, spec.width, spec.depth, 0.0, spec.foundation)
            _add_quad(tris, verts, _ensure_list(_jitter(colors["foundation"], rng, 0.015)))
    for face in ("front", "back", "left", "right"):
        verts = _face_vertices(face, spec.width, spec.depth, spec.foundation, height)
        base_color = _jitter(colors["body"], rng, 0.02)
        _add_quad(tris, verts, _ensure_list(base_color))
        style = spec.facades.get(face) or spec.facades.get("default")
        _apply_facade(tris, face, spec, dims, colors, style, rng)
    if spec.features.get("porch"):
        porch_depth = 2.2
        porch_height = spec.foundation + 0.2
        verts = [
            (-spec.width / 3.0, porch_height, spec.depth / 2.0 + porch_depth),
            (spec.width / 3.0, porch_height, spec.depth / 2.0 + porch_depth),
            (spec.width / 3.0, porch_height, spec.depth / 2.0),
            (-spec.width / 3.0, porch_height, spec.depth / 2.0),
        ]
        _add_quad(tris, verts, _ensure_list(_mix(colors["foundation"], colors["trim"], 0.4)))
    _build_roof(tris, spec, dims, colors, rng)
    top = _top_vertices(spec.width, spec.depth, 0.0)
    _add_triangle(tris, top[0], top[1], top[2], _ensure_list(colors["foundation"], 3))
    _add_triangle(tris, top[0], top[2], top[3], _ensure_list(colors["foundation"], 3))
    return tris


def _build_pyramid_geometry(
    spec: BuildingSpec,
    colors: Dict[str, np.ndarray],
    rng: np.random.Generator,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    tris: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    steps = max(1, int(spec.features.get("steps", 5)))
    height = float(spec.features.get("pyramid_height", spec.roof_height))
    top_ratio = float(spec.features.get("top_ratio", 0.18))
    base_width = spec.width
    base_depth = spec.depth
    top_width = base_width * top_ratio
    top_depth = base_depth * top_ratio
    step_height = height / steps
    for i in range(steps):
        y0 = i * step_height
        y1 = (i + 1) * step_height
        t0 = i / steps
        t1 = (i + 1) / steps
        w0 = base_width - (base_width - top_width) * t0
        w1 = base_width - (base_width - top_width) * t1
        d0 = base_depth - (base_depth - top_depth) * t0
        d1 = base_depth - (base_depth - top_depth) * t1
        color = _jitter(_mix(colors["body"], colors["trim"], 0.15 + 0.25 * (i % 2)), rng, 0.015)
        front = [(-w0 / 2.0, y0, d0 / 2.0), (w0 / 2.0, y0, d0 / 2.0), (w1 / 2.0, y1, d1 / 2.0), (-w1 / 2.0, y1, d1 / 2.0)]
        back = [(w0 / 2.0, y0, -d0 / 2.0), (-w0 / 2.0, y0, -d0 / 2.0), (-w1 / 2.0, y1, -d1 / 2.0), (w1 / 2.0, y1, -d1 / 2.0)]
        left = [(-w0 / 2.0, y0, -d0 / 2.0), (-w0 / 2.0, y0, d0 / 2.0), (-w1 / 2.0, y1, d1 / 2.0), (-w1 / 2.0, y1, -d1 / 2.0)]
        right = [(w0 / 2.0, y0, d0 / 2.0), (w0 / 2.0, y0, -d0 / 2.0), (w1 / 2.0, y1, -d1 / 2.0), (w1 / 2.0, y1, d1 / 2.0)]
        for quad in (front, back, left, right):
            _add_quad(tris, quad, _ensure_list(color))
    top = _top_vertices(top_width, top_depth, height)
    _add_triangle(tris, top[0], top[1], top[2], _ensure_list(colors["roof"], 3))
    _add_triangle(tris, top[0], top[2], top[3], _ensure_list(colors["roof"], 3))
    if spec.features.get("temple"):
        temple_width = top_width * 0.6
        temple_depth = top_depth * 0.6
        base = height
        top_h = height + 4.0
        verts_top = _top_vertices(temple_width, temple_depth, top_h)
        _add_triangle(tris, verts_top[0], verts_top[1], verts_top[2], _ensure_list(_mix(colors["trim"], colors["roof"], 0.4), 3))
        _add_triangle(tris, verts_top[0], verts_top[2], verts_top[3], _ensure_list(_mix(colors["trim"], colors["roof"], 0.4), 3))
        for face in ("front", "back", "left", "right"):
            verts = _face_vertices(face, temple_width, temple_depth, base, top_h)
            _add_quad(tris, verts, _ensure_list(_mix(colors["trim"], colors["foundation"], 0.5)))
    base_plate = _top_vertices(base_width * 1.2, base_depth * 1.2, 0.0)
    _add_triangle(tris, base_plate[0], base_plate[1], base_plate[2], _ensure_list(colors["foundation"], 3))
    _add_triangle(tris, base_plate[0], base_plate[2], base_plate[3], _ensure_list(colors["foundation"], 3))
    return tris


def _assemble_vertices(
    tris: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    position: Tuple[float, float, float],
    rotation: float,
) -> np.ndarray:
    if not tris:
        return np.zeros((0, 10), dtype="f4")
    rot = _rotation_matrix_y(rotation)
    pos = np.array(position, dtype="f4")
    out = np.zeros((len(tris), 10), dtype="f4")
    for i, (p, n, c) in enumerate(tris):
        rp = rot @ p + pos
        rn = rot @ n
        norm = float(np.linalg.norm(rn))
        if norm > 1e-6:
            rn = rn / norm
        out[i, 0:3] = rp
        out[i, 3:6] = rn
        out[i, 6:10] = c
    return out


def _polyline_lengths(points: Sequence[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray, float]:
    pts = np.asarray(points, dtype="f4")
    diffs = np.diff(pts, axis=0)
    lengths = np.linalg.norm(diffs, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    return lengths, cumulative, float(cumulative[-1]) if len(cumulative) else 0.0


def _point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-6:
        return float(np.linalg.norm(p - a))
    t = np.clip(float(np.dot(p - a, ab)) / denom, 0.0, 1.0)
    closest = a + t * ab
    return float(np.linalg.norm(p - closest))


def _distance_to_path(point: Tuple[float, float], path: Sequence[Tuple[float, float]]) -> float:
    if len(path) < 2:
        return float("inf")
    p = np.array(point, dtype="f4")
    pts = np.asarray(path, dtype="f4")
    min_dist = float("inf")
    for i in range(len(pts) - 1):
        d = _point_segment_distance(p, pts[i], pts[i + 1])
        if d < min_dist:
            min_dist = d
    return min_dist


def _footprint_heights(
    terrain,
    spec: BuildingSpec,
    center: Tuple[float, float],
    rotation: float,
) -> Tuple[Optional[float], List[float]]:
    cx, cz = center
    corners = [
        (-spec.width / 2.0, -spec.depth / 2.0),
        (spec.width / 2.0, -spec.depth / 2.0),
        (spec.width / 2.0, spec.depth / 2.0),
        (-spec.width / 2.0, spec.depth / 2.0),
    ]
    heights: List[float] = []
    for x, z in corners:
        gx, gz = _rotate_xz(x, z, rotation)
        gx += cx
        gz += cz
        h = float(terrain.get_height(gx, gz))
        if not np.isfinite(h):
            return None, []
        heights.append(h)
    center_height = float(terrain.get_height(cx, cz))
    if not np.isfinite(center_height):
        return None, []
    heights.append(center_height)
    base = min(heights)
    return base, heights


def _bounds_ok(
    terrain,
    center: Tuple[float, float],
    radius: float,
) -> bool:
    cx, cz = center
    margin = radius + 3.0
    return (
        margin < cx < terrain.width - margin
        and margin < cz < terrain.height - margin
    )


def _overlaps(
    occupied: List[Tuple[float, float, float]],
    center: Tuple[float, float],
    radius: float,
    padding: float,
) -> bool:
    cx, cz = center
    for ox, oz, orad in occupied:
        if (cx - ox) ** 2 + (cz - oz) ** 2 < (radius + orad + padding) ** 2:
            return True
    return False


def _instance_radius(spec: BuildingSpec) -> float:
    return 0.6 * math.sqrt(spec.width ** 2 + spec.depth ** 2)


def _make_instance(
    terrain,
    spec: BuildingSpec,
    center: Tuple[float, float],
    rotation: float,
    padding: float,
    occupied: List[Tuple[float, float, float]],
) -> Optional[BuildingInstance]:
    radius = _instance_radius(spec)
    if not _bounds_ok(terrain, center, radius):
        return None
    base, heights = _footprint_heights(terrain, spec, center, rotation)
    if base is None or not heights:
        return None
    if max(heights) - min(heights) > spec.max_slope:
        return None
    if _overlaps(occupied, center, radius, padding):
        return None
    occupied.append((center[0], center[1], radius))
    return BuildingInstance(spec=spec, center=center, rotation=rotation, base_height=base + spec.base_offset)


def _sample_along_path(
    path: Sequence[Tuple[float, float]],
    lengths: np.ndarray,
    cumulative: np.ndarray,
    total_length: float,
    spacing_range: Tuple[float, float],
    start: float,
    rng: np.random.Generator,
) -> Iterable[Tuple[np.ndarray, np.ndarray, float]]:
    if len(path) < 2 or total_length <= 0.1:
        return []
    pos = start
    idx = 0
    acc = 0.0
    while pos < total_length:
        while idx < len(lengths) and cumulative[idx + 1] < pos:
            idx += 1
        if idx >= len(lengths):
            break
        seg_len = float(lengths[idx])
        if seg_len <= 1e-6:
            pos += rng.uniform(*spacing_range)
            continue
        t = (pos - cumulative[idx]) / seg_len
        p0 = np.array(path[idx], dtype="f4")
        p1 = np.array(path[idx + 1], dtype="f4")
        point = p0 + (p1 - p0) * t
        tangent = (p1 - p0) / seg_len
        yield point, tangent, pos
        spacing = rng.uniform(*spacing_range)
        pos += spacing


def _place_residential(
    terrain,
    path: Sequence[Tuple[float, float]],
    plan: Dict,
    rng: np.random.Generator,
    occupied: List[Tuple[float, float, float]],
) -> List[BuildingInstance]:
    lengths, cumulative, total_length = _polyline_lengths(path)
    if total_length <= 0.0:
        return []
    lane_width = float(plan.get("lane_width", 3.2))
    lanes = int(plan.get("lanes", 2))
    shoulder = float(plan.get("shoulder", 0.6))
    half_road = 0.5 * lane_width * lanes + shoulder
    instances: List[BuildingInstance] = []
    for point, tangent, dist in _sample_along_path(path, lengths, cumulative, total_length, (18.0, 28.0), 35.0, rng):
        tangent2 = tangent / max(np.linalg.norm(tangent), 1e-6)
        normal = np.array([-tangent2[1], tangent2[0]], dtype="f4")
        for side in (-1, 1):
            if rng.random() < 0.35:
                continue
            spec = _spec_house(rng)
            side_normal = normal * side
            setback = half_road + rng.uniform(5.0, 14.0) + spec.depth * 0.5
            offset = side_normal * setback
            jitter = tangent2 * rng.uniform(-4.0, 4.0)
            center = (float(point[0] + offset[0] + jitter[0]), float(point[1] + offset[1] + jitter[1]))
            rotation = math.atan2(-side_normal[0], -side_normal[1])
            inst = _make_instance(terrain, spec, center, rotation, 1.8, occupied)
            if inst:
                instances.append(inst)
    return instances


def _place_commercial(
    terrain,
    path: Sequence[Tuple[float, float]],
    plan: Dict,
    rng: np.random.Generator,
    occupied: List[Tuple[float, float, float]],
) -> List[BuildingInstance]:
    lengths, cumulative, total_length = _polyline_lengths(path)
    if total_length <= 0.0:
        return []
    lane_width = float(plan.get("lane_width", 3.2))
    lanes = int(plan.get("lanes", 2))
    shoulder = float(plan.get("shoulder", 0.6))
    half_road = 0.5 * lane_width * lanes + shoulder
    instances: List[BuildingInstance] = []
    for point, tangent, dist in _sample_along_path(path, lengths, cumulative, total_length, (28.0, 42.0), 45.0, rng):
        tangent2 = tangent / max(np.linalg.norm(tangent), 1e-6)
        normal = np.array([-tangent2[1], tangent2[0]], dtype="f4")
        side = -1 if rng.random() < 0.5 else 1
        spec = _choose_spec("roadside", rng)
        side_normal = normal * side
        setback = half_road + rng.uniform(8.0, 20.0) + spec.depth * 0.5
        if spec.type in {"strip_mall", "industrial"}:
            setback += 4.0
        center = (
            float(point[0] + side_normal[0] * setback + tangent2[0] * rng.uniform(-6.0, 6.0)),
            float(point[1] + side_normal[1] * setback + tangent2[1] * rng.uniform(-6.0, 6.0)),
        )
        rotation = math.atan2(-side_normal[0], -side_normal[1])
        inst = _make_instance(terrain, spec, center, rotation, 2.5, occupied)
        if inst:
            instances.append(inst)
    return instances


def _place_remote(
    terrain,
    path: Sequence[Tuple[float, float]],
    plan: Dict,
    rng: np.random.Generator,
    occupied: List[Tuple[float, float, float]],
    count: int,
) -> List[BuildingInstance]:
    instances: List[BuildingInstance] = []
    lane_width = float(plan.get("lane_width", 3.2))
    lanes = int(plan.get("lanes", 2))
    shoulder = float(plan.get("shoulder", 0.6))
    half_road = 0.5 * lane_width * lanes + shoulder
    for _ in range(count):
        attempts = 0
        while attempts < 20:
            attempts += 1
            choice = rng.random()
            if choice < 0.2:
                spec = _choose_spec("landmark", rng)
            elif choice < 0.5:
                spec = _choose_spec("tall", rng)
            elif choice < 0.75:
                spec = _choose_spec("industrial", rng)
            else:
                spec = _choose_spec("roadside", rng)
            radius = _instance_radius(spec)
            cx = float(rng.uniform(radius + 4.0, terrain.width - radius - 4.0))
            cz = float(rng.uniform(radius + 4.0, terrain.height - radius - 4.0))
            if path:
                dist = _distance_to_path((cx, cz), path)
                if dist < half_road + max(12.0, spec.depth):
                    continue
            rotation = float(rng.uniform(0.0, math.tau))
            inst = _make_instance(terrain, spec, (cx, cz), rotation, 3.5, occupied)
            if inst:
                instances.append(inst)
                break
    return instances


def _build_instance_vertices(
    terrain,
    instance: BuildingInstance,
    palette: Dict[str, List[np.ndarray]],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    spec = instance.spec
    local_rng = np.random.default_rng(rng.integers(0, 2**32))
    colors = _select_colors(palette, spec, local_rng)
    if spec.type == "pyramid":
        tris = _build_pyramid_geometry(spec, colors, local_rng)
    else:
        tris = _build_rectangular(spec, colors, local_rng)
    position = (instance.center[0], instance.base_height, instance.center[1])
    vertices = _assemble_vertices(tris, position, instance.rotation)
    return vertices, spec.noise_scale


def generate_buildings(
    terrain,
    road_points: Optional[Sequence[Tuple[float, float]]],
    plan: Optional[Dict],
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, object]:
    if rng is None:
        rng = np.random.default_rng()
    palette = _random_palette(rng)
    occupied: List[Tuple[float, float, float]] = []
    instances: List[BuildingInstance] = []
    if road_points and plan:
        instances.extend(_place_residential(terrain, road_points, plan, rng, occupied))
        instances.extend(_place_commercial(terrain, road_points, plan, rng, occupied))
    area = terrain.width * terrain.height
    remote_target = int(max(6, min(20, area / 25000)))
    instances.extend(
        _place_remote(
            terrain,
            road_points or [],
            plan or {},
            rng,
            occupied,
            remote_target,
        )
    )
    vertices_list: List[np.ndarray] = []
    noise_values: List[float] = []
    for inst in instances:
        verts, noise = _build_instance_vertices(terrain, inst, palette, rng)
        if verts.size:
            vertices_list.append(verts)
            noise_values.append(noise)
    if vertices_list:
        vertices = np.concatenate(vertices_list, axis=0)
        avg_noise = float(np.mean(noise_values)) if noise_values else 0.0
    else:
        vertices = np.zeros((0, 10), dtype="f4")
        avg_noise = 0.0
    return {
        "vertices": vertices.astype("f4"),
        "instances": instances,
        "palette": palette,
        "noise_scale": avg_noise,
    }
