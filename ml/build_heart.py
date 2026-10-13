"""Build a schematic heart and snap coronary paths onto it.

The surface is the classic algebraic heart, rotated so the apex points down
and skewed slightly toward anatomical left. Vessel polylines are chosen from
surface vertices so LAD, LCX, and RCA sit on the epicardium.

Axes after export: +Y base, -Y apex, +Z anterior, +X anatomical left.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from skimage.measure import marching_cubes

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "web" / "public" / "anatomy"
PLOT_DIR = ROOT / "ml" / "plots"


def algebraic_heart(grid: int = 88) -> trimesh.Trimesh:
    lin = np.linspace(-1.55, 1.55, grid)
    step = float(lin[1] - lin[0])
    x, y, z = np.meshgrid(lin, lin, lin, indexing="ij")
    blob = x**2 + 2.25 * y**2 + z**2 - 1.0
    volume = blob**3 - (x**2) * (z**3) - (9.0 / 80.0) * (y**2) * (z**3)
    verts, faces, _, _ = marching_cubes(volume, level=0.0, spacing=(step, step, step))
    verts = verts + lin[0]
    # Formula axes: x left-right lobes, z from apex (negative) toward the cleft (positive).
    oriented = np.column_stack([verts[:, 0], verts[:, 2], verts[:, 1]])
    oriented[:, 0] *= 1.02
    oriented[:, 1] *= 1.08
    oriented[:, 2] *= 1.18
    apex_drop = np.clip(-oriented[:, 1], 0, None)
    apex_drop = apex_drop / (apex_drop.max() + 1e-8)
    oriented[:, 0] += 0.34 * apex_drop
    mesh = trimesh.Trimesh(vertices=oriented, faces=faces, process=True)
    parts = mesh.split(only_watertight=False)
    mesh = max(parts, key=lambda part: len(part.faces))
    trimesh.smoothing.filter_laplacian(mesh, lamb=0.35, iterations=6)
    mesh.fix_normals()
    return mesh


def _band(vertices: np.ndarray, y: float, width: float) -> np.ndarray:
    return np.where(np.abs(vertices[:, 1] - y) <= width)[0]


def _pick(vertices: np.ndarray, candidates: np.ndarray, score: np.ndarray) -> np.ndarray:
    if len(candidates) == 0:
        raise RuntimeError("Vessel path found an empty surface band.")
    return vertices[candidates[int(np.argmax(score))]]


def lad_points(vertices: np.ndarray) -> list[np.ndarray]:
    apex = vertices[int(np.argmin(vertices[:, 1]))]
    ys = np.linspace(0.72, apex[1] + 0.05, 8)
    points = []
    for y in ys:
        width = 0.09
        candidates = _band(vertices, y, width)
        while len(candidates) < 8 and width < 0.35:
            width += 0.04
            candidates = _band(vertices, y, width)
        x_target = np.interp(y, [apex[1], 0.72], [apex[0] * 0.85, 0.18])
        score = vertices[candidates, 2] - 0.45 * np.abs(vertices[candidates, 0] - x_target)
        points.append(_pick(vertices, candidates, score))
    points.append(apex)
    return points


def lcx_points(vertices: np.ndarray, origin: np.ndarray) -> list[np.ndarray]:
    points = [origin]
    samples = [
        (0.55, 0.45, 1.0),
        (0.28, 0.85, 0.2),
        (0.05, 0.95, -0.6),
        (-0.12, 0.7, -1.2),
    ]
    for y, x_target, z_weight in samples:
        width = 0.1
        candidates = _band(vertices, y, width)
        candidates = candidates[vertices[candidates, 0] > 0.05]
        while len(candidates) < 5 and width < 0.4:
            width += 0.05
            candidates = _band(vertices, y, width)
            candidates = candidates[vertices[candidates, 0] > 0.0]
        score = z_weight * vertices[candidates, 2] + 0.8 * vertices[candidates, 0]
        score -= 0.25 * np.abs(vertices[candidates, 0] - x_target)
        points.append(_pick(vertices, candidates, score))
    return points


def rca_points(vertices: np.ndarray) -> list[np.ndarray]:
    samples = [
        (0.62, -0.2, 1.0),
        (0.38, -0.7, 0.4),
        (0.08, -0.9, -0.3),
        (-0.12, -0.55, -1.1),
        (-0.28, -0.15, -1.3),
    ]
    points = []
    for y, x_target, z_weight in samples:
        width = 0.1
        candidates = _band(vertices, y, width)
        candidates = candidates[vertices[candidates, 0] < 0.05]
        while len(candidates) < 5 and width < 0.4:
            width += 0.05
            candidates = _band(vertices, y, width)
            candidates = candidates[vertices[candidates, 0] < 0.15]
        score = z_weight * vertices[candidates, 2] - 0.15 * np.abs(vertices[candidates, 0] - x_target)
        points.append(_pick(vertices, candidates, score))
    return points


def smooth_and_lift(points: list[np.ndarray], vertices: np.ndarray, normals: np.ndarray, lift: float) -> list[list[float]]:
    raw = np.vstack(points)
    if len(raw) >= 3:
        kernel = np.array([0.2, 0.6, 0.2])
        padded = np.vstack([raw[0], raw, raw[-1]])
        smoothed = np.vstack(
            [np.convolve(padded[:, axis], kernel, mode="valid") for axis in range(3)]
        ).T
        smoothed[0] = raw[0]
        smoothed[-1] = raw[-1]
    else:
        smoothed = raw
    from scipy.spatial import cKDTree

    _, index = cKDTree(vertices).query(smoothed)
    lifted = vertices[index] + normals[index] * lift
    return np.round(lifted, 5).tolist()


def branch_from(points: list[list[float]], vertices: np.ndarray, normals: np.ndarray, x_sign: float, lift: float) -> list[list[float]]:
    from scipy.spatial import cKDTree

    anchors = [1, len(points) // 2]
    tree = cKDTree(vertices)
    branches = []
    for anchor_index in anchors:
        start = np.asarray(points[min(anchor_index, len(points) - 1)], dtype=float)
        steps = []
        cursor = start.copy()
        for step in range(1, 4):
            target = cursor + np.array([0.16 * x_sign, -0.08, 0.05 if x_sign > 0 else 0.02])
            band = np.where(np.linalg.norm(vertices - target, axis=1) < 0.45)[0]
            if len(band) == 0:
                break
            score = x_sign * vertices[band, 0] + 0.3 * vertices[band, 2] - 0.4 * np.abs(vertices[band, 1] - target[1])
            chosen = vertices[band[int(np.argmax(score))]]
            steps.append(chosen)
            cursor = chosen
        if len(steps) >= 2:
            _, index = tree.query(np.vstack([start, steps]))
            lifted = vertices[index] + normals[index] * lift
            branches.append(np.round(lifted, 5).tolist())
    return branches


def build_vessels(mesh: trimesh.Trimesh) -> list[dict]:
    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    lad = lad_points(vertices)
    lcx = lcx_points(vertices, lad[0])
    rca = rca_points(vertices)
    specs = [
        ("lad", "LAD", "Left anterior descending", 0.046, lad, 1.0,
         "Runs down the front of the heart toward the apex. Supplies the anterior wall and much of the septum."),
        ("lcx", "LCX", "Left circumflex", 0.04, lcx, 1.0,
         "Runs in the left atrioventricular groove toward the side and back. Supplies the lateral wall."),
        ("rca", "RCA", "Right coronary artery", 0.042, rca, -1.0,
         "Runs in the right atrioventricular groove. Supplies the right side and, in a typical right-dominant heart, the underside."),
    ]
    vessels = []
    for vid, name, full, radius, points, x_sign, blurb in specs:
        lifted = smooth_and_lift(points, vertices, normals, lift=radius * 0.55)
        vessels.append(
            {
                "id": vid,
                "name": name,
                "full": full,
                "blurb": blurb,
                "radius": radius,
                "points": lifted,
                "branches": [
                    {"radius": radius * 0.46, "points": branch}
                    for branch in branch_from(lifted, vertices, normals, x_sign, lift=radius * 0.3)
                ],
            }
        )
    return vessels


def preview(mesh: trimesh.Trimesh, vessels: list[dict]) -> None:
    fig = plt.figure(figsize=(10, 10))
    views = [("Anterior", 8, -90), ("Left", 8, 0), ("Posterior", 8, 90), ("Apex", -80, -90)]
    verts = mesh.vertices
    step = max(1, len(mesh.faces) // 3500)
    faces = mesh.faces[::step]
    colors = {"lad": "#c43c3c", "lcx": "#d89a22", "rca": "#1f8f96"}
    for index, (title, elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 2, index, projection="3d")
        ax.plot_trisurf(
            verts[:, 0], verts[:, 1], verts[:, 2],
            triangles=faces, color="#9a3d4a", alpha=0.38, linewidth=0, antialiased=False,
        )
        for vessel in vessels:
            pts = np.asarray(vessel["points"])
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=colors[vessel["id"]], lw=2.6)
            ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2], color=colors[vessel["id"]], s=12)
        ax.set_title(title)
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((1, 1.15, 0.8))
        ax.set_axis_off()
    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_DIR / "heart_preview.png", dpi=140)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mesh = algebraic_heart()
    print(f"faces={len(mesh.faces)} bounds={np.round(mesh.bounds, 3).tolist()}", flush=True)
    color = np.tile(np.array([154, 62, 76, 255], dtype=np.uint8), (len(mesh.vertices), 1))
    mesh.visual.vertex_colors = color
    mesh.export(OUT_DIR / "heart.glb")
    vessels = build_vessels(mesh)
    payload = {
        "axes": {
            "y": "base is positive, apex is negative",
            "z": "anterior is positive",
            "x": "anatomical left is positive",
        },
        "note": "Procedural schematic from the algebraic heart surface in ml/build_heart.py. Not a patient imaging mesh.",
        "vessels": vessels,
    }
    (OUT_DIR / "vessels.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    preview(mesh, vessels)
    for vessel in vessels:
        pts = np.asarray(vessel["points"])
        print(vessel["id"], "n", len(pts), "start", pts[0].round(2), "end", pts[-1].round(2), "branches", len(vessel["branches"]))
    print("wrote anatomy", flush=True)


if __name__ == "__main__":
    main()
