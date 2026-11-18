from pathlib import Path
import cv2
import numpy as np


def intrinsics(width, height, hfov=75.0):
    if width < 2 or height < 2 or not 10 < hfov < 170:
        raise ValueError('Invalid image dimensions or horizontal field of view.')
    f = width / (2 * np.tan(np.deg2rad(hfov) / 2))
    return np.array([[f, 0, (width - 1) / 2], [0, f, (height - 1) / 2], [0, 0, 1]], dtype=float)


def backproject(rgb, depth, K, stride=3, valid=None):
    if depth.shape != rgb.shape[:2] or stride < 1:
        raise ValueError('Depth must match the image; stride must be positive.')
    v, u = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    z = depth[::stride, ::stride]
    rays = np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(K).T
    points = rays * z[..., None]
    ok = np.isfinite(points).all(-1) & (z > 0)
    if valid is not None:
        ok &= valid[::stride, ::stride]
    return points[ok].astype(np.float32), rgb[::stride, ::stride][ok]


def ply_bytes(points, colors, faces=None):
    points, colors = np.asarray(points), np.asarray(colors)
    if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape:
        raise ValueError('Expected matching Nx3 points and RGB colors.')
    if not np.isfinite(points).all():
        raise ValueError('Point coordinates must be finite.')
    header = f'ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\n'
    header += 'property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n'
    if faces is not None:
        faces = np.asarray(faces, dtype=np.int32)
        if faces.ndim != 2 or faces.shape[1] != 3 or (faces < 0).any() or (faces >= len(points)).any():
            raise ValueError('Invalid triangle indices.')
        header += f'element face {len(faces)}\nproperty list uchar int vertex_indices\n'
    verts = np.empty(len(points), dtype=[('p', '<f4', (3,)), ('c', 'u1', (3,))])
    verts['p'], verts['c'] = points, np.clip(colors, 0, 255).astype(np.uint8)
    result = (header + 'end_header\n').encode() + verts.tobytes()
    if faces is not None:
        triangles = np.empty(len(faces), dtype=[('n', 'u1'), ('i', '<i4', (3,))])
        triangles['n'], triangles['i'] = 3, faces
        result += triangles.tobytes()
    return result


def render_cloud(points, colors, yaw=0.0, pitch=-0.15, size=(800, 540)):
    """Orthographic orbit preview with a per-pixel nearest-point z buffer."""
    w, h = size
    canvas = np.full((h, w, 3), (11, 17, 26), dtype=np.uint8)
    if not len(points):
        return canvas
    p = np.asarray(points, dtype=float)
    p = p - np.median(p, axis=0)
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    R = np.array([[cy, 0, sy], [sp * sy, cp, -sp * cy], [-cp * sy, sp, cp * cy]])
    q = p @ R.T
    scale = min(w, h) * .8 / max(np.percentile(np.abs(q[:, :2]), 98) * 2, 1e-6)
    uv = np.round(q[:, :2] * scale + [w / 2, h / 2]).astype(int)
    ok = np.isfinite(q).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    uv, z, c = uv[ok], q[ok, 2], np.asarray(colors)[ok]
    flat = uv[:, 1] * w + uv[:, 0]
    order = np.argsort(z, kind='stable')
    _, first = np.unique(flat[order], return_index=True)
    keep = order[first]
    canvas[uv[keep, 1], uv[keep, 0]] = c[keep]
    return canvas


def save_cloud(path, points, colors, faces=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(ply_bytes(points, colors, faces))
