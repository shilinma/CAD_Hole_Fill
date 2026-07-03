#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAD 孔洞补片三角化。

该模块不再沿用旧的前沿推进状态机，而是围绕一个新的核心函数
`generate_hole_patch_mesh()` 组织补洞流程。新流程会尽量利用上游分析出的
参数域、闭合方式与边界来源信息：

1. 选择工作域：优先使用参数域边界，否则退回局部平面。
2. 自适应细分边界：对特征闭合段、曲面参数域和尖角区域更密集采样。
3. 生成内部采样点：在工作域中布置受边界和特征约束影响的内部点。
4. Delaunay 候选三角化：过滤越界/穿边三角形。
5. 局部质量优化：边翻转与内部点平滑。
6. 将新点抬回 3D 曲面，压缩未使用点并输出补片。

公开接口：
- `triangulate_hole_patch`
- `triangulate_ordered_hole_boundary`

旧名 `advancing_front_triangulate_*` 仍保留为兼容别名。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AbstractSet, Callable, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np

try:
    from scipy.spatial import Delaunay as _Delaunay

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _Delaunay = None  # type: ignore[assignment]
    _HAS_SCIPY = False

SMALL_ANGLE_DEG = 75.0
EQUILATERAL_HEIGHT_SCALE = 0.8660254037844386  # sqrt(3) / 2


@dataclass
class HolePatchContext:
    boundary_xyz: np.ndarray
    boundary_edges: np.ndarray
    boundary_uv: Optional[np.ndarray]
    lift_fn: Optional[Callable[[np.ndarray], np.ndarray]]
    boundary_sources: Optional[Sequence[int]]
    closure_kind: str
    parameterization_kind: str
    open_boundary_count: Optional[int]
    feature_point_vertex_indices: Optional[Tuple[int, int]]
    reference_normal: Optional[np.ndarray]
    small_angle_deg: float
    density_scale: float
    seam_constrained_edges: Optional[AbstractSet[Tuple[int, int]]] = None


def order_boundary_loop_from_edges(n_vertices: int, edges: np.ndarray) -> np.ndarray:
    """将无向边表恢复为单一简单闭合环。"""
    edges = np.asarray(edges, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("boundary_edges 需要是 (M, 2) 的整数数组")
    if n_vertices < 3:
        raise ValueError("边界顶点数不足 3")
    if edges.size == 0:
        raise ValueError("boundary_edges 为空")

    adj: List[List[int]] = [[] for _ in range(n_vertices)]
    for a, b in edges:
        ia, ib = int(a), int(b)
        if ia < 0 or ia >= n_vertices or ib < 0 or ib >= n_vertices:
            raise ValueError("boundary_edges 含越界顶点索引")
        if ia == ib:
            raise ValueError("boundary_edges 含退化自环")
        adj[ia].append(ib)
        adj[ib].append(ia)

    for i, nb in enumerate(adj):
        if len(nb) != 2:
            raise ValueError(
                f"顶点 {i} 的度为 {len(nb)}，期望简单闭合环上每点度数为 2"
            )

    start = 0
    loop: List[int] = [start]
    prev = -1
    cur = adj[start][0]
    while cur != start:
        loop.append(cur)
        nxt = adj[cur][0] if adj[cur][1] == prev else adj[cur][1]
        prev, cur = cur, nxt
        if len(loop) > n_vertices + 1:
            raise ValueError("无法闭合边界环")

    if len(loop) != n_vertices:
        raise ValueError("边表与顶点数不一致或存在多个环")
    return np.array(loop, dtype=np.int64)


def _newell_normal(points_xyz: np.ndarray) -> np.ndarray:
    acc = np.zeros(3, dtype=np.float64)
    n = int(points_xyz.shape[0])
    for i in range(n):
        p0 = points_xyz[i]
        p1 = points_xyz[(i + 1) % n]
        acc[0] += (p0[1] - p1[1]) * (p0[2] + p1[2])
        acc[1] += (p0[2] - p1[2]) * (p0[0] + p1[0])
        acc[2] += (p0[0] - p1[0]) * (p0[1] + p1[1])
    ln = float(np.linalg.norm(acc))
    if ln < 1e-15:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return acc / ln


def build_plane_frame(
    boundary_xyz: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回 `(centroid, normal, u_axis, v_axis)`。"""
    centroid = np.mean(boundary_xyz, axis=0)
    normal = _newell_normal(boundary_xyz)
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(normal, ref))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u_axis = np.cross(normal, ref)
    u_axis = u_axis / (float(np.linalg.norm(u_axis)) + 1e-15)
    v_axis = np.cross(normal, u_axis)
    v_axis = v_axis / (float(np.linalg.norm(v_axis)) + 1e-15)
    return centroid, normal, u_axis, v_axis


def project_to_plane_2d(
    points_xyz: np.ndarray,
    centroid: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
) -> np.ndarray:
    rel = points_xyz - centroid.reshape(1, 3)
    return np.column_stack([rel @ u_axis, rel @ v_axis])


def _find_simple_planar_projection(
    ordered_xyz: np.ndarray,
    reference_normal: Optional[np.ndarray],
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    多候选投影法向，寻找使闭合折线在 2D 上无自交且面积非退化的平面展开。
    用于圆柱/圆锥解析 UV 或默认 Newell 展开失败时的兜底。
    """
    ordered_xyz = np.asarray(ordered_xyz, dtype=np.float64)
    n = int(ordered_xyz.shape[0])
    if n < 3:
        return None
    centroid = np.mean(ordered_xyz, axis=0)
    rel = ordered_xyz - centroid.reshape(1, 3)
    candidates: List[np.ndarray] = []

    def add_candidate(vec: np.ndarray) -> None:
        ln = float(np.linalg.norm(vec))
        if ln < 1e-12:
            return
        v = vec / ln
        for existing in candidates:
            if abs(float(np.dot(existing, v))) > 0.999:
                return
        candidates.append(v)

    try:
        _, _, u0, v0 = build_plane_frame(ordered_xyz)
        add_candidate(np.cross(u0, v0))
    except Exception:
        pass
    if reference_normal is not None:
        add_candidate(np.asarray(reference_normal, dtype=np.float64))
    if n >= 3:
        _, _, vh = np.linalg.svd(rel, full_matrices=False)
        add_candidate(np.asarray(vh[-1], dtype=np.float64))
        add_candidate(np.asarray(vh[0], dtype=np.float64))
    for i in range(min(n, 12)):
        e0 = rel[(i + 1) % n] - rel[i]
        e1 = rel[(i + 2) % n] - rel[(i + 1) % n]
        add_candidate(np.cross(e0, e1))
    for ax in (
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
    ):
        add_candidate(ax)

    id_loop = list(range(n))
    diag_base = max(float(np.linalg.norm(np.ptp(ordered_xyz, axis=0))), 1.0)

    for normal in candidates:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(normal, ref))) > 0.9:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        u_axis = np.cross(normal, ref)
        u_ln = float(np.linalg.norm(u_axis))
        if u_ln < 1e-12:
            continue
        u_axis = u_axis / u_ln
        v_axis = np.cross(normal, u_axis)
        v_axis = v_axis / (float(np.linalg.norm(v_axis)) + 1e-15)
        uv = np.column_stack([rel @ u_axis, rel @ v_axis])
        du = float(np.linalg.norm(np.ptp(uv, axis=0)))
        diag = max(du, diag_base)
        eps = max(1e-12, 1e-10 * diag)
        if abs(signed_polygon_area_2d(uv)) <= eps:
            continue
        if _polygon_is_simple(id_loop, uv, eps):
            return centroid, u_axis, v_axis, uv
    return None


def signed_polygon_area_2d(poly: np.ndarray) -> float:
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def orient_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _point_on_segment_2d(a: np.ndarray, b: np.ndarray, p: np.ndarray, eps: float) -> bool:
    if abs(orient_2d(a, b, p)) > eps:
        return False
    return (
        min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
    )


def _segments_intersect_2d(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    eps: float,
) -> bool:
    o1 = orient_2d(a, b, c)
    o2 = orient_2d(a, b, d)
    o3 = orient_2d(c, d, a)
    o4 = orient_2d(c, d, b)

    def sgn(x: float) -> int:
        if x > eps:
            return 1
        if x < -eps:
            return -1
        return 0

    s1, s2, s3, s4 = sgn(o1), sgn(o2), sgn(o3), sgn(o4)
    if s1 != s2 and s3 != s4:
        return True
    if s1 == 0 and _point_on_segment_2d(a, b, c, eps):
        return True
    if s2 == 0 and _point_on_segment_2d(a, b, d, eps):
        return True
    if s3 == 0 and _point_on_segment_2d(c, d, a, eps):
        return True
    if s4 == 0 and _point_on_segment_2d(c, d, b, eps):
        return True
    return False


def _segments_intersect_proper_2d(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    eps: float,
) -> bool:
    o1 = orient_2d(a, b, c)
    o2 = orient_2d(a, b, d)
    o3 = orient_2d(c, d, a)
    o4 = orient_2d(c, d, b)
    return (
        ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps))
        and ((o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps))
    )


def _point_in_triangle_2d(
    p: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    eps: float,
) -> bool:
    o1 = orient_2d(a, b, p)
    o2 = orient_2d(b, c, p)
    o3 = orient_2d(c, a, p)
    return o1 >= -eps and o2 >= -eps and o3 >= -eps


def _point_in_polygon_2d(
    p: np.ndarray,
    polygon: Sequence[int],
    points_2d: np.ndarray,
    eps: float,
) -> bool:
    inside = False
    px, py = float(p[0]), float(p[1])
    n = len(polygon)
    for i in range(n):
        a = points_2d[int(polygon[i])]
        b = points_2d[int(polygon[(i + 1) % n])]
        if _point_on_segment_2d(a, b, p, eps):
            return True
        yi, yj = float(a[1]), float(b[1])
        intersects = ((yi > py) != (yj > py)) and (
            px
            < (float(b[0]) - float(a[0])) * (py - yi) / (yj - yi + 1e-30)
            + float(a[0])
        )
        if intersects:
            inside = not inside
    return inside


def _triangle_min_angle(pa: np.ndarray, pb: np.ndarray, pc: np.ndarray) -> float:
    def _angle(u: np.ndarray, v: np.ndarray) -> float:
        nu = float(np.linalg.norm(u))
        nv = float(np.linalg.norm(v))
        if nu < 1e-15 or nv < 1e-15:
            return 0.0
        cos_uv = np.clip(float(np.dot(u, v)) / (nu * nv), -1.0, 1.0)
        return float(np.arccos(cos_uv))

    return min(
        _angle(pb - pa, pc - pa),
        _angle(pa - pb, pc - pb),
        _angle(pa - pc, pb - pc),
    )


def _edge_length_quality_score(
    pa: np.ndarray, pb: np.ndarray, pc: np.ndarray, target_len: Optional[float]
) -> float:
    if target_len is None or target_len <= 1e-15:
        return 0.0
    lengths = np.array(
        [
            float(np.linalg.norm(pb - pa)),
            float(np.linalg.norm(pc - pb)),
            float(np.linalg.norm(pa - pc)),
        ],
        dtype=np.float64,
    )
    rel_dev = np.mean(np.abs(lengths - target_len) / max(target_len, 1e-15))
    spread = float(np.std(lengths) / max(target_len, 1e-15))
    return 1.0 / (1.0 + 1.4 * rel_dev + 0.8 * spread)


def _triangle_score(
    pa: np.ndarray,
    pb: np.ndarray,
    pc: np.ndarray,
    target_len: Optional[float] = None,
) -> float:
    area2 = abs(orient_2d(pa, pb, pc))
    perimeter = (
        float(np.linalg.norm(pb - pa))
        + float(np.linalg.norm(pc - pb))
        + float(np.linalg.norm(pa - pc))
    )
    return (
        _triangle_min_angle(pa, pb, pc)
        + area2 / max(perimeter * perimeter, 1e-15)
        + 0.9 * _edge_length_quality_score(pa, pb, pc, target_len)
    )


def _ccw_triangle(a: int, b: int, c: int, points_2d: np.ndarray) -> Tuple[int, int, int]:
    if orient_2d(points_2d[a], points_2d[b], points_2d[c]) >= 0.0:
        return (a, b, c)
    return (a, c, b)


def _boundary_edge_set(loop: Sequence[int]) -> set[Tuple[int, int]]:
    return {
        (int(loop[i]), int(loop[(i + 1) % len(loop)]))
        if int(loop[i]) < int(loop[(i + 1) % len(loop)])
        else (int(loop[(i + 1) % len(loop)]), int(loop[i]))
        for i in range(len(loop))
    }


def _edges_in_triangulation(faces: np.ndarray) -> set[Tuple[int, int]]:
    """三角化中出现的无向边集合。"""
    out: set[Tuple[int, int]] = set()
    for tri in np.asarray(faces, dtype=np.int64):
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, v in ((a, b), (b, c), (c, a)):
            out.add((u, v) if u < v else (v, u))
    return out


def _missing_protected_edges(
    faces: np.ndarray,
    protected_edges: AbstractSet[Tuple[int, int]],
) -> set[Tuple[int, int]]:
    present = _edges_in_triangulation(faces)
    return {e for e in protected_edges if e not in present}


def _find_interior_steiner_uv(
    loop: Sequence[int],
    points_2d: np.ndarray,
    eps: float,
) -> np.ndarray:
    """在简单多边形内部找 Steiner 点（扇形三角化用）。"""
    loop_list = [int(v) for v in loop]
    poly_pts = points_2d[np.array(loop_list, dtype=np.int64)]
    centroid = np.mean(poly_pts, axis=0)
    if _point_in_polygon_2d(centroid, loop_list, points_2d, eps):
        return np.asarray(centroid, dtype=np.float64).reshape(2)
    mn = np.min(poly_pts, axis=0)
    mx = np.max(poly_pts, axis=0)
    for u in np.linspace(0.15, 0.85, 8):
        for v in np.linspace(0.15, 0.85, 8):
            cand = mn + u * (mx - mn)
            cand[1] = mn[1] + v * (mx[1] - mn[1])
            if _point_in_polygon_2d(cand, loop_list, points_2d, eps):
                return np.asarray(cand, dtype=np.float64).reshape(2)
    raise RuntimeError("triangulation: 无法找到多边形内部 Steiner 点")


def _steiner_fan_triangulation(
    loop: Sequence[int],
    points_2d: np.ndarray,
    steiner_idx: int,
    eps: float,
) -> np.ndarray:
    """扇形三角化：每条边界边 (a,b) 与内部 Steiner 点组成三角形。"""
    loop_list = [int(v) for v in loop]
    faces: List[Tuple[int, int, int]] = []
    for i in range(len(loop_list)):
        a = int(loop_list[i])
        b = int(loop_list[(i + 1) % len(loop_list)])
        faces.append(_ccw_triangle(a, b, int(steiner_idx), points_2d))
    return np.array(faces, dtype=np.int64)


def _constrained_polygon_triangulation(
    loop: Sequence[int],
    points_2d: np.ndarray,
    protected_edges: AbstractSet[Tuple[int, int]],
    eps: float,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    简单多边形约束三角化：输出必须覆盖全部 ``protected_edges``。

    优先耳切；失败或缺边时退回内部 Steiner 扇形（保证锐棱角 seam 完整）。
    返回 ``(faces, steiner_uv)``；无新增 Steiner 时 ``steiner_uv`` 为 ``None``。
    """
    protected = set(protected_edges)
    loop_list = [int(v) for v in loop]
    if len(loop_list) == 3:
        a, b, c = loop_list
        faces = np.array(
            [_ccw_triangle(a, b, c, points_2d)],
            dtype=np.int64,
        )
        if not _missing_protected_edges(faces, protected):
            return faces, None
    try:
        faces = _ear_clip_triangulation(loop, points_2d, eps)
        if not _missing_protected_edges(faces, protected):
            return faces, None
    except RuntimeError:
        pass
    steiner_uv = _find_interior_steiner_uv(loop, points_2d, eps)
    steiner_idx = int(points_2d.shape[0])
    pts_ext = np.vstack([points_2d, steiner_uv.reshape(1, 2)])
    faces = _steiner_fan_triangulation(loop, pts_ext, steiner_idx, eps)
    missing = _missing_protected_edges(faces, protected)
    if missing:
        raise RuntimeError(
            "triangulation: 约束三角化仍缺 protected 边: "
            f"{sorted(missing)}"
        )
    return faces, steiner_uv


def _protected_edge_set(
    loop: Sequence[int],
    constrained_edges: Optional[AbstractSet[Tuple[int, int]]],
) -> set[Tuple[int, int]]:
    protected = _boundary_edge_set(loop)
    if constrained_edges:
        protected = protected | {tuple(sorted((int(a), int(b)))) for a, b in constrained_edges}
    return protected


def _accept_faces_if_covers(
    candidate: np.ndarray,
    protected_edges: AbstractSet[Tuple[int, int]],
    fallback: np.ndarray,
) -> np.ndarray:
    """仅当 candidate 覆盖全部 protected 边时才采纳，否则保留 fallback。"""
    if _missing_protected_edges(candidate, protected_edges):
        return fallback
    return candidate


def _polygon_is_simple(loop: Sequence[int], points_2d: np.ndarray, eps: float) -> bool:
    n = len(loop)
    for i in range(n):
        a0 = int(loop[i])
        a1 = int(loop[(i + 1) % n])
        pa0 = points_2d[a0]
        pa1 = points_2d[a1]
        for j in range(i + 1, n):
            if (j + 1) % n == i or (i + 1) % n == j:
                continue
            b0 = int(loop[j])
            b1 = int(loop[(j + 1) % n])
            if len({a0, a1, b0, b1}) < 4:
                continue
            if _segments_intersect_2d(pa0, pa1, points_2d[b0], points_2d[b1], eps):
                return False
    return True


def _point_to_segment_distance_2d(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    lab2 = float(np.dot(ab, ab))
    if lab2 < 1e-30:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab)) / lab2
    t = max(0.0, min(1.0, t))
    q = a + t * ab
    return float(np.linalg.norm(p - q))


def _safe_normalize_3d(vec: np.ndarray) -> np.ndarray:
    nrm = float(np.linalg.norm(vec))
    if nrm < 1e-15:
        return np.zeros(3, dtype=np.float64)
    return vec / nrm


def _loop_edge_lengths(loop: Sequence[int], points_2d: np.ndarray) -> np.ndarray:
    n = len(loop)
    lengths = np.zeros(n, dtype=np.float64)
    for i in range(n):
        a = points_2d[int(loop[i])]
        b = points_2d[int(loop[(i + 1) % n])]
        lengths[i] = float(np.linalg.norm(b - a))
    return lengths


def _interior_refinement_budget(
    polygon_area: float,
    mean_h: float,
    n_boundary: int,
    parameterization_kind: str,
    closure_kind: str,
) -> int:
    """
    按子孔面积与目标边长估算内部 Steiner 上限；小楔形允许 0。

    平面小 patch 优先边界三角化不加内部点；圆柱/曲面域按面积比例适度细化。
    """
    if int(n_boundary) <= 3:
        return 0
    cell = max(float(mean_h) * float(mean_h), 1e-24)
    area = max(float(polygon_area), 0.0)
    kind = str(parameterization_kind)
    zero_area_factor = {
        "plane": 1.15,
        "local_plane": 1.15,
        "cylinder": 1.35,
        "cylinder_tangent": 1.35,
        "cylinder_local_plane": 1.35,
        "sphere_tangent": 1.25,
        "cone": 1.25,
        "developable_strip": 1.2,
    }.get(kind, 1.2)
    if area <= zero_area_factor * cell:
        return 0
    density = {
        "plane": 0.72,
        "local_plane": 0.72,
        "cylinder": 0.95,
        "cylinder_tangent": 0.95,
        "cylinder_local_plane": 0.9,
        "sphere_tangent": 0.88,
        "cone": 0.9,
        "developable_strip": 0.8,
    }.get(kind, 0.82)
    if str(closure_kind) == "curve_arc_partition":
        est = int(np.ceil(density * area / cell))
        return max(0, min(80, est))
    if str(closure_kind) == "multi_patch_cell":
        est = int(np.ceil(max(1.0, density) * 2.5 * area / cell))
        return max(0, min(180, est))
    est = int(np.ceil(density * area / cell))
    return max(0, min(160, est))


def _vertex_spacing_field(
    loop: Sequence[int],
    points_2d: np.ndarray,
    density_scale: float,
    closure_kind: str,
    parameterization_kind: str,
    eps: float,
) -> np.ndarray:
    edge_lengths = _loop_edge_lengths(loop, points_2d)
    base_mean = max(float(np.mean(edge_lengths)), eps * 100.0)
    spacing = np.zeros(len(loop), dtype=np.float64)
    for i in range(len(loop)):
        local_len = 0.5 * (edge_lengths[i - 1] + edge_lengths[i])
        local_len = max(local_len, 0.55 * base_mean)
        mult = 1.0
        if closure_kind == "multi_patch_cell":
            mult *= 1.05
        if parameterization_kind in {
            "cylinder",
            "cylinder_tangent",
            "cylinder_local_plane",
            "sphere_tangent",
        }:
            mult *= 0.95
        spacing[i] = local_len * mult / max(density_scale, 1e-12)
    if closure_kind == "multi_patch_cell":
        return np.clip(spacing, 0.65 * base_mean, 1.45 * base_mean)
    return np.clip(spacing, 0.5 * base_mean, 1.6 * base_mean)


def _compact_vertices(
    n_boundary: int,
    vertices_xyz: np.ndarray,
    faces: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    used = {int(v) for tri in faces for v in tri}
    keep_order = list(range(n_boundary)) + sorted(
        idx for idx in used if idx >= n_boundary
    )
    remap = {old: new for new, old in enumerate(keep_order)}
    vertices_out = vertices_xyz[np.array(keep_order, dtype=np.int64)]
    faces_out = np.array(
        [[remap[int(v)] for v in tri] for tri in faces.tolist()],
        dtype=np.int64,
    )
    return vertices_out, faces_out


def _choose_working_domain(
    context: HolePatchContext,
    loop: np.ndarray,
) -> Tuple[np.ndarray, Callable[[np.ndarray], np.ndarray], np.ndarray]:
    ordered_xyz = context.boundary_xyz[loop]
    centroid, _normal, u_axis, v_axis = build_plane_frame(ordered_xyz)
    planar_uv = project_to_plane_2d(ordered_xyz, centroid, u_axis, v_axis)

    n = int(loop.shape[0])
    id_loop = list(range(n))
    diag_base = max(float(np.linalg.norm(np.ptp(ordered_xyz, axis=0))), 1.0)

    def _uv_usable(uv: np.ndarray, loop_indices: Sequence[int]) -> bool:
        du = float(np.linalg.norm(np.ptp(uv, axis=0)))
        diag = max(du, diag_base)
        eps = max(1e-12, 1e-10 * diag)
        if abs(signed_polygon_area_2d(uv)) <= eps:
            return False
        return _polygon_is_simple([int(x) for x in loop_indices], uv, eps)

    def _planar_lift_fn() -> Callable[[np.ndarray], np.ndarray]:
        def planar_lift(uv: np.ndarray) -> np.ndarray:
            p_uv = np.asarray(uv, dtype=np.float64).reshape(2)
            return centroid + float(p_uv[0]) * u_axis + float(p_uv[1]) * v_axis

        return planar_lift

    if context.boundary_uv is not None and context.lift_fn is not None:
        uv_loop = np.asarray(context.boundary_uv[loop], dtype=np.float64)
        if _uv_usable(uv_loop, id_loop):
            global_uv = np.asarray(context.boundary_uv, dtype=np.float64)
            return uv_loop, cast(Callable[[np.ndarray], np.ndarray], context.lift_fn), global_uv

    if not _uv_usable(planar_uv, id_loop):
        alt = _find_simple_planar_projection(ordered_xyz, context.reference_normal)
        if alt is not None:
            centroid, u_axis, v_axis, planar_uv = alt

    if _uv_usable(planar_uv, id_loop):
        return planar_uv, _planar_lift_fn(), planar_uv

    raise RuntimeError(
        "triangulation: 参数域与平面投影均非简单多边形，拒绝圆盘 IDW 近似三角化"
    )



def _smooth_interior_points(
    polygon_loop: Sequence[int],
    points_2d: np.ndarray,
    first_interior_idx: int,
    spacing: float,
    eps: float,
) -> np.ndarray:
    if first_interior_idx >= points_2d.shape[0]:
        return points_2d

    out = points_2d.copy()
    boundary_poly = [int(v) for v in polygon_loop]
    margin = 0.25 * spacing
    for _ in range(2):
        prev = out.copy()
        for idx in range(first_interior_idx, prev.shape[0]):
            p = prev[idx]
            d = np.linalg.norm(prev - p.reshape(1, 2), axis=1)
            neighbors = prev[(d > 1e-12) & (d < 1.9 * spacing)]
            if neighbors.shape[0] < 3:
                continue
            candidate = 0.55 * p + 0.45 * np.mean(neighbors, axis=0)
            if not _point_in_polygon_2d(candidate, boundary_poly, prev, eps):
                continue
            keep = True
            for i in range(len(boundary_poly)):
                a = prev[boundary_poly[i]]
                b = prev[boundary_poly[(i + 1) % len(boundary_poly)]]
                if _point_to_segment_distance_2d(candidate, a, b) < margin:
                    keep = False
                    break
            if keep:
                out[idx] = candidate
    return out


def _edge_is_polygon_legal(
    u: int,
    v: int,
    polygon_loop: Sequence[int],
    points_2d: np.ndarray,
    boundary_edges: set[Tuple[int, int]],
    eps: float,
) -> bool:
    key = (u, v) if u < v else (v, u)
    if key in boundary_edges:
        return True
    pu = points_2d[u]
    pv = points_2d[v]
    mid = 0.5 * (pu + pv)
    if not _point_in_polygon_2d(mid, polygon_loop, points_2d, eps):
        return False
    for i in range(len(polygon_loop)):
        a = int(polygon_loop[i])
        b = int(polygon_loop[(i + 1) % len(polygon_loop)])
        if len({u, v, a, b}) < 4:
            continue
        if _segments_intersect_proper_2d(pu, pv, points_2d[a], points_2d[b], eps):
            return False
    return True


def _triangle_is_polygon_legal(
    tri: Tuple[int, int, int],
    polygon_loop: Sequence[int],
    points_2d: np.ndarray,
    boundary_edges: set[Tuple[int, int]],
    constrained_edges: set[Tuple[int, int]],
    eps: float,
) -> bool:
    a, b, c = tri
    pa, pb, pc = points_2d[a], points_2d[b], points_2d[c]
    if abs(orient_2d(pa, pb, pc)) <= eps:
        return False
    centroid = (pa + pb + pc) / 3.0
    if not _point_in_polygon_2d(centroid, polygon_loop, points_2d, eps):
        return False
    return (
        _edge_is_polygon_legal(a, b, polygon_loop, points_2d, boundary_edges | constrained_edges, eps)
        and _edge_is_polygon_legal(b, c, polygon_loop, points_2d, boundary_edges | constrained_edges, eps)
        and _edge_is_polygon_legal(c, a, polygon_loop, points_2d, boundary_edges | constrained_edges, eps)
    )


def _build_delaunay_faces(
    polygon_loop: Sequence[int],
    points_2d: np.ndarray,
    constrained_edges: Optional[set[Tuple[int, int]]],
    eps: float,
) -> np.ndarray:
    if not _HAS_SCIPY:
        raise RuntimeError("triangulation: 需要 scipy.spatial.Delaunay")
    delaunay = _Delaunay(points_2d)
    boundary_edges = _boundary_edge_set(polygon_loop)
    constraints = constrained_edges if constrained_edges is not None else set()
    faces: List[Tuple[int, int, int]] = []
    seen: set[Tuple[int, int, int]] = set()
    for tri_raw in delaunay.simplices:
        tri = _ccw_triangle(
            int(tri_raw[0]),
            int(tri_raw[1]),
            int(tri_raw[2]),
            points_2d,
        )
        if not _triangle_is_polygon_legal(
            tri,
            polygon_loop,
            points_2d,
            boundary_edges,
            constraints,
            eps,
        ):
            continue
        key = tuple(sorted(tri))
        if key in seen:
            continue
        seen.add(key)
        faces.append(tri)
    if not faces:
        raise RuntimeError("triangulation: Delaunay 未生成有效三角形")
    return np.array(faces, dtype=np.int64)


def _is_valid_ear(
    polygon: Sequence[int],
    corner_idx: int,
    points_2d: np.ndarray,
    eps: float,
) -> bool:
    m = len(polygon)
    a = int(polygon[(corner_idx - 1) % m])
    b = int(polygon[corner_idx])
    c = int(polygon[(corner_idx + 1) % m])
    pa = points_2d[a]
    pb = points_2d[b]
    pc = points_2d[c]
    if orient_2d(pa, pb, pc) <= eps:
        return False
    for vid in polygon:
        if int(vid) in {a, b, c}:
            continue
        if _point_in_triangle_2d(points_2d[int(vid)], pa, pb, pc, eps):
            return False
    for j in range(m):
        u = int(polygon[j])
        v = int(polygon[(j + 1) % m])
        if len({u, v, a, c}) < 4:
            continue
        if _segments_intersect_2d(pa, pc, points_2d[u], points_2d[v], eps):
            return False
    return _point_in_polygon_2d(0.5 * (pa + pc), polygon, points_2d, eps)


def _ear_clip_triangulation(
    loop: Sequence[int],
    points_2d: np.ndarray,
    eps: float,
) -> np.ndarray:
    polygon = [int(v) for v in loop]
    faces: List[Tuple[int, int, int]] = []
    max_iters = max(8 * len(polygon), 32)
    iters = 0
    while len(polygon) > 3:
        best_idx = -1
        best_score = -float("inf")
        best_face: Optional[Tuple[int, int, int]] = None
        for idx in range(len(polygon)):
            if not _is_valid_ear(polygon, idx, points_2d, eps):
                continue
            a = polygon[(idx - 1) % len(polygon)]
            b = polygon[idx]
            c = polygon[(idx + 1) % len(polygon)]
            score = _triangle_score(points_2d[a], points_2d[b], points_2d[c])
            if score > best_score:
                best_idx = idx
                best_score = score
                best_face = _ccw_triangle(a, b, c, points_2d)
        if best_face is None:
            raise RuntimeError("triangulation: 耳切失败，边界可能退化或参数域存在自交")
        faces.append(best_face)
        polygon.pop(best_idx)
        iters += 1
        if iters > max_iters:
            raise RuntimeError("triangulation: 耳切超过最大迭代次数")
    if len(polygon) == 3:
        faces.append(_ccw_triangle(polygon[0], polygon[1], polygon[2], points_2d))
    return np.array(faces, dtype=np.int64)


def _build_edge_to_faces(
    faces: Sequence[Tuple[int, int, int]]
) -> dict[Tuple[int, int], List[int]]:
    edge_to_faces: dict[Tuple[int, int], List[int]] = {}
    for fi, tri in enumerate(faces):
        for u, v in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = (u, v) if u < v else (v, u)
            edge_to_faces.setdefault(key, []).append(fi)
    return edge_to_faces


def _other_vertex_of_triangle(tri: Tuple[int, int, int], edge: Tuple[int, int]) -> int:
    for v in tri:
        if v not in edge:
            return int(v)
    raise RuntimeError("triangulation: 无法在三角形中找到对边顶点")


def _optimize_internal_edges(
    loop: Sequence[int],
    points_2d: np.ndarray,
    faces_in: np.ndarray,
    constrained_edges: Optional[set[Tuple[int, int]]],
    eps: float,
) -> np.ndarray:
    boundary_edges = _boundary_edge_set(loop)
    protected_edges = boundary_edges | (constrained_edges if constrained_edges is not None else set())
    polygon = [int(v) for v in loop]
    faces = [tuple(int(x) for x in tri) for tri in faces_in.tolist()]
    max_rounds = max(2 * len(faces), 8)
    for _ in range(max_rounds):
        improved = False
        edge_to_faces = _build_edge_to_faces(faces)
        for edge, owners in edge_to_faces.items():
            if len(owners) != 2 or edge in protected_edges:
                continue
            f0, f1 = owners
            tri0 = faces[f0]
            tri1 = faces[f1]
            a, b = edge
            c = _other_vertex_of_triangle(tri0, edge)
            d = _other_vertex_of_triangle(tri1, edge)
            if len({a, b, c, d}) != 4:
                continue
            pa, pb = points_2d[a], points_2d[b]
            pc, pd = points_2d[c], points_2d[d]
            if not _segments_intersect_proper_2d(pa, pb, pc, pd, eps):
                continue

            cand0 = _ccw_triangle(c, d, a, points_2d)
            cand1 = _ccw_triangle(d, c, b, points_2d)
            cent0 = np.mean(points_2d[np.array(cand0, dtype=np.int64)], axis=0)
            cent1 = np.mean(points_2d[np.array(cand1, dtype=np.int64)], axis=0)
            if not _point_in_polygon_2d(cent0, polygon, points_2d, eps):
                continue
            if not _point_in_polygon_2d(cent1, polygon, points_2d, eps):
                continue
            if not _triangle_is_polygon_legal(
                cand0,
                polygon,
                points_2d,
                boundary_edges,
                constrained_edges if constrained_edges is not None else set(),
                eps,
            ):
                continue
            if not _triangle_is_polygon_legal(
                cand1,
                polygon,
                points_2d,
                boundary_edges,
                constrained_edges if constrained_edges is not None else set(),
                eps,
            ):
                continue

            old_score = _triangle_score(
                points_2d[tri0[0]], points_2d[tri0[1]], points_2d[tri0[2]]
            ) + _triangle_score(
                points_2d[tri1[0]], points_2d[tri1[1]], points_2d[tri1[2]]
            )
            new_score = _triangle_score(
                points_2d[cand0[0]], points_2d[cand0[1]], points_2d[cand0[2]]
            ) + _triangle_score(
                points_2d[cand1[0]], points_2d[cand1[1]], points_2d[cand1[2]]
            )
            if new_score <= old_score + 1e-8:
                continue

            faces[f0] = cand0
            faces[f1] = cand1
            improved = True
            break
        if not improved:
            break
    return np.array(faces, dtype=np.int64)


def _prepare_direct_patch_data(
    context: HolePatchContext,
) -> Tuple[np.ndarray, np.ndarray, Callable[[np.ndarray], np.ndarray], float]:
    boundary_xyz = np.asarray(context.boundary_xyz, dtype=np.float64)
    n0 = int(boundary_xyz.shape[0])
    if context.boundary_uv is not None and context.lift_fn is not None:
        boundary_uv = np.asarray(context.boundary_uv, dtype=np.float64)
        if boundary_uv.shape == (n0, 2):
            diag = max(
                float(np.linalg.norm(np.ptp(boundary_uv, axis=0))),
                float(np.linalg.norm(np.ptp(boundary_xyz, axis=0))),
                1.0,
            )
            return boundary_xyz.copy(), boundary_uv.copy(), context.lift_fn, max(
                1e-12, 1e-10 * diag
            )

    centroid, _normal, u_axis, v_axis = build_plane_frame(boundary_xyz)
    uv = project_to_plane_2d(boundary_xyz, centroid, u_axis, v_axis)

    def planar_lift(p_uv: np.ndarray) -> np.ndarray:
        p_uv = np.asarray(p_uv, dtype=np.float64)
        return centroid + p_uv[0] * u_axis + p_uv[1] * v_axis

    diag = max(
        float(np.linalg.norm(np.ptp(uv, axis=0))),
        float(np.linalg.norm(np.ptp(boundary_xyz, axis=0))),
        1.0,
    )
    return boundary_xyz.copy(), uv, planar_lift, max(1e-12, 1e-10 * diag)


def _derive_constrained_edges(
    loop: Sequence[int],
    boundary_sources: Sequence[int],
) -> set[Tuple[int, int]]:
    def _flush_chain(indices: Sequence[int]) -> set[Tuple[int, int]]:
        edges: set[Tuple[int, int]] = set()
        for i in range(len(indices) - 1):
            u = int(indices[i])
            v = int(indices[i + 1])
            if u == v:
                continue
            edges.add((u, v) if u < v else (v, u))
        return edges

    loop_list = [int(v) for v in loop]
    if len(loop_list) < 3 or len(boundary_sources) != len(loop_list):
        return set()
    constraints: set[Tuple[int, int]] = set()
    current_chain: List[int] = [int(loop_list[0])]
    for i in range(1, len(loop_list)):
        if int(boundary_sources[i]) >= 0:
            current_chain.append(int(loop_list[i]))
            continue
        if len(current_chain) >= 2:
            constraints.update(_flush_chain(current_chain))
        current_chain = [int(loop_list[i])]
    if len(current_chain) >= 2:
        constraints.update(_flush_chain(current_chain))
    return constraints


def _laplacian_smooth_patch_vertices(
    vertices_xyz: np.ndarray,
    faces: np.ndarray,
    n_boundary: int,
    lift_fn: Callable[[np.ndarray], np.ndarray],
    points_uv: np.ndarray,
    iterations: int = 2,
) -> np.ndarray:
    out = np.asarray(vertices_xyz, dtype=np.float64, copy=True)
    if out.shape[0] <= n_boundary:
        return out
    adjacency: List[set[int]] = [set() for _ in range(out.shape[0])]
    for tri in faces:
        a, b, c = (int(tri[0]), int(tri[1]), int(tri[2]))
        adjacency[a].update([b, c])
        adjacency[b].update([a, c])
        adjacency[c].update([a, b])
    for _ in range(max(0, int(iterations))):
        prev = out.copy()
        for idx in range(n_boundary, out.shape[0]):
            if not adjacency[idx]:
                continue
            target = np.mean(prev[np.array(sorted(adjacency[idx]), dtype=np.int64)], axis=0)
            blended = 0.45 * prev[idx] + 0.55 * target
            lifted = np.asarray(lift_fn(points_uv[idx]), dtype=np.float64).reshape(3)
            out[idx] = 0.4 * blended + 0.6 * lifted
    for idx in range(n_boundary, out.shape[0]):
        out[idx] = np.asarray(lift_fn(points_uv[idx]), dtype=np.float64).reshape(3)
    return out



def _triangle_centroid_2d(pa: np.ndarray, pb: np.ndarray, pc: np.ndarray) -> np.ndarray:
    return (pa + pb + pc) / 3.0


def _triangle_incenter_2d(pa: np.ndarray, pb: np.ndarray, pc: np.ndarray) -> np.ndarray:
    a = float(np.linalg.norm(pc - pb))
    b = float(np.linalg.norm(pc - pa))
    c = float(np.linalg.norm(pb - pa))
    s = a + b + c
    if s < 1e-15:
        return _triangle_centroid_2d(pa, pb, pc)
    return (a * pa + b * pb + c * pc) / s


def _triangle_circumcenter_2d(pa: np.ndarray, pb: np.ndarray, pc: np.ndarray) -> np.ndarray:
    ax, ay = float(pa[0]), float(pa[1])
    bx, by = float(pb[0]), float(pb[1])
    cx, cy = float(pc[0]), float(pc[1])
    d = 2.0 * (
        ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)
    )
    if abs(d) < 1e-15:
        return _triangle_centroid_2d(pa, pb, pc)
    a2 = ax * ax + ay * ay
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    return np.array([ux, uy], dtype=np.float64)


def _local_surface_normal(
    lift_fn: Callable[[np.ndarray], np.ndarray],
    uv: np.ndarray,
    eps: float,
) -> np.ndarray:
    h = max(1e-5, 8.0 * eps)
    u = np.array([h, 0.0], dtype=np.float64)
    v = np.array([0.0, h], dtype=np.float64)
    p = np.asarray(lift_fn(uv), dtype=np.float64)
    pu = np.asarray(lift_fn(uv + u), dtype=np.float64)
    pv = np.asarray(lift_fn(uv + v), dtype=np.float64)
    n = np.cross(pu - p, pv - p)
    return _safe_normalize_3d(n)


def _orient_faces_by_local_surface(
    vertices_xyz: np.ndarray,
    faces: np.ndarray,
    points_uv: np.ndarray,
    lift_fn: Callable[[np.ndarray], np.ndarray],
    fallback_normal: np.ndarray,
    eps: float,
) -> np.ndarray:
    out = np.array(faces, dtype=np.int64, copy=True)
    fallback = _safe_normalize_3d(fallback_normal)
    for i, tri in enumerate(out):
        tri_idx = np.array(tri, dtype=np.int64)
        uv_centroid = np.mean(points_uv[tri_idx], axis=0)
        local_normal = _local_surface_normal(lift_fn, uv_centroid, eps)
        if float(np.linalg.norm(local_normal)) < 1e-12:
            local_normal = fallback
        pa = vertices_xyz[int(tri[0])]
        pb = vertices_xyz[int(tri[1])]
        pc = vertices_xyz[int(tri[2])]
        tri_normal = np.cross(pb - pa, pc - pa)
        if float(np.dot(tri_normal, local_normal)) < 0.0:
            out[i] = np.array([tri[0], tri[2], tri[1]], dtype=np.int64)
    return out


def _boundary_size_at_point(
    p: np.ndarray,
    loop: Sequence[int],
    points_2d: np.ndarray,
    boundary_spacing: np.ndarray,
) -> float:
    loop_arr = np.array(loop, dtype=np.int64)
    boundary_pts = points_2d[loop_arr]
    d = np.linalg.norm(boundary_pts - p.reshape(1, 2), axis=1)
    if d.shape[0] == 0:
        return 1.0
    k = min(3, d.shape[0])
    idx = np.argpartition(d, k - 1)[:k]
    w = 1.0 / np.maximum(d[idx], 1e-12)
    return float(np.sum(w * boundary_spacing[idx]) / np.sum(w))


def _candidate_point_is_admissible(
    p: np.ndarray,
    loop: Sequence[int],
    points_2d: np.ndarray,
    target_h: float,
    closure_kind: str,
    constrained_edges: Optional[set[Tuple[int, int]]],
    eps: float,
) -> bool:
    if not _point_in_polygon_2d(p, loop, points_2d, eps):
        return False
    dist_vertices = np.linalg.norm(points_2d - p.reshape(1, 2), axis=1)
    vertex_margin = 0.45 * target_h
    edge_margin = 0.28 * target_h
    if closure_kind == "multi_patch_cell":
        # 允许内部点更接近已有约束，使密边界尺度能向子孔内部传播。
        vertex_margin = 0.48 * target_h
        edge_margin = 0.34 * target_h
    if np.any(dist_vertices < vertex_margin):
        return False
    for i in range(len(loop)):
        a = points_2d[int(loop[i])]
        b = points_2d[int(loop[(i + 1) % len(loop)])]
        if _point_to_segment_distance_2d(p, a, b) < edge_margin:
            return False
    if constrained_edges:
        for u, v in constrained_edges:
            a = points_2d[int(u)]
            b = points_2d[int(v)]
            if _point_to_segment_distance_2d(p, a, b) < edge_margin:
                return False
    return True



def _select_refinement_candidate(
    faces: np.ndarray,
    loop: Sequence[int],
    points_2d: np.ndarray,
    boundary_spacing: np.ndarray,
    closure_kind: str,
    constrained_edges: Optional[set[Tuple[int, int]]],
    eps: float,
) -> Optional[np.ndarray]:
    best_badness = 0.0
    best_point: Optional[np.ndarray] = None
    min_angle_target = np.deg2rad(32.0)
    if closure_kind == "multi_patch_cell":
        min_angle_target = np.deg2rad(24.0)
    for tri in faces:
        pa = points_2d[int(tri[0])]
        pb = points_2d[int(tri[1])]
        pc = points_2d[int(tri[2])]
        centroid = _triangle_centroid_2d(pa, pb, pc)
        target_h = _boundary_size_at_point(centroid, loop, points_2d, boundary_spacing)
        lengths = np.array(
            [
                float(np.linalg.norm(pb - pa)),
                float(np.linalg.norm(pc - pb)),
                float(np.linalg.norm(pa - pc)),
            ],
            dtype=np.float64,
        )
        max_len = float(np.max(lengths))
        min_angle = _triangle_min_angle(pa, pb, pc)
        area = 0.5 * abs(orient_2d(pa, pb, pc))
        length_badness = max(0.0, max_len / max(target_h, 1e-12) - 1.55)
        angle_badness = max(0.0, (min_angle_target - min_angle) / min_angle_target)
        area_badness = max(
            0.0,
            area / max(0.4330127018922193 * target_h * target_h, 1e-12) - 1.35,
        )
        badness = max(length_badness, area_badness) + 0.7 * angle_badness
        if closure_kind == "multi_patch_cell":
            longest_edge = int(np.argmax(lengths))
            edge_pairs = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
            long_u, long_v = edge_pairs[longest_edge]
            edge = (int(long_u), int(long_v))
            edge = edge if edge[0] < edge[1] else (edge[1], edge[0])
            if constrained_edges and edge in constrained_edges:
                continue
            # 多 patch 子孔内部不能比边界粗太多，较早拆分长边/大面积三角形。
            if max_len < 1.05 * target_h or area_badness < 0.16:
                continue
            badness = max(length_badness, area_badness) + 0.35 * angle_badness
            if badness < 0.22:
                continue
        if badness <= best_badness + 1e-12:
            continue
        candidate_list = [
            _triangle_circumcenter_2d(pa, pb, pc),
            _triangle_centroid_2d(pa, pb, pc),
            _triangle_incenter_2d(pa, pb, pc),
        ]
        accepted = False
        for candidate in candidate_list:
            if not _candidate_point_is_admissible(
                candidate,
                loop,
                points_2d,
                target_h,
                closure_kind,
                constrained_edges,
                eps,
            ):
                continue
            best_badness = badness
            best_point = candidate
            accepted = True
            break
        if not accepted:
            continue
    return best_point


def _refine_curve_arc_partition_triangulation(
    loop: Sequence[int],
    points_uv_list: List[np.ndarray],
    points_xyz_list: List[np.ndarray],
    lift_fn: Callable[[np.ndarray], np.ndarray],
    density_scale: float,
    parameterization_kind: str,
    constrained_edges: Optional[set[Tuple[int, int]]],
    eps: float,
) -> np.ndarray:
    """
    curve_arc 楔形子孔：UV 约束三角化 + 可选尺寸场细化。

    契约：``protected``（闭合环全部边 + seam）必须全程保留；Delaunay 重建
    若丢边则拒绝采纳。平面假 8 字自交不在此处理（上游已跳过平面拆片）。
    """
    protected = _protected_edge_set(loop, constrained_edges)
    points_uv = np.vstack(points_uv_list)
    faces, steiner_uv = _constrained_polygon_triangulation(
        loop, points_uv, protected, eps
    )
    if steiner_uv is not None:
        points_uv_list.append(np.asarray(steiner_uv, dtype=np.float64).reshape(2))
        points_xyz_list.append(
            np.asarray(lift_fn(steiner_uv), dtype=np.float64).reshape(3)
        )

    boundary_spacing = _vertex_spacing_field(
        loop,
        np.vstack(points_uv_list),
        density_scale,
        "curve_arc_partition",
        parameterization_kind,
        eps,
    )
    polygon_area = 0.5 * abs(
        signed_polygon_area_2d(
            np.vstack(points_uv_list)[np.array(loop, dtype=np.int64)]
        )
    )
    mean_h = max(float(np.mean(boundary_spacing)), eps * 100.0)
    max_new_points = _interior_refinement_budget(
        polygon_area,
        mean_h,
        len(loop),
        parameterization_kind,
        "curve_arc_partition",
    )
    if max_new_points <= 0:
        missing = _missing_protected_edges(faces, protected)
        if missing:
            raise RuntimeError(
                "triangulation: curve_arc 约束三角化缺 protected 边: "
                f"{sorted(missing)}"
            )
        return faces

    first_interior_idx = len(points_uv_list)
    stalled = 0
    for _ in range(max_new_points):
        points_uv = np.vstack(points_uv_list)
        selected = _select_refinement_candidate(
            faces,
            loop,
            points_uv,
            boundary_spacing,
            "curve_arc_partition",
            constrained_edges,
            eps,
        )
        if selected is None:
            break
        candidate_uv = selected
        candidate_xyz = np.asarray(lift_fn(candidate_uv), dtype=np.float64).reshape(3)
        points_uv_list.append(np.asarray(candidate_uv, dtype=np.float64).reshape(2))
        points_xyz_list.append(candidate_xyz)
        points_uv_candidate = np.vstack(points_uv_list)
        try:
            new_faces = _build_delaunay_faces(
                loop, points_uv_candidate, constrained_edges, eps
            )
        except RuntimeError:
            points_uv_list.pop()
            points_xyz_list.pop()
            stalled += 1
            if stalled >= 6:
                break
            continue
        faces = _accept_faces_if_covers(new_faces, protected, faces)
        stalled = 0

    points_uv = np.vstack(points_uv_list)
    points_uv = _smooth_interior_points(
        loop, points_uv, first_interior_idx, mean_h, eps
    )
    for idx in range(first_interior_idx, points_uv.shape[0]):
        points_xyz_list[idx] = np.asarray(
            lift_fn(points_uv[idx]), dtype=np.float64
        ).reshape(3)
    try:
        candidate = _build_delaunay_faces(loop, points_uv, constrained_edges, eps)
    except RuntimeError:
        try:
            candidate = _optimize_internal_edges(
                loop, points_uv, faces, constrained_edges, eps
            )
        except RuntimeError:
            candidate = faces
    faces = _accept_faces_if_covers(candidate, protected, faces)
    missing = _missing_protected_edges(faces, protected)
    if missing:
        raise RuntimeError(
            "triangulation: curve_arc 约束三角化缺 protected 边: "
            f"{sorted(missing)}"
        )
    return faces


def _refine_triangulation_with_size_field(
    loop: Sequence[int],
    points_uv_list: List[np.ndarray],
    points_xyz_list: List[np.ndarray],
    lift_fn: Callable[[np.ndarray], np.ndarray],
    density_scale: float,
    closure_kind: str,
    parameterization_kind: str,
    constrained_edges: Optional[set[Tuple[int, int]]],
    eps: float,
) -> np.ndarray:
    if closure_kind == "curve_arc_partition":
        return _refine_curve_arc_partition_triangulation(
            loop,
            points_uv_list,
            points_xyz_list,
            lift_fn,
            density_scale,
            parameterization_kind,
            constrained_edges,
            eps,
        )
    points_uv = np.vstack(points_uv_list)
    points_xyz = np.vstack(points_xyz_list)
    boundary_spacing = _vertex_spacing_field(
        loop,
        points_uv,
        density_scale,
        closure_kind,
        parameterization_kind,
        eps,
    )
    try:
        faces = _build_delaunay_faces(loop, points_uv, constrained_edges, eps)
    except RuntimeError:
        faces = _ear_clip_triangulation(loop, points_uv, eps)
    if closure_kind == "multi_patch_cell":
        try:
            faces = _optimize_internal_edges(loop, points_uv, faces, constrained_edges, eps)
        except RuntimeError:
            pass
    polygon_area = 0.5 * abs(
        signed_polygon_area_2d(points_uv[np.array(loop, dtype=np.int64)])
    )
    mean_h = max(float(np.mean(boundary_spacing)), eps * 100.0)
    max_new_points = _interior_refinement_budget(
        polygon_area,
        mean_h,
        len(loop),
        parameterization_kind,
        closure_kind,
    )
    if max_new_points <= 0:
        protected = _boundary_edge_set(loop)
        if constrained_edges:
            protected = protected | set(constrained_edges)
        return _accept_faces_if_covers(faces, protected, faces)

    first_interior_idx = len(points_uv_list)
    stalled = 0
    for _ in range(max_new_points):
        points_uv = np.vstack(points_uv_list)
        selected = _select_refinement_candidate(
            faces,
            loop,
            points_uv,
            boundary_spacing,
            closure_kind,
            constrained_edges,
            eps,
        )
        if selected is None:
            break
        candidate_uv = selected
        candidate_xyz = np.asarray(lift_fn(candidate_uv), dtype=np.float64).reshape(3)
        points_uv_list.append(np.asarray(candidate_uv, dtype=np.float64).reshape(2))
        points_xyz_list.append(candidate_xyz)
        points_uv_candidate = np.vstack(points_uv_list)
        try:
            new_faces = _build_delaunay_faces(loop, points_uv_candidate, constrained_edges, eps)
        except RuntimeError:
            points_uv_list.pop()
            points_xyz_list.pop()
            stalled += 1
            if stalled >= 6:
                break
            continue
        faces = new_faces
        stalled = 0

    points_uv = np.vstack(points_uv_list)
    points_uv = _smooth_interior_points(loop, points_uv, first_interior_idx, mean_h, eps)
    for idx in range(first_interior_idx, points_uv.shape[0]):
        points_xyz_list[idx] = np.asarray(lift_fn(points_uv[idx]), dtype=np.float64).reshape(3)
    try:
        faces = _build_delaunay_faces(loop, points_uv, constrained_edges, eps)
    except RuntimeError:
        faces = _optimize_internal_edges(loop, points_uv, faces, constrained_edges, eps)
    return faces


def _triangulate_strip_patch(context: HolePatchContext) -> Tuple[np.ndarray, np.ndarray]:
    boundary_xyz, boundary_uv, lift_fn, eps = _prepare_direct_patch_data(context)
    reference_normal = (
        np.asarray(context.reference_normal, dtype=np.float64)
        if context.reference_normal is not None
        else _newell_normal(boundary_xyz)
    )
    n0 = int(boundary_xyz.shape[0])
    open_count = context.open_boundary_count
    if open_count is None:
        raise RuntimeError("triangulation: 缺少 strip 子孔开链长度")
    open_count = int(open_count)
    if open_count < 2 or open_count >= n0:
        raise RuntimeError("triangulation: strip 子孔开链长度非法")
    points_uv_list = [np.asarray(boundary_uv[i], dtype=np.float64).reshape(2) for i in range(n0)]
    points_xyz_list = [np.asarray(boundary_xyz[i], dtype=np.float64).reshape(3) for i in range(n0)]
    loop = list(range(n0))
    boundary_sources = (
        [int(x) for x in context.boundary_sources]
        if context.boundary_sources is not None and len(context.boundary_sources) == n0
        else list(range(n0))
    )
    constrained_edges = _derive_constrained_edges(loop, boundary_sources)
    if context.seam_constrained_edges:
        constrained_edges = set(constrained_edges) | {
            tuple(sorted((int(a), int(b))))
            for a, b in context.seam_constrained_edges
        }
    faces = _refine_triangulation_with_size_field(
        loop,
        points_uv_list,
        points_xyz_list,
        lift_fn,
        max(0.6, min(1.8, float(context.density_scale))),
        context.closure_kind,
        context.parameterization_kind,
        constrained_edges,
        eps,
    )
    if faces.size == 0:
        raise RuntimeError("triangulation: strip 细化三角化失败")
    points_uv = np.vstack(points_uv_list)
    vertices_xyz = np.vstack(points_xyz_list).astype(np.float64, copy=False)
    vertices_xyz = _laplacian_smooth_patch_vertices(
        vertices_xyz,
        faces,
        n0,
        lift_fn,
        points_uv,
    )
    faces_arr = _orient_faces_by_local_surface(
        vertices_xyz,
        faces,
        points_uv,
        lift_fn,
        reference_normal,
        eps,
    )
    return _compact_vertices(n0, vertices_xyz, faces_arr)



def _loop_plane_projection_is_simple(
    boundary_xyz: np.ndarray,
    loop: Sequence[int],
    eps: float,
) -> bool:
    """闭合环在 Newell 平面上的投影是否为简单多边形（无自交）。"""
    loop_list = [int(v) for v in loop]
    if len(loop_list) < 3:
        return True
    ordered_xyz = np.asarray(boundary_xyz, dtype=np.float64)[np.array(loop_list, dtype=np.int64)]
    centroid, _, u_axis, v_axis = build_plane_frame(ordered_xyz)
    uv = project_to_plane_2d(ordered_xyz, centroid, u_axis, v_axis)
    local_loop = list(range(len(loop_list)))
    return _polygon_is_simple(local_loop, uv, eps)


def _chain_along_loop(i0: int, i1: int, n: int) -> List[int]:
    if i0 < i1:
        return list(range(i0, i1 + 1))
    return list(range(i0, n)) + list(range(0, i1 + 1))


def _feature_positions_on_loop(
    loop: Sequence[int],
    boundary_sources: Sequence[int],
    feature_point_vertex_indices: Tuple[int, int],
) -> Optional[Tuple[int, int]]:
    fp0, fp1 = int(feature_point_vertex_indices[0]), int(feature_point_vertex_indices[1])
    if fp0 == fp1:
        return None
    loop_list = [int(v) for v in loop]
    loop_sources = [int(boundary_sources[int(v)]) for v in loop_list]
    try:
        i0 = loop_sources.index(fp0)
        i1 = loop_sources.index(fp1)
    except ValueError:
        return None
    return i0, i1


def _line_intersection_params_2d(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> Optional[Tuple[float, float, np.ndarray]]:
    r = b - a
    s = d - c
    den = float(r[0] * s[1] - r[1] * s[0])
    if abs(den) < 1e-15:
        return None
    q = c - a
    t = float((q[0] * s[1] - q[1] * s[0]) / den)
    u = float((q[0] * r[1] - q[1] * r[0]) / den)
    return t, u, a + t * r


def _find_loop_self_intersection(
    loop_uv: np.ndarray,
    eps: float,
) -> Optional[Tuple[int, int, float, float, np.ndarray]]:
    """仅接受边内部的真交点；汇交处近似重合端点不算可拆分自交。"""
    n = int(loop_uv.shape[0])
    best: Optional[Tuple[int, int, float, float, np.ndarray, float]] = None
    for i in range(n):
        i_next = (i + 1) % n
        a = loop_uv[i]
        b = loop_uv[i_next]
        for j in range(i + 1, n):
            j_next = (j + 1) % n
            if i_next == j or j_next == i:
                continue
            if i == 0 and j_next == 0:
                continue
            c = loop_uv[j]
            d = loop_uv[j_next]
            if not _segments_intersect_proper_2d(a, b, c, d, eps):
                continue
            hit = _line_intersection_params_2d(a, b, c, d)
            if hit is None:
                continue
            t, u, p = hit
            param_tol = max(1e-8, 10.0 * eps)
            if not (param_tol < t < 1.0 - param_tol and param_tol < u < 1.0 - param_tol):
                continue
            interior_margin = min(t, 1.0 - t, u, 1.0 - u)
            if best is None or interior_margin > best[5]:
                best = (i, j, t, u, p, interior_margin)
    if best is None:
        return None
    return best[0], best[1], best[2], best[3], best[4]


def boundary_coincidence_tolerance(boundary_xyz: np.ndarray) -> float:
    pts = np.asarray(boundary_xyz, dtype=np.float64)
    if pts.size == 0:
        return 1e-9
    return max(1e-9, 1e-5 * float(np.linalg.norm(np.ptp(pts, axis=0))))


def sanitize_closed_ring(
    closed_points: np.ndarray,
    sources: Sequence[int],
    *,
    tol: Optional[float] = None,
) -> Tuple[np.ndarray, List[int], int]:
    """按环序合并连续近似重合点（子孔构造阶段，与三角化阶段契约一致）。"""
    closed_points = np.asarray(closed_points, dtype=np.float64)
    n0 = int(closed_points.shape[0])
    if n0 < 3:
        return closed_points, [int(x) for x in sources], 0
    if tol is None:
        tol = boundary_coincidence_tolerance(closed_points)
    sources_list = [int(x) for x in sources]
    keep = [0]
    for i in range(1, n0):
        if float(np.linalg.norm(closed_points[i] - closed_points[keep[-1]])) > float(tol):
            keep.append(i)
    if len(keep) < 3 or len(keep) == n0:
        return closed_points, sources_list, 0
    return (
        closed_points[np.asarray(keep, dtype=np.int64)],
        [sources_list[i] for i in keep],
        int(n0 - len(keep)),
    )


def assess_patch_boundary_readiness(
    boundary_xyz: np.ndarray,
    boundary_uv: Optional[np.ndarray] = None,
    *,
    tol: Optional[float] = None,
) -> Dict[str, object]:
    """补洞前分层验收：返回是否可三角化及工作域选择依据（供批处理/调试）。"""
    pts = np.asarray(boundary_xyz, dtype=np.float64)
    n0 = int(pts.shape[0])
    edges = np.array([[i, (i + 1) % n0] for i in range(n0)], dtype=np.int64)
    if tol is None:
        tol = boundary_coincidence_tolerance(pts)
    s_xyz, s_uv, s_edges, _s_src, _ = _collapse_coincident_boundary_vertices(
        pts,
        boundary_uv,
        edges,
        list(range(n0)),
        None,
        tol=tol,
    )
    n1 = int(s_xyz.shape[0])
    loop = order_boundary_loop_from_edges(n1, s_edges)
    loop_list = loop.tolist()
    diag = max(float(np.linalg.norm(np.ptp(s_xyz, axis=0))), 1.0)
    eps = max(1e-12, 1e-10 * diag)
    plane_simple = _loop_plane_projection_is_simple(s_xyz, loop_list, eps)
    param_simple = False
    if s_uv is not None:
        param_simple = _polygon_is_simple(list(range(n1)), np.asarray(s_uv, dtype=np.float64), eps)
    ready = bool(plane_simple or param_simple)
    working_domain = (
        "parameterization"
        if param_simple
        else "plane_projection"
        if plane_simple
        else "none"
    )
    return {
        "ready": ready,
        "working_domain": working_domain,
        "plane_projection_simple": plane_simple,
        "parameterization_simple": param_simple,
        "n_boundary_in": n0,
        "n_boundary_sanitized": n1,
        "collapsed_vertices": int(n0 - n1),
    }


def _collapse_coincident_boundary_vertices(
    boundary_xyz: np.ndarray,
    boundary_uv: Optional[np.ndarray],
    boundary_edges: np.ndarray,
    boundary_sources: Sequence[int],
    seam_constrained_edges: Optional[AbstractSet[Tuple[int, int]]],
    *,
    tol: float,
) -> Tuple[
    np.ndarray,
    Optional[np.ndarray],
    np.ndarray,
    List[int],
    Optional[AbstractSet[Tuple[int, int]]],
]:
    """
    按环序合并连续近似重合的边界点（交线汇交虚拟端重复），避免平面投影假自交。
    """
    boundary_xyz = np.asarray(boundary_xyz, dtype=np.float64)
    n0 = int(boundary_xyz.shape[0])
    if n0 < 3:
        return boundary_xyz, boundary_uv, boundary_edges, [int(x) for x in boundary_sources], seam_constrained_edges

    loop = order_boundary_loop_from_edges(n0, boundary_edges)
    ordered_xyz = boundary_xyz[loop]
    keep_local = [0]
    for i in range(1, int(loop.shape[0])):
        if float(np.linalg.norm(ordered_xyz[i] - ordered_xyz[keep_local[-1]])) > float(tol):
            keep_local.append(i)
    if len(keep_local) < 3 or len(keep_local) == int(loop.shape[0]):
        return (
            boundary_xyz,
            boundary_uv,
            boundary_edges,
            [int(x) for x in boundary_sources],
            seam_constrained_edges,
        )

    kept_gids = [int(loop[i]) for i in keep_local]
    old_to_new = {int(gid): ni for ni, gid in enumerate(kept_gids)}
    new_xyz = boundary_xyz[np.asarray(kept_gids, dtype=np.int64)]
    new_uv = None
    if boundary_uv is not None:
        new_uv = np.asarray(boundary_uv, dtype=np.float64)[np.asarray(kept_gids, dtype=np.int64)]
    new_sources = [int(boundary_sources[gid]) for gid in kept_gids]
    new_n = int(new_xyz.shape[0])
    new_edges = np.array(
        [[i, (i + 1) % new_n] for i in range(new_n)],
        dtype=np.int64,
    )
    new_seam = seam_constrained_edges
    if seam_constrained_edges:
        remapped: set[Tuple[int, int]] = set()
        for a, b in seam_constrained_edges:
            ia, ib = int(a), int(b)
            if ia not in old_to_new or ib not in old_to_new:
                continue
            na, nb = int(old_to_new[ia]), int(old_to_new[ib])
            if na == nb:
                continue
            remapped.add((na, nb) if na < nb else (nb, na))
        new_seam = frozenset(remapped)
    return new_xyz, new_uv, new_edges, new_sources, new_seam


def _project_chain_to_plane_uv(
    vertices_xyz: np.ndarray,
    chain: Sequence[int],
) -> np.ndarray:
    chain_xyz = vertices_xyz[np.array([int(v) for v in chain], dtype=np.int64)]
    centroid, _, u_axis, v_axis = build_plane_frame(chain_xyz)
    return project_to_plane_2d(chain_xyz, centroid, u_axis, v_axis)


def _make_planar_lift_fn(
    centroid: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    origin = np.asarray(centroid, dtype=np.float64).reshape(3)
    u = np.asarray(u_axis, dtype=np.float64).reshape(3)
    v = np.asarray(v_axis, dtype=np.float64).reshape(3)

    def planar_lift(uv: np.ndarray) -> np.ndarray:
        p_uv = np.asarray(uv, dtype=np.float64).reshape(2)
        return origin + float(p_uv[0]) * u + float(p_uv[1]) * v

    return planar_lift


def _triangulate_chain_in_own_plane(
    vertices_xyz: np.ndarray,
    chain: Sequence[int],
    eps: float,
) -> Optional[np.ndarray]:
    chain_list = [int(v) for v in chain]
    if len(chain_list) < 3:
        return None
    chain_uv = _project_chain_to_plane_uv(vertices_xyz, chain_list)
    local_loop = list(range(len(chain_list)))
    if not _polygon_is_simple(local_loop, chain_uv, eps):
        return None
    try:
        local_faces = _ear_clip_triangulation(local_loop, chain_uv, eps)
    except RuntimeError:
        return None
    return np.array(
        [[chain_list[int(v)] for v in tri] for tri in local_faces.tolist()],
        dtype=np.int64,
    )


def _generate_self_intersection_split_patch_mesh(
    context: HolePatchContext,
    loop: np.ndarray,
    reference_normal: np.ndarray,
    eps: float,
    boundary_sources: Sequence[int],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    平面补片的闭合边界若是 8 字形，先在真实拟合平面投影中找到交叉点，
    将交叉点插入为 Steiner 顶点，再拆成两个简单环分别做尺寸场细化三角化。
    """
    boundary_xyz = np.asarray(context.boundary_xyz, dtype=np.float64)
    loop_list = [int(v) for v in loop.tolist()]
    ordered_xyz = boundary_xyz[np.array(loop_list, dtype=np.int64)]
    centroid, _, u_axis, v_axis = build_plane_frame(ordered_xyz)
    loop_uv = project_to_plane_2d(ordered_xyz, centroid, u_axis, v_axis)
    hit = _find_loop_self_intersection(loop_uv, eps)
    if hit is None:
        return None
    i, j, t, _u, _p_uv = hit

    n0 = int(boundary_xyz.shape[0])
    a0 = int(loop_list[i])
    a1 = int(loop_list[(i + 1) % len(loop_list)])
    p_xyz = (1.0 - t) * boundary_xyz[a0] + t * boundary_xyz[a1]
    vertices_xyz = np.vstack([boundary_xyz, p_xyz.reshape(1, 3)])
    split_idx = n0

    chain_a = [split_idx]
    chain_a.extend(loop_list[(i + 1) : (j + 1)])

    chain_b = [split_idx]
    chain_b.extend(loop_list[(j + 1) :])
    chain_b.extend(loop_list[: (i + 1)])

    rel = vertices_xyz - centroid.reshape(1, 3)
    uv_all = np.column_stack([rel @ u_axis, rel @ v_axis])
    points_uv_list = [
        np.asarray(uv_all[k], dtype=np.float64).reshape(2) for k in range(int(vertices_xyz.shape[0]))
    ]
    points_xyz_list = [
        np.asarray(vertices_xyz[k], dtype=np.float64).reshape(3)
        for k in range(int(vertices_xyz.shape[0]))
    ]
    planar_lift = _make_planar_lift_fn(centroid, u_axis, v_axis)
    sources_ext = [int(boundary_sources[k]) for k in range(n0)] + [-1]
    density_scale = max(0.6, min(1.8, float(context.density_scale)))

    face_blocks: List[np.ndarray] = []
    for chain in (chain_a, chain_b):
        if not _loop_plane_projection_is_simple(vertices_xyz, chain, eps):
            return None
        chain_sources = [sources_ext[int(v)] for v in chain]
        constrained_edges = _derive_constrained_edges(chain, chain_sources)
        try:
            faces = _refine_triangulation_with_size_field(
                chain,
                points_uv_list,
                points_xyz_list,
                planar_lift,
                density_scale,
                context.closure_kind,
                "local_plane",
                constrained_edges,
                eps,
            )
        except RuntimeError:
            faces = _triangulate_chain_in_own_plane(vertices_xyz, chain, eps)
            if faces is None:
                return None
        if faces is None or faces.size == 0:
            return None
        face_blocks.append(faces)
    faces = np.vstack(face_blocks) if face_blocks else np.empty((0, 3), dtype=np.int64)
    if faces.size == 0:
        return None

    points_uv = np.vstack(points_uv_list)
    vertices_xyz = np.vstack(points_xyz_list).astype(np.float64, copy=False)
    n_patch_boundary = n0 + 1
    vertices_xyz = _laplacian_smooth_patch_vertices(
        vertices_xyz,
        faces,
        n_patch_boundary,
        planar_lift,
        points_uv,
        iterations=2,
    )
    faces = _orient_faces_by_local_surface(
        vertices_xyz,
        faces,
        points_uv,
        planar_lift,
        reference_normal,
        eps,
    )
    return _compact_vertices(n0, vertices_xyz, faces)


def _generate_feature_split_patch_mesh(
    context: HolePatchContext,
    loop: np.ndarray,
    loop_uv: np.ndarray,
    lift_fn: Callable[[np.ndarray], np.ndarray],
    boundary_sources: Sequence[int],
    points_uv_list: List[np.ndarray],
    points_xyz_list: List[np.ndarray],
    reference_normal: np.ndarray,
    eps: float,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    multi_patch_cell：整环在拟合平面上投影自交时，沿两特征角点拆成两片简单域分别三角化。
    对应 arrangement 子孔在两面墙之间折返、单平面 8 字边界的情形（如 hole_test2 patch 2）。
    """
    if context.closure_kind not in {
        "multi_patch_cell",
        "opening_carrier_boundary",
        "curve_arc_partition",
    }:
        return None
    if context.feature_point_vertex_indices is None:
        return None
    pos = _feature_positions_on_loop(
        loop.tolist(), boundary_sources, context.feature_point_vertex_indices
    )
    if pos is None:
        return None
    i0, i1 = pos
    n_loop = int(loop.shape[0])
    if n_loop < 4:
        return None
    boundary_xyz = np.asarray(context.boundary_xyz, dtype=np.float64)
    loop_list = [int(v) for v in loop.tolist()]
    refined_sources = [int(boundary_sources[int(v)]) for v in loop_list]

    def _chain_vertex_ids(i_start: int, i_end: int) -> List[int]:
        pos_chain = _chain_along_loop(i_start, i_end, n_loop)
        return [loop_list[p] for p in pos_chain]

    chain_a = _chain_vertex_ids(i0, i1)
    chain_b = _chain_vertex_ids(i1, i0)
    if len(chain_a) < 3 or len(chain_b) < 3:
        return None

    for chain in (chain_a, chain_b):
        if not _loop_plane_projection_is_simple(
            boundary_xyz, chain, eps
        ):
            return None

    # 两链各自闭合三角化会在共享对角线上叠出 4 条共边（非流形）。
    # 改为整环 + 特征对角线约束，一次细化三角化。
    v_diag_a = int(loop_list[int(i0)])
    v_diag_b = int(loop_list[int(i1)])
    ordered_xyz = boundary_xyz[np.array(loop_list, dtype=np.int64)]
    centroid, _, u_axis, v_axis = build_plane_frame(ordered_xyz)
    rel = boundary_xyz - centroid.reshape(1, 3)
    uv_all = np.column_stack([rel @ u_axis, rel @ v_axis])
    points_uv_list = [
        np.asarray(uv_all[k], dtype=np.float64).reshape(2)
        for k in range(int(boundary_xyz.shape[0]))
    ]
    points_xyz_list = [
        np.asarray(boundary_xyz[k], dtype=np.float64).reshape(3)
        for k in range(int(boundary_xyz.shape[0]))
    ]
    planar_lift = (
        context.lift_fn
        if context.lift_fn is not None
        else _make_planar_lift_fn(centroid, u_axis, v_axis)
    )
    density_scale = max(0.6, min(1.8, float(context.density_scale)))
    constrained_edges = _derive_constrained_edges(loop_list, refined_sources)
    if context.seam_constrained_edges:
        constrained_edges = set(constrained_edges) | {
            tuple(sorted((int(a), int(b))))
            for a, b in context.seam_constrained_edges
        }
    diag_key = (
        (v_diag_a, v_diag_b) if v_diag_a < v_diag_b else (v_diag_b, v_diag_a)
    )
    constrained_edges.add(diag_key)
    try:
        faces = _refine_triangulation_with_size_field(
            loop_list,
            points_uv_list,
            points_xyz_list,
            planar_lift,
            density_scale,
            context.closure_kind,
            context.parameterization_kind,
            constrained_edges,
            eps,
        )
    except RuntimeError:
        return None
    if faces is None or faces.size == 0:
        return None

    n0 = int(boundary_xyz.shape[0])
    points_uv = np.vstack(points_uv_list)
    vertices_xyz = np.vstack(points_xyz_list).astype(np.float64, copy=False)
    vertices_xyz = _laplacian_smooth_patch_vertices(
        vertices_xyz,
        faces,
        n0,
        planar_lift,
        points_uv,
        iterations=2,
    )
    faces = _orient_faces_by_local_surface(
        vertices_xyz,
        faces,
        points_uv,
        planar_lift,
        reference_normal,
        eps,
    )
    return _compact_vertices(n0, vertices_xyz, faces)


def _uv_closed_polygon_is_simple(uv: np.ndarray) -> bool:
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    n = int(uv.shape[0])
    if n < 3:
        return True
    loop = list(range(n))
    diag = max(float(np.linalg.norm(np.ptp(uv, axis=0))), 1.0)
    eps = max(1e-12, 1e-10 * diag)
    return _polygon_is_simple(loop, uv, eps)


def _generate_generic_patch_mesh(context: HolePatchContext) -> Tuple[np.ndarray, np.ndarray]:
    boundary_xyz = np.asarray(context.boundary_xyz, dtype=np.float64)
    reference_normal = (
        np.asarray(context.reference_normal, dtype=np.float64)
        if context.reference_normal is not None
        else _newell_normal(boundary_xyz)
    )
    if boundary_xyz.ndim != 2 or boundary_xyz.shape[1] != 3:
        raise ValueError("boundary_points 需要是 (N, 3) 数组")
    n0 = int(boundary_xyz.shape[0])
    if n0 < 3:
        raise ValueError("边界顶点数不足 3")

    if context.boundary_uv is not None:
        boundary_uv = np.asarray(context.boundary_uv, dtype=np.float64)
        if boundary_uv.ndim != 2 or boundary_uv.shape[1] != 2:
            raise ValueError("boundary_points_2d 需要是 (N, 2) 数组")
        if boundary_uv.shape[0] != n0:
            raise ValueError("boundary_points_2d 与 boundary_points 顶点数不一致")

    boundary_sources = (
        [int(x) for x in context.boundary_sources]
        if context.boundary_sources is not None
        else list(range(n0))
    )
    if len(boundary_sources) != n0:
        boundary_sources = list(range(n0))

    collapse_tol = boundary_coincidence_tolerance(boundary_xyz)
    boundary_xyz, boundary_uv, boundary_edges, boundary_sources, seam_edges = (
        _collapse_coincident_boundary_vertices(
            boundary_xyz,
            boundary_uv if context.boundary_uv is not None else None,
            context.boundary_edges,
            boundary_sources,
            context.seam_constrained_edges,
            tol=collapse_tol,
        )
    )
    context.boundary_xyz = boundary_xyz
    context.boundary_edges = boundary_edges
    context.boundary_sources = boundary_sources
    if boundary_uv is not None:
        context.boundary_uv = boundary_uv
    context.seam_constrained_edges = seam_edges
    n0 = int(boundary_xyz.shape[0])
    if n0 < 3:
        raise ValueError("边界顶点数不足 3")

    loop = order_boundary_loop_from_edges(n0, boundary_edges)
    loop_uv, lift_fn, seed_uv = _choose_working_domain(context, loop)

    if signed_polygon_area_2d(loop_uv) < 0.0:
        loop = loop[::-1].copy()
        loop_uv = loop_uv[::-1].copy()
        boundary_sources = [boundary_sources[int(idx)] for idx in loop]
    else:
        boundary_sources = [boundary_sources[int(idx)] for idx in loop]

    points_uv_list = [
        np.asarray(seed_uv[i], dtype=np.float64).reshape(2) for i in range(n0)
    ]
    points_xyz_list = [
        np.asarray(boundary_xyz[i], dtype=np.float64).reshape(3) for i in range(n0)
    ]
    for local_idx, gid in enumerate(loop):
        points_uv_list[int(gid)] = loop_uv[local_idx]

    diag = max(
        float(np.linalg.norm(np.ptp(loop_uv, axis=0))),
        float(np.linalg.norm(np.ptp(boundary_xyz, axis=0))),
        1.0,
    )
    eps = max(1e-12, 1e-10 * diag)
    if abs(signed_polygon_area_2d(loop_uv)) <= eps:
        raise RuntimeError("triangulation: 边界在工作域中退化为近共线")
    if not _polygon_is_simple(loop.tolist(), np.asarray(points_uv_list), eps):
        raise RuntimeError("triangulation: 边界在工作域中存在自交")

    plane_simple = _loop_plane_projection_is_simple(boundary_xyz, loop.tolist(), eps)
    # curve_arc：UV 已验简单；平面投影在汇交处可能假 8 字，禁止平面 Steiner 拆片。
    if not plane_simple and str(context.closure_kind) != "curve_arc_partition":
        split_mesh = _generate_self_intersection_split_patch_mesh(
            context,
            loop,
            reference_normal,
            eps,
            boundary_sources,
        )
        if split_mesh is not None:
            return split_mesh
        split_mesh = _generate_feature_split_patch_mesh(
            context,
            loop,
            loop_uv,
            lift_fn,
            boundary_sources,
            points_uv_list,
            points_xyz_list,
            reference_normal,
            eps,
        )
        if split_mesh is not None:
            return split_mesh

    density_scale = max(0.6, min(1.8, float(context.density_scale)))
    refined_loop = loop.tolist()
    refined_sources = [int(src) for src in boundary_sources]
    constrained_edges = _derive_constrained_edges(refined_loop, refined_sources)
    if context.seam_constrained_edges:
        constrained_edges = set(constrained_edges) | {
            tuple(sorted((int(a), int(b))))
            for a, b in context.seam_constrained_edges
        }
    points_uv = np.vstack(points_uv_list)
    polygon_area = 0.5 * abs(signed_polygon_area_2d(points_uv[np.array(refined_loop, dtype=np.int64)]))
    if polygon_area <= eps:
        raise RuntimeError("triangulation: 子孔工作域面积过小")

    faces = _refine_triangulation_with_size_field(
        refined_loop,
        points_uv_list,
        points_xyz_list,
        lift_fn,
        density_scale,
        context.closure_kind,
        context.parameterization_kind,
        constrained_edges,
        eps,
    )
    points_uv = np.vstack(points_uv_list)
    if faces.size == 0:
        protected = _protected_edge_set(refined_loop, constrained_edges)
        faces, steiner_uv = _constrained_polygon_triangulation(
            refined_loop, points_uv, protected, eps
        )
        if steiner_uv is not None:
            points_uv_list.append(np.asarray(steiner_uv, dtype=np.float64).reshape(2))
            points_xyz_list.append(
                np.asarray(lift_fn(steiner_uv), dtype=np.float64).reshape(3)
            )
            points_uv = np.vstack(points_uv_list)
    vertices_xyz = np.vstack(points_xyz_list).astype(np.float64, copy=False)
    smooth_iters = (
        0
        if context.parameterization_kind
        in {
            "cylinder",
            "cylinder_tangent",
            "cylinder_local_plane",
            "developable_strip",
        }
        else 2
    )
    vertices_xyz = _laplacian_smooth_patch_vertices(
        vertices_xyz,
        faces,
        n0,
        lift_fn,
        points_uv,
        iterations=smooth_iters,
    )
    faces = _orient_faces_by_local_surface(
        vertices_xyz,
        faces,
        points_uv,
        lift_fn,
        reference_normal,
        eps,
    )
    return _compact_vertices(n0, vertices_xyz, faces)


def generate_hole_patch_mesh(context: HolePatchContext) -> Tuple[np.ndarray, np.ndarray]:
    """
    基于补洞上下文生成局部补片。

    该函数是新的核心入口，尽量利用：
    - 参数域边界 `boundary_uv`
    - 2D -> 3D 抬升函数 `lift_fn`
    - 边界来源 `boundary_sources`
    - 闭合类型 `closure_kind`
    - 参数化类型 `parameterization_kind`
    """
    if (
        context.closure_kind
        not in {
            "closed_loop",
            "multi_patch_cell",
            "opening_carrier_boundary",
            "curve_arc_partition",
        }
        and context.open_boundary_count is not None
    ):
        return _triangulate_strip_patch(context)
    return _generate_generic_patch_mesh(context)


def triangulate_hole_patch(
    boundary_points: np.ndarray,
    boundary_edges: np.ndarray,
    *,
    small_angle_deg: float = SMALL_ANGLE_DEG,
    kd_tree: bool = True,
    height_scale: float = EQUILATERAL_HEIGHT_SCALE,
    boundary_points_2d: Optional[np.ndarray] = None,
    lift_point_from_2d: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    boundary_sources: Optional[Sequence[int]] = None,
    closure_kind: str = "closed_loop",
    parameterization_kind: str = "local_plane",
    open_boundary_count: Optional[int] = None,
    feature_point_vertex_indices: Optional[Tuple[int, int]] = None,
    reference_normal: Optional[np.ndarray] = None,
    seam_constrained_edges: Optional[AbstractSet[Tuple[int, int]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    孔洞补片三角化入口。

    额外支持：
    - `boundary_sources`
    - `closure_kind`
    - `parameterization_kind`
    """
    _ = kd_tree
    density_scale = max(0.6, min(1.8, height_scale / EQUILATERAL_HEIGHT_SCALE))
    context = HolePatchContext(
        boundary_xyz=np.asarray(boundary_points, dtype=np.float64),
        boundary_edges=np.asarray(boundary_edges, dtype=np.int64),
        boundary_uv=None
        if boundary_points_2d is None
        else np.asarray(boundary_points_2d, dtype=np.float64),
        lift_fn=lift_point_from_2d,
        boundary_sources=boundary_sources,
        closure_kind=str(closure_kind),
        parameterization_kind=str(parameterization_kind),
        open_boundary_count=None if open_boundary_count is None else int(open_boundary_count),
        feature_point_vertex_indices=feature_point_vertex_indices,
        reference_normal=None if reference_normal is None else np.asarray(reference_normal, dtype=np.float64),
        small_angle_deg=float(small_angle_deg),
        density_scale=float(density_scale),
        seam_constrained_edges=seam_constrained_edges,
    )
    return generate_hole_patch_mesh(context)


def triangulate_ordered_hole_boundary(
    boundary_points_ordered: np.ndarray,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """边界点已按环顺序排列时，自动构造边表并三角化。"""
    boundary_points_ordered = np.asarray(boundary_points_ordered, dtype=np.float64)
    n = int(boundary_points_ordered.shape[0])
    edges = np.array([[i, (i + 1) % n] for i in range(n)], dtype=np.int64)
    return triangulate_hole_patch(boundary_points_ordered, edges, **kwargs)


