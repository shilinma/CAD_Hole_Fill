#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
解析面驱动的局部交线恢复。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from .surface_fitting import (
    SurfaceFit,
    is_analytic_surface_type,
    is_transition_surface_type,
    project_point_to_surface_pair,
)


@dataclass
class IntersectionCurve:
    patch_pair: Tuple[int, int]
    curve_kind: str
    curve_points: np.ndarray
    endpoints_on_boundary: np.ndarray
    curve_confidence: str
    source_surface_types: Tuple[str, str]
    endpoint_vertex_indices: Tuple[int, int] = (-1, -1)


@dataclass
class AnalyticCurve:
    """两解析面求交的完整几何对象（可无限延伸）。"""

    patch_pair: Tuple[int, int]
    kind: str
    fit_a: SurfaceFit
    fit_b: SurfaceFit
    line_point: Optional[np.ndarray] = None
    line_dir: Optional[np.ndarray] = None
    circle_center: Optional[np.ndarray] = None
    circle_normal: Optional[np.ndarray] = None
    circle_radius: Optional[float] = None


@dataclass
class BoundedCurveSegment:
    """解析交线裁剪到孔洞后的有界段。"""

    analytic: AnalyticCurve
    t_start: float
    t_end: float
    curve_points: np.ndarray
    boundary_vertex_indices: Tuple[int, int]
    clip_confidence: str
    start_xyz: Optional[np.ndarray] = None
    end_xyz: Optional[np.ndarray] = None


def _safe_normalize(vec: np.ndarray) -> np.ndarray:
    nrm = float(np.linalg.norm(vec))
    if nrm < 1e-15:
        return np.zeros_like(vec, dtype=np.float64)
    return np.asarray(vec, dtype=np.float64) / nrm


def _orthonormal_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = np.asarray(normal, dtype=np.float64)
    n = n / (float(np.linalg.norm(n)) + 1e-15)
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(n, ref))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = np.cross(n, ref)
    u = u / (float(np.linalg.norm(u)) + 1e-15)
    v = np.cross(n, u)
    v = v / (float(np.linalg.norm(v)) + 1e-15)
    return u, v


def _plane_plane_line(
    fit_a: SurfaceFit, fit_b: SurfaceFit
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    p1 = np.asarray(fit_a.surface_params["point"], dtype=np.float64)
    n1 = np.asarray(fit_a.surface_params["normal"], dtype=np.float64)
    p2 = np.asarray(fit_b.surface_params["point"], dtype=np.float64)
    n2 = np.asarray(fit_b.surface_params["normal"], dtype=np.float64)
    direction = np.cross(n1, n2)
    ld = float(np.linalg.norm(direction))
    if ld < 1e-12:
        return None
    direction = direction / ld
    d1 = float(np.dot(n1, p1))
    d2 = float(np.dot(n2, p2))
    # 取交线中距离原点最近的一点，解:
    #   n1 · x = d1
    #   n2 · x = d2
    #   direction · x = 0
    a = np.vstack([n1, n2, direction])
    b = np.array([d1, d2, 0.0], dtype=np.float64)
    try:
        point, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
    except np.linalg.LinAlgError:
        point = 0.5 * (p1 + p2)
    return point, direction


def _plane_sphere_circle(
    plane_fit: SurfaceFit, sphere_fit: SurfaceFit
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    plane_point = np.asarray(plane_fit.surface_params["point"], dtype=np.float64)
    normal = np.asarray(plane_fit.surface_params["normal"], dtype=np.float64)
    center = np.asarray(sphere_fit.surface_params["center"], dtype=np.float64)
    radius = float(sphere_fit.surface_params["radius"])
    signed = float(np.dot(center - plane_point, normal))
    if abs(signed) >= radius:
        return None
    circle_center = center - signed * normal
    circle_radius = float(np.sqrt(max(radius * radius - signed * signed, 1e-12)))
    return circle_center, normal, circle_radius


def _sphere_sphere_circle(
    fit_a: SurfaceFit, fit_b: SurfaceFit
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    c1 = np.asarray(fit_a.surface_params["center"], dtype=np.float64)
    c2 = np.asarray(fit_b.surface_params["center"], dtype=np.float64)
    r1 = float(fit_a.surface_params["radius"])
    r2 = float(fit_b.surface_params["radius"])
    delta = c2 - c1
    d = float(np.linalg.norm(delta))
    if d < 1e-12 or d > r1 + r2 or d < abs(r1 - r2):
        return None
    normal = delta / d
    a = (d * d + r1 * r1 - r2 * r2) / (2.0 * d)
    center = c1 + a * normal
    radius = float(np.sqrt(max(r1 * r1 - a * a, 1e-12)))
    return center, normal, radius


def _normalize_angle(theta: float) -> float:
    out = float(theta)
    while out < 0.0:
        out += 2.0 * np.pi
    while out >= 2.0 * np.pi:
        out -= 2.0 * np.pi
    return out


def _angle_on_circle(
    point: np.ndarray, center: np.ndarray, u_axis: np.ndarray, v_axis: np.ndarray
) -> float:
    rel = point - center
    return _normalize_angle(float(np.arctan2(np.dot(rel, v_axis), np.dot(rel, u_axis))))


def _circle_arc_sweep_delta(
    center: np.ndarray,
    normal: np.ndarray,
    radius: float,
    start: np.ndarray,
    end: np.ndarray,
    guide: Optional[np.ndarray],
    n_probe: int,
    *,
    short_arc_only: bool = False,
) -> float:
    """
    两角点之间的有向扫角 delta（弧度）。

    - 默认取 |delta|<=pi 的短弧；若 ``short_arc_only`` 则恒为短弧（孔洞交线夹在角点之间，禁止绕远路）。
    - 否则仅当 guide 明确更贴近长弧中点时才取长弧。
    """
    u_axis, v_axis = _orthonormal_basis(normal)
    start_angle = _angle_on_circle(start, center, u_axis, v_axis)
    end_angle = _angle_on_circle(end, center, u_axis, v_axis)

    def sample_with_delta(delta: float) -> np.ndarray:
        ts = np.linspace(0.0, 1.0, max(2, int(n_probe)), dtype=np.float64)
        angles = start_angle + delta * ts
        pts = []
        for ang in angles:
            pts.append(center + radius * (np.cos(ang) * u_axis + np.sin(ang) * v_axis))
        return np.array(pts, dtype=np.float64)

    delta_short = end_angle - start_angle
    if delta_short > np.pi:
        delta_short -= 2.0 * np.pi
    elif delta_short < -np.pi:
        delta_short += 2.0 * np.pi
    delta_long = delta_short - 2.0 * np.pi if delta_short > 0 else delta_short + 2.0 * np.pi

    if short_arc_only or guide is None:
        return float(delta_short)

    guide_angle = _angle_on_circle(guide, center, u_axis, v_axis)
    candidates = [sample_with_delta(delta_short), sample_with_delta(delta_long)]
    scores: list[float] = []
    for pts in candidates:
        mid = pts[len(pts) // 2]
        mid_angle = _angle_on_circle(mid, center, u_axis, v_axis)
        score = min(
            abs(mid_angle - guide_angle),
            abs(mid_angle - guide_angle + 2.0 * np.pi),
            abs(mid_angle - guide_angle - 2.0 * np.pi),
        )
        scores.append(float(score))
    score_short, score_long = scores[0], scores[1]
    # 仅当 guide 明显更贴近长弧中点时才取长弧（阈值偏严，减少误判绕远路）。
    angular_margin = 0.85
    if score_long + angular_margin < score_short:
        return float(delta_long)
    return float(delta_short)


def _sample_circle_arc(
    center: np.ndarray,
    normal: np.ndarray,
    radius: float,
    start: np.ndarray,
    end: np.ndarray,
    guide: Optional[np.ndarray],
    n_samples: int,
    *,
    short_arc_only: bool = False,
) -> np.ndarray:
    delta = _circle_arc_sweep_delta(
        center,
        normal,
        radius,
        start,
        end,
        guide,
        max(8, n_samples),
        short_arc_only=short_arc_only,
    )
    u_axis, v_axis = _orthonormal_basis(normal)
    start_angle = _angle_on_circle(start, center, u_axis, v_axis)
    ts = np.linspace(0.0, 1.0, n_samples, dtype=np.float64)
    angles = start_angle + delta * ts
    pts = []
    for ang in angles:
        pts.append(center + radius * (np.cos(ang) * u_axis + np.sin(ang) * v_axis))
    return np.array(pts, dtype=np.float64)


def _project_endpoints_for_line(
    line_point: np.ndarray,
    line_dir: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    def project(p: np.ndarray) -> np.ndarray:
        t = float(np.dot(p - line_point, line_dir))
        return line_point + t * line_dir

    return project(start), project(end)


def _sample_line(start: np.ndarray, end: np.ndarray, n_samples: int) -> np.ndarray:
    ts = np.linspace(0.0, 1.0, n_samples, dtype=np.float64).reshape(-1, 1)
    return start.reshape(1, 3) * (1.0 - ts) + end.reshape(1, 3) * ts


def _sample_general_curve(
    fit_a: SurfaceFit,
    fit_b: SurfaceFit,
    start: np.ndarray,
    end: np.ndarray,
    guide: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """
    沿两端点 **欧氏弦** 做凸组合再投到两曲面交线上。

    这样采样始终对应 t∈[0,1] 上 ``(1-t)*start + t*end`` 一族，不会沿二次 Bézier
    或错误圆弧支路「弯出」两角点之间的有效段落；guide 仅保留接口兼容，不使用。
    """
    _ = guide
    start = np.asarray(start, dtype=np.float64).reshape(3)
    end = np.asarray(end, dtype=np.float64).reshape(3)
    ts = np.linspace(0.0, 1.0, n_samples, dtype=np.float64)
    pts = []
    for t in ts:
        p = (1.0 - t) * start + t * end
        p = _project_to_surface_system([fit_a, fit_b], p)
        pts.append(p)
    return np.array(pts, dtype=np.float64)


def _ordered_surface_pair_kind(type_a: str, type_b: str) -> str:
    ordered = tuple(sorted((str(type_a), str(type_b))))
    return f"{ordered[0]}_{ordered[1]}"


def _surface_residual_and_gradient(
    fit: SurfaceFit, point: np.ndarray
) -> Tuple[float, np.ndarray]:
    p = np.asarray(point, dtype=np.float64)
    params = fit.surface_params

    if fit.surface_type == "plane":
        plane_point = np.asarray(params["point"], dtype=np.float64)
        normal = _safe_normalize(np.asarray(params["normal"], dtype=np.float64))
        return float(np.dot(p - plane_point, normal)), normal

    if fit.surface_type == "sphere":
        center = np.asarray(params["center"], dtype=np.float64)
        radius = float(params["radius"])
        direction = p - center
        ln = float(np.linalg.norm(direction))
        if ln < 1e-15:
            return -radius, np.array([1.0, 0.0, 0.0], dtype=np.float64)
        grad = direction / ln
        return ln - radius, grad

    if fit.surface_type == "cylinder":
        axis_point = np.asarray(params["point"], dtype=np.float64)
        axis = _safe_normalize(np.asarray(params["axis"], dtype=np.float64))
        radius = float(params["radius"])
        rel = p - axis_point
        axial = float(np.dot(rel, axis))
        foot = axis_point + axial * axis
        radial = p - foot
        lr = float(np.linalg.norm(radial))
        if lr < 1e-15:
            grad = np.cross(axis, np.array([1.0, 0.0, 0.0], dtype=np.float64))
            if float(np.linalg.norm(grad)) < 1e-15:
                grad = np.cross(axis, np.array([0.0, 1.0, 0.0], dtype=np.float64))
            grad = _safe_normalize(grad)
            return -radius, grad
        grad = radial / lr
        return lr - radius, grad

    if fit.surface_type == "cone":
        apex = np.asarray(params["apex"], dtype=np.float64)
        axis = _safe_normalize(np.asarray(params["axis"], dtype=np.float64))
        half_angle = float(params["half_angle"])
        rel = p - apex
        axial = float(np.dot(rel, axis))
        foot = apex + axial * axis
        radial = p - foot
        lr = float(np.linalg.norm(radial))
        radial_grad = radial / lr if lr > 1e-15 else _safe_normalize(
            np.cross(axis, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        )
        grad = radial_grad - np.sign(axial if abs(axial) > 1e-12 else 1.0) * np.tan(half_angle) * axis
        return lr - abs(axial) * np.tan(half_angle), _safe_normalize(grad)

    return 0.0, np.zeros(3, dtype=np.float64)


def _project_to_surface_system(
    fits: Sequence[SurfaceFit],
    point: np.ndarray,
    *,
    iterations: int = 10,
) -> np.ndarray:
    p = np.asarray(point, dtype=np.float64).copy()
    if not fits:
        return p
    for _ in range(max(1, int(iterations))):
        residuals: List[float] = []
        jac_rows: List[np.ndarray] = []
        for fit in fits:
            residual, grad = _surface_residual_and_gradient(fit, p)
            if float(np.linalg.norm(grad)) < 1e-12:
                continue
            residuals.append(float(residual))
            jac_rows.append(np.asarray(grad, dtype=np.float64))
        if not jac_rows:
            break
        j = np.vstack(jac_rows)
        r = np.asarray(residuals, dtype=np.float64)
        try:
            delta, _, _, _ = np.linalg.lstsq(j, -r, rcond=None)
        except np.linalg.LinAlgError:
            break
        p = p + delta
        if float(np.linalg.norm(delta)) < 1e-9:
            break
    return p


def _fit_support_diag(fits: Sequence[SurfaceFit]) -> float:
    points = [fit.support_points for fit in fits if len(fit.support_points) > 0]
    if not points:
        return 1.0
    merged = np.vstack(points)
    diag = float(np.linalg.norm(np.ptp(merged, axis=0)))
    return max(diag, 1e-12)


def _min_total_curve_samples(min_samples: int) -> int:
    """Minimum total polyline vertices (>=2). ``min_samples=0`` => endpoint-only (2)."""
    return 2 if int(min_samples) <= 0 else int(max(2, min_samples))


def feature_curve_sample_count(
    curve_length: float,
    reference_step: float,
    *,
    min_interior_samples: int = 0,
    max_samples: int = 96,
) -> int:
    """
    Total vertices along a bounded feature curve from reference spacing.

    ``min_interior_samples=0`` allows straight segments with no interior samples.
    """
    min_total = _min_total_curve_samples(min_interior_samples)
    if not np.isfinite(curve_length) or curve_length <= 0:
        return min_total
    step = float(reference_step)
    if not np.isfinite(step) or step <= 1e-15:
        return min_total
    if float(curve_length) <= 1.35 * step:
        return 2
    n = int(np.ceil(float(curve_length) / step)) + 1
    return int(np.clip(n, min_total, max_samples))


def _sample_count_from_reference_step(
    curve_length: float,
    reference_step: float,
    *,
    min_samples: int = 0,
    max_samples: int = 96,
) -> int:
    """
    按固定目标步长 ``reference_step`` 估计采样点数：约 ``ceil(L/step)+1``，
    用于以孔边角点处网格尺度（公共边或局部三角边长）控制交线密度。
    """
    return feature_curve_sample_count(
        curve_length,
        reference_step,
        min_interior_samples=min_samples,
        max_samples=max_samples,
    )


def _adaptive_sample_count(
    curve_length: float,
    support_diag: float,
    *,
    min_samples: int = 0,
    max_samples: int = 96,
    segments_per_diag: float = 24.0,
) -> int:
    """
    按估计弧长与局部尺度（两 patch 支撑点包围盒对角线）决定采样点数，避免过密或过稀。

    目标步长约 ``support_diag / segments_per_diag``；点数 ``≈ ceil(L / step) + 1`` 并夹在
    ``[min_total, max_samples]``，其中 ``min_samples=0`` 表示允许仅保留两端点。
    """
    min_total = _min_total_curve_samples(min_samples)
    if not np.isfinite(curve_length) or curve_length <= 0:
        return min_total
    if support_diag < 1e-12:
        support_diag = 1.0
    target_step = max(1e-8 * support_diag, support_diag / float(segments_per_diag))
    n = int(np.ceil(curve_length / target_step)) + 1
    return int(np.clip(n, min_total, max_samples))


def _general_curve_length_estimate(
    fit_a: SurfaceFit,
    fit_b: SurfaceFit,
    start: np.ndarray,
    end: np.ndarray,
    guide: np.ndarray,
) -> float:
    """与 ``_sample_general_curve`` 一致：弦中点投影后折线长度上界（不用 guide，避免估长跑偏）。"""
    _ = guide
    start = np.asarray(start, dtype=np.float64).reshape(3)
    end = np.asarray(end, dtype=np.float64).reshape(3)
    mid = _project_to_surface_system([fit_a, fit_b], 0.5 * (start + end))
    c0 = float(np.linalg.norm(start - mid))
    c1 = float(np.linalg.norm(mid - end))
    chord = float(np.linalg.norm(start - end))
    return float(max(0.5 * (c0 + c1) + 0.5 * chord, chord))


def _snap_curve_points_to_corner_vertices(
    pts: np.ndarray,
    corner_start_xyz: np.ndarray,
    corner_end_xyz: np.ndarray,
    endpoint_vertex_indices: Tuple[int, int],
) -> np.ndarray:
    """
    将交线折线首尾强制为拓扑角点在网格上的坐标。

    采样/圆弧参数化/双边投影会在端点产生微小漂移，与子孔边界使用的顶点坐标不一致；
    强制对齐可避免「管段穿出角点球」及子孔缝不闭合。
    """
    out = np.asarray(pts, dtype=np.float64, copy=True)
    if out.ndim != 2 or out.shape[0] == 0 or out.shape[1] != 3:
        return out
    e0, e1 = int(endpoint_vertex_indices[0]), int(endpoint_vertex_indices[1])
    if e0 >= 0:
        out[0] = np.asarray(corner_start_xyz, dtype=np.float64).reshape(3)
    if e1 >= 0:
        out[-1] = np.asarray(corner_end_xyz, dtype=np.float64).reshape(3)
    return out


def recover_curve_between_points(
    fit_a: SurfaceFit,
    fit_b: SurfaceFit,
    start_point: np.ndarray,
    end_point: np.ndarray,
    guide_point: np.ndarray,
    *,
    n_samples: Optional[int] = None,
    min_samples: int = 0,
    max_samples: int = 96,
    segments_per_diag: float = 24.0,
    endpoint_vertex_indices: Tuple[int, int] = (-1, -1),
    chord_vs_reference_rel_tol: float = 0.45,
    intersection_sampling_reference_step: Optional[float] = None,
) -> IntersectionCurve:
    pair = tuple(sorted((int(fit_a.patch_label), int(fit_b.patch_label))))
    source_types = (fit_a.surface_type, fit_b.surface_type)
    start = np.asarray(start_point, dtype=np.float64)
    end = np.asarray(end_point, dtype=np.float64)
    guide = np.asarray(guide_point, dtype=np.float64)
    pair_kind = _ordered_surface_pair_kind(fit_a.surface_type, fit_b.surface_type)
    diag = _fit_support_diag([fit_a, fit_b])

    def n_for(length: float) -> int:
        min_total = _min_total_curve_samples(min_samples)
        ref_step = (
            float(intersection_sampling_reference_step)
            if intersection_sampling_reference_step is not None
            and float(intersection_sampling_reference_step) > 1e-15
            else None
        )
        if (
            ref_step is not None
            and np.isfinite(ref_step)
            and np.isfinite(length)
            and length > 0
        ):
            if float(length) <= 1.35 * ref_step:
                return 2
            rel = abs(float(length) - ref_step) / ref_step
            if rel <= float(chord_vs_reference_rel_tol) and pair_kind == "plane_plane":
                return 2
        if ref_step is not None and np.isfinite(ref_step):
            adaptive = _sample_count_from_reference_step(
                length,
                ref_step,
                min_samples=int(min_samples),
                max_samples=max_samples,
            )
        else:
            adaptive = _adaptive_sample_count(
                length,
                diag,
                min_samples=int(min_samples),
                max_samples=max_samples,
                segments_per_diag=segments_per_diag,
            )
        if n_samples is not None:
            return int(np.clip(max(min_total, int(n_samples)), min_total, max_samples))
        return adaptive

    if is_transition_surface_type(fit_a.surface_type) or is_transition_surface_type(fit_b.surface_type):
        start_p = _project_to_surface_system([fit_a, fit_b], start)
        end_p = _project_to_surface_system([fit_a, fit_b], end)
        guide_p = _project_to_surface_system([fit_a, fit_b], guide)
        length = _general_curve_length_estimate(fit_a, fit_b, start_p, end_p, guide_p)
        pts = _sample_general_curve(
            fit_a, fit_b, start_p, end_p, guide_p, n_for(length)
        )
        pts = _snap_curve_points_to_corner_vertices(pts, start, end, endpoint_vertex_indices)
        return IntersectionCurve(
            patch_pair=pair,
            curve_kind="transition_guided_curve",
            curve_points=pts,
            endpoints_on_boundary=np.vstack([pts[0], pts[-1]]),
            curve_confidence="low",
            source_surface_types=source_types,
            endpoint_vertex_indices=tuple(int(x) for x in endpoint_vertex_indices),
        )

    if fit_a.surface_type == "plane" and fit_b.surface_type == "plane":
        line = _plane_plane_line(fit_a, fit_b)
        if line is not None:
            line_point, line_dir = line
            start_p, end_p = _project_endpoints_for_line(line_point, line_dir, start, end)
            length = float(np.linalg.norm(end_p - start_p))
            n_pts = n_for(length)
            pts = _sample_line(start_p, end_p, n_pts)
            pts = _snap_curve_points_to_corner_vertices(pts, start, end, endpoint_vertex_indices)
            # After endpoint snap, resample on the authoritative chord so interior
            # points cannot drift off the straight feature segment.
            if n_pts <= 2:
                pts = np.vstack([pts[0], pts[-1]])
            else:
                pts = _sample_line(pts[0], pts[-1], n_pts)
            return IntersectionCurve(
                patch_pair=pair,
                curve_kind="line",
                curve_points=pts,
                endpoints_on_boundary=np.vstack([pts[0], pts[-1]]),
                curve_confidence="high",
                source_surface_types=source_types,
                endpoint_vertex_indices=tuple(int(x) for x in endpoint_vertex_indices),
            )

    circle = None
    if fit_a.surface_type == "plane" and fit_b.surface_type == "sphere":
        circle = _plane_sphere_circle(fit_a, fit_b)
    elif fit_a.surface_type == "sphere" and fit_b.surface_type == "plane":
        circle = _plane_sphere_circle(fit_b, fit_a)
    elif fit_a.surface_type == "sphere" and fit_b.surface_type == "sphere":
        circle = _sphere_sphere_circle(fit_a, fit_b)

    if circle is not None:
        center, normal, radius = circle
        start_p = project_point_to_surface_pair(fit_a, fit_b, start)
        end_p = project_point_to_surface_pair(fit_a, fit_b, end)
        guide_p = project_point_to_surface_pair(fit_a, fit_b, guide)
        delta = _circle_arc_sweep_delta(
            center,
            normal,
            radius,
            start_p,
            end_p,
            guide_p,
            16,
            short_arc_only=True,
        )
        length = abs(float(radius) * float(delta))
        pts = _sample_circle_arc(
            center,
            normal,
            radius,
            start_p,
            end_p,
            guide_p,
            n_for(length),
            short_arc_only=True,
        )
        pts = _snap_curve_points_to_corner_vertices(pts, start, end, endpoint_vertex_indices)
        return IntersectionCurve(
            patch_pair=pair,
            curve_kind="circle_arc",
            curve_points=pts,
            endpoints_on_boundary=np.vstack([pts[0], pts[-1]]),
            curve_confidence="high",
            source_surface_types=source_types,
            endpoint_vertex_indices=tuple(int(x) for x in endpoint_vertex_indices),
        )

    confidence = (
        "medium"
        if is_analytic_surface_type(fit_a.surface_type)
        and is_analytic_surface_type(fit_b.surface_type)
        else "low"
    )
    start_p = _project_to_surface_system([fit_a, fit_b], start)
    end_p = _project_to_surface_system([fit_a, fit_b], end)
    guide_p = _project_to_surface_system([fit_a, fit_b], guide)
    length = _general_curve_length_estimate(fit_a, fit_b, start_p, end_p, guide_p)
    pts = _sample_general_curve(fit_a, fit_b, start_p, end_p, guide_p, n_for(length))
    pts = _snap_curve_points_to_corner_vertices(pts, start, end, endpoint_vertex_indices)
    return IntersectionCurve(
        patch_pair=pair,
        curve_kind=f"{pair_kind}_curve" if confidence == "medium" else "surface_pair_curve",
        curve_points=pts,
        endpoints_on_boundary=np.vstack([pts[0], pts[-1]]),
        curve_confidence=confidence,
        source_surface_types=source_types,
        endpoint_vertex_indices=tuple(int(x) for x in endpoint_vertex_indices),
    )


def analytic_intersection(
    fit_a: SurfaceFit,
    fit_b: SurfaceFit,
) -> Optional[AnalyticCurve]:
    """仅依赖 SurfaceFit 求两解析面的交线/交圆，不指定端点。"""
    pair = tuple(sorted((int(fit_a.patch_label), int(fit_b.patch_label))))
    if is_transition_surface_type(fit_a.surface_type) or is_transition_surface_type(
        fit_b.surface_type
    ):
        return None
    if not is_analytic_surface_type(fit_a.surface_type) or not is_analytic_surface_type(
        fit_b.surface_type
    ):
        return None

    if fit_a.surface_type == "plane" and fit_b.surface_type == "plane":
        line = _plane_plane_line(fit_a, fit_b)
        if line is None:
            return None
        lp, ld = line
        return AnalyticCurve(
            patch_pair=pair,
            kind="line",
            fit_a=fit_a,
            fit_b=fit_b,
            line_point=np.asarray(lp, dtype=np.float64),
            line_dir=_safe_normalize(np.asarray(ld, dtype=np.float64)),
        )

    circle = None
    if fit_a.surface_type == "plane" and fit_b.surface_type == "sphere":
        circle = _plane_sphere_circle(fit_a, fit_b)
    elif fit_a.surface_type == "sphere" and fit_b.surface_type == "plane":
        circle = _plane_sphere_circle(fit_b, fit_a)
    elif fit_a.surface_type == "sphere" and fit_b.surface_type == "sphere":
        circle = _sphere_sphere_circle(fit_a, fit_b)

    if circle is not None:
        center, normal, radius = circle
        return AnalyticCurve(
            patch_pair=pair,
            kind="circle",
            fit_a=fit_a,
            fit_b=fit_b,
            circle_center=np.asarray(center, dtype=np.float64),
            circle_normal=_safe_normalize(np.asarray(normal, dtype=np.float64)),
            circle_radius=float(radius),
        )

    if fit_a.surface_type == "plane" and fit_b.surface_type == "cylinder":
        plane_fit, cyl_fit = fit_a, fit_b
    elif fit_a.surface_type == "cylinder" and fit_b.surface_type == "plane":
        plane_fit, cyl_fit = fit_b, fit_a
    else:
        plane_fit = cyl_fit = None

    if plane_fit is not None and cyl_fit is not None:
        n = _safe_normalize(np.asarray(plane_fit.surface_params["normal"], dtype=np.float64))
        axis = _safe_normalize(np.asarray(cyl_fit.surface_params["axis"], dtype=np.float64))
        direction = np.cross(n, axis)
        dn = float(np.linalg.norm(direction))
        if dn > 1e-10:
            direction = direction / dn
            axis_pt = np.asarray(cyl_fit.surface_params["point"], dtype=np.float64)
            plane_pt = np.asarray(plane_fit.surface_params["point"], dtype=np.float64)
            d_plane = float(np.dot(n, plane_pt))
            denom = float(np.dot(axis, n))
            if abs(denom) > 1e-10:
                t_axis = float(np.dot(axis_pt - plane_pt, n) / denom)
                line_point = axis_pt + t_axis * axis
            else:
                line_point = axis_pt
            line_point = line_point + (d_plane - float(np.dot(n, line_point))) * n
            return AnalyticCurve(
                patch_pair=pair,
                kind="line",
                fit_a=fit_a,
                fit_b=fit_b,
                line_point=np.asarray(line_point, dtype=np.float64),
                line_dir=direction,
            )

    return AnalyticCurve(
        patch_pair=pair,
        kind="general",
        fit_a=fit_a,
        fit_b=fit_b,
    )


def _analytic_curve_as_line(
    curve: AnalyticCurve,
    guide_point: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    guide = np.asarray(guide_point, dtype=np.float64).reshape(3)
    general = _general_analytic_as_line(curve, guide)
    if general is not None:
        lp = np.asarray(general[0], dtype=np.float64)
        ld = _safe_normalize(np.asarray(general[1], dtype=np.float64))
        return lp, ld
    if (
        curve.kind == "line"
        and curve.line_point is not None
        and curve.line_dir is not None
    ):
        lp = np.asarray(curve.line_point, dtype=np.float64)
        ld = _safe_normalize(np.asarray(curve.line_dir, dtype=np.float64))
        t_guide = float(np.dot(guide - lp, ld))
        return lp + t_guide * ld, ld
    return None


def intersect_analytic_curves(
    ac_a: AnalyticCurve,
    ac_b: AnalyticCurve,
    *,
    guide_point: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """两解析交线（直线或局部直线近似）的交点。"""
    guide = (
        np.asarray(guide_point, dtype=np.float64).reshape(3)
        if guide_point is not None
        else np.zeros(3, dtype=np.float64)
    )
    line_a = _analytic_curve_as_line(ac_a, guide)
    line_b = _analytic_curve_as_line(ac_b, guide)
    if line_a is None or line_b is None:
        return None
    p1, d1 = line_a
    p2, d2 = line_b
    cross = np.cross(d1, d2)
    if float(np.linalg.norm(cross)) < 1e-10:
        return None
    try:
        ts, _, _, _ = np.linalg.lstsq(np.column_stack([d1, -d2]), p2 - p1, rcond=None)
        return np.asarray(p1 + float(ts[0]) * d1, dtype=np.float64)
    except np.linalg.LinAlgError:
        return None


def _intersection_tangent_at(
    fit_a: SurfaceFit,
    fit_b: SurfaceFit,
    point: np.ndarray,
) -> np.ndarray:
    _, ga = _surface_residual_and_gradient(fit_a, point)
    _, gb = _surface_residual_and_gradient(fit_b, point)
    return _safe_normalize(np.cross(ga, gb))


def _general_analytic_as_line(
    curve: AnalyticCurve,
    hole_center: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    p0 = _project_to_surface_system([curve.fit_a, curve.fit_b], hole_center)
    direction = _intersection_tangent_at(curve.fit_a, curve.fit_b, p0)
    if float(np.linalg.norm(direction)) < 1e-12:
        return None
    return p0, direction


def _loop_points(vertices: np.ndarray, loop: Sequence[int]) -> np.ndarray:
    return np.asarray(vertices[np.asarray(loop, dtype=np.int64)], dtype=np.float64)


def _point_in_loop_polygon_3d(
    vertices: np.ndarray,
    loop: Sequence[int],
    point: np.ndarray,
) -> bool:
    pts = _loop_points(vertices, loop)
    if pts.shape[0] < 3:
        return False
    center = np.mean(pts, axis=0)
    rel = pts - center
    _, _, vh = np.linalg.svd(rel, full_matrices=False)
    u_axis, v_axis = vh[0], vh[1]
    poly = np.column_stack([rel @ u_axis, rel @ v_axis])
    p2 = np.array(
        [float(np.dot(point - center, u_axis)), float(np.dot(point - center, v_axis))],
        dtype=np.float64,
    )
    x, y = float(p2[0]), float(p2[1])
    inside = False
    n = poly.shape[0]
    for i in range(n):
        x0, y0 = float(poly[i, 0]), float(poly[i, 1])
        x1, y1 = float(poly[(i + 1) % n, 0]), float(poly[(i + 1) % n, 1])
        if ((y0 > y) != (y1 > y)) and (
            x < (x1 - x0) * (y - y0) / (y1 - y0 + 1e-30) + x0
        ):
            inside = not inside
    return inside


def _nearest_loop_vertex(
    vertices: np.ndarray,
    loop: Sequence[int],
    point: np.ndarray,
) -> int:
    pts = np.asarray(loop, dtype=np.int64)
    coords = vertices[pts]
    dists = np.linalg.norm(coords - np.asarray(point, dtype=np.float64).reshape(1, 3), axis=1)
    return int(pts[int(np.argmin(dists))])


def _line_segment_closest_parameters(
    line_point: np.ndarray,
    line_dir: np.ndarray,
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    eps: float = 1e-9,
    max_distance: float = 1e-6,
) -> List[float]:
    """直线 P+tD 与线段 AB 的交点参数 t（若有）。"""
    d = np.asarray(line_dir, dtype=np.float64)
    a = np.asarray(seg_a, dtype=np.float64)
    b = np.asarray(seg_b, dtype=np.float64)
    s = b - a
    cross_ds = np.cross(d, s)
    denom = float(np.dot(cross_ds, cross_ds))
    if denom < eps * eps:
        # 平行：只有线段确实贴近交线时，才把端点投影为裁剪候选。
        dist_a = float(np.linalg.norm(np.cross(a - line_point, d)))
        dist_b = float(np.linalg.norm(np.cross(b - line_point, d)))
        if min(dist_a, dist_b) > max(float(max_distance), eps):
            return []
        t0 = float(np.dot(a - line_point, d))
        t1 = float(np.dot(b - line_point, d))
        out = [t0, t1]
        return sorted(out)

    t = float(np.dot(np.cross(a - line_point, s), cross_ds) / denom)
    u = float(np.dot(np.cross(a - line_point, d), cross_ds) / denom)
    if -1e-6 <= u <= 1.0 + 1e-6:
        p_line = line_point + t * d
        p_seg = a + u * s
        if float(np.linalg.norm(p_line - p_seg)) > max(float(max_distance), eps):
            return []
        return [t]
    return []


def _collect_line_clip_parameters(
    line_point: np.ndarray,
    line_dir: np.ndarray,
    vertices: np.ndarray,
    loop: Sequence[int],
    tol: float,
) -> List[float]:
    ts: List[float] = []
    n = len(loop)
    for i in range(n):
        va = int(loop[i])
        vb = int(loop[(i + 1) % n])
        ts.extend(
            _line_segment_closest_parameters(
                line_point,
                line_dir,
                vertices[va],
                vertices[vb],
                max_distance=tol,
            )
        )
    for vi in loop:
        p = vertices[int(vi)]
        foot = line_point + float(np.dot(p - line_point, line_dir)) * line_dir
        if float(np.linalg.norm(p - foot)) <= tol:
            ts.append(float(np.dot(p - line_point, line_dir)))
    return ts


def _all_line_segment_intervals(
    ts: Sequence[float],
    line_point: np.ndarray,
    line_dir: np.ndarray,
    hole_center: np.ndarray,
    vertices: np.ndarray,
    loop: Sequence[int],
) -> List[Tuple[float, float]]:
    if not ts:
        return []
    t_center = float(np.dot(hole_center - line_point, line_dir))
    uniq = sorted(set(float(t) for t in ts))
    if len(uniq) == 1:
        half = max(
            float(np.linalg.norm(_loop_points(vertices, loop) - hole_center, axis=1).max()) * 0.5,
            1e-6,
        )
        return [(t_center - half, t_center + half)]

    intervals: List[Tuple[float, float]] = []
    for i in range(len(uniq) - 1):
        t0, t1 = uniq[i], uniq[i + 1]
        if t0 > t1:
            t0, t1 = t1, t0
        if t1 - t0 < 1e-12:
            continue
        mid = line_point + 0.5 * (t0 + t1) * line_dir
        if _point_in_loop_polygon_3d(vertices, loop, mid):
            intervals.append((t0, t1))
    if intervals:
        return intervals

    # fallback: single interval bracketing hole center
    one = _select_line_segment_interval(
        ts, line_point, line_dir, hole_center, vertices, loop
    )
    return [one] if one is not None else []


def _select_line_segment_interval(
    ts: Sequence[float],
    line_point: np.ndarray,
    line_dir: np.ndarray,
    hole_center: np.ndarray,
    vertices: np.ndarray,
    loop: Sequence[int],
) -> Optional[Tuple[float, float]]:
    if not ts:
        return None
    t_center = float(np.dot(hole_center - line_point, line_dir))
    uniq = sorted(set(float(t) for t in ts))
    if len(uniq) == 1:
        half = max(
            float(np.linalg.norm(_loop_points(vertices, loop) - hole_center, axis=1).max()) * 0.5,
            1e-6,
        )
        return t_center - half, t_center + half
    best: Optional[Tuple[float, float]] = None
    best_score = float("inf")
    for i in range(len(uniq) - 1):
        t0, t1 = uniq[i], uniq[i + 1]
        if t0 > t1:
            t0, t1 = t1, t0
        mid = line_point + 0.5 * (t0 + t1) * line_dir
        if not _point_in_loop_polygon_3d(vertices, loop, mid):
            continue
        if not (t0 - 1e-9 <= t_center <= t1 + 1e-9):
            continue
        score = abs(t1 - t0)
        if score < best_score:
            best_score = score
            best = (t0, t1)
    if best is not None:
        return best
    # 若无包含孔心的区间，取最近的一对
    for i in range(len(uniq) - 1):
        t0, t1 = uniq[i], uniq[i + 1]
        if t0 > t1:
            t0, t1 = t1, t0
        mid = line_point + 0.5 * (t0 + t1) * line_dir
        if not _point_in_loop_polygon_3d(vertices, loop, mid):
            continue
        score = abs(0.5 * (t0 + t1) - t_center)
        if score < best_score:
            best_score = score
            best = (t0, t1)
    return best


def _circle_plane_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return _orthonormal_basis(normal)


def _circle_point_at_angle(
    center: np.ndarray,
    radius: float,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    angle: float,
) -> np.ndarray:
    return center + radius * (np.cos(angle) * u_axis + np.sin(angle) * v_axis)


def _collect_circle_clip_angles(
    center: np.ndarray,
    normal: np.ndarray,
    radius: float,
    vertices: np.ndarray,
    loop: Sequence[int],
    tol: float,
) -> List[float]:
    u_axis, v_axis = _circle_plane_basis(normal)
    angles: List[float] = []
    n = len(loop)
    for i in range(n):
        va = int(loop[i])
        vb = int(loop[(i + 1) % n])
        for p in (vertices[va], vertices[vb]):
            rel = p - center
            dist_plane = abs(float(np.dot(rel, normal)))
            radial = rel - float(np.dot(rel, normal)) * normal
            dist_rad = float(np.linalg.norm(radial))
            if dist_plane <= tol and abs(dist_rad - radius) <= max(tol, 0.02 * radius):
                angles.append(
                    _angle_on_circle(p, center, u_axis, v_axis)
                )
        # segment-plane intersection then check radius
        a, b = vertices[va], vertices[vb]
        da = float(np.dot(a - center, normal))
        db = float(np.dot(b - center, normal))
        if da * db < 0:
            t = da / (da - db)
            p = a + t * (b - a)
            rel = p - center
            radial = rel - float(np.dot(rel, normal)) * normal
            if abs(float(np.linalg.norm(radial)) - radius) <= max(tol, 0.02 * radius):
                angles.append(_angle_on_circle(p, center, u_axis, v_axis))
    return angles


def _select_circle_arc_interval(
    angles: Sequence[float],
    center: np.ndarray,
    normal: np.ndarray,
    radius: float,
    hole_center: np.ndarray,
    vertices: np.ndarray,
    loop: Sequence[int],
) -> Optional[Tuple[float, float]]:
    u_axis, v_axis = _circle_plane_basis(normal)
    center_angle = _angle_on_circle(hole_center, center, u_axis, v_axis)
    if not angles:
        return 0.0, 2.0 * np.pi
    uniq = sorted(set(_normalize_angle(float(a)) for a in angles))
    if len(uniq) < 2:
        a0 = uniq[0]
        return a0, a0 + np.pi
    best: Optional[Tuple[float, float]] = None
    best_score = float("inf")
    for i in range(len(uniq)):
        a0 = uniq[i]
        a1 = uniq[(i + 1) % len(uniq)]
        if i + 1 == len(uniq):
            a1 = uniq[0] + 2.0 * np.pi
        mid_angle = 0.5 * (a0 + a1)
        mid = _circle_point_at_angle(center, radius, u_axis, v_axis, mid_angle)
        if not _point_in_loop_polygon_3d(vertices, loop, mid):
            continue
        # center_angle in [a0,a1]?
        ca = _normalize_angle(center_angle)
        if a0 <= ca <= a1 or (a1 > 2 * np.pi and ca <= a1 - 2 * np.pi):
            score = abs(a1 - a0)
            if score < best_score:
                best_score = score
                best = (a0, a1)
    if best is not None:
        return best
    return uniq[0], uniq[1]


def _sample_analytic_between(
    curve: AnalyticCurve,
    t_start: float,
    t_end: float,
    hole_center: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    if curve.kind == "line" and curve.line_point is not None and curve.line_dir is not None:
        lp = curve.line_point
        ld = curve.line_dir
        ts = np.linspace(t_start, t_end, max(2, n_samples), dtype=np.float64)
        return np.array([lp + t * ld for t in ts], dtype=np.float64)
    if (
        curve.kind == "circle"
        and curve.circle_center is not None
        and curve.circle_normal is not None
        and curve.circle_radius is not None
    ):
        u_axis, v_axis = _circle_plane_basis(curve.circle_normal)
        delta = float(t_end - t_start)
        start_pt = _circle_point_at_angle(
            curve.circle_center,
            curve.circle_radius,
            u_axis,
            v_axis,
            t_start,
        )
        end_pt = _circle_point_at_angle(
            curve.circle_center,
            curve.circle_radius,
            u_axis,
            v_axis,
            t_end if abs(delta) <= np.pi else t_start + delta,
        )
        guide = hole_center
        return _sample_circle_arc(
            curve.circle_center,
            curve.circle_normal,
            curve.circle_radius,
            start_pt,
            end_pt,
            guide,
            max(2, n_samples),
            short_arc_only=True,
        )
    line = _general_analytic_as_line(curve, hole_center)
    if line is None:
        return np.zeros((0, 3), dtype=np.float64)
    lp, ld = line
    ts = np.linspace(t_start, t_end, max(2, n_samples), dtype=np.float64)
    pts = np.array([lp + t * ld for t in ts], dtype=np.float64)
    return np.array(
        [
            _project_to_surface_system([curve.fit_a, curve.fit_b], p)
            for p in pts
        ],
        dtype=np.float64,
    )


def sample_unbounded_analytic_curve(
    curve: AnalyticCurve,
    hole_center: np.ndarray,
    *,
    half_extent: Optional[float] = None,
    n_samples: int = 32,
) -> np.ndarray:
    """用于 debug：沿解析交线采样一段（虚线）。"""
    if half_extent is None:
        half_extent = max(_fit_support_diag([curve.fit_a, curve.fit_b]) * 0.75, 1e-3)
    if curve.kind == "line" and curve.line_point is not None and curve.line_dir is not None:
        return _sample_analytic_between(
            curve, -half_extent, half_extent, hole_center, n_samples
        )
    if curve.kind == "circle" and curve.circle_radius is not None:
        return _sample_analytic_between(curve, 0.0, 2.0 * np.pi, hole_center, n_samples)
    line = _general_analytic_as_line(curve, hole_center)
    if line is None:
        return np.zeros((0, 3), dtype=np.float64)
    return _sample_analytic_between(curve, -half_extent, half_extent, hole_center, n_samples)


def clip_analytic_curve_to_hole(
    curve: AnalyticCurve,
    vertices: np.ndarray,
    loop: Sequence[int],
    *,
    hole_center: np.ndarray,
    loop_mean_edge: Optional[float] = None,
    vertex_labels: Optional[Mapping[int, Sequence[int]]] = None,
    arc_corner_hints: Optional[Mapping[Tuple[int, int], Sequence[int]]] = None,
) -> List[BoundedCurveSegment]:
    """解析交线与孔洞边界环裁剪，返回孔内有界段。"""
    verts = np.asarray(vertices, dtype=np.float64)
    hc = np.asarray(hole_center, dtype=np.float64)
    loop_pts = _loop_points(verts, loop)
    diag = max(float(np.linalg.norm(np.ptp(loop_pts, axis=0))), 1e-12)
    tol = max(1e-8 * diag, 0.02 * float(loop_mean_edge or 0.0), 1e-9)

    ac = curve
    line = _analytic_curve_as_line(ac, hc)
    if line is not None:
        ac = AnalyticCurve(
            patch_pair=ac.patch_pair,
            kind="line",
            fit_a=ac.fit_a,
            fit_b=ac.fit_b,
            line_point=line[0],
            line_dir=line[1],
        )

    segments: List[BoundedCurveSegment] = []
    ref_step = loop_mean_edge
    pair_key = tuple(sorted(curve.patch_pair))
    pref = None
    if arc_corner_hints is not None:
        pref = arc_corner_hints.get(pair_key)

    if ac.kind == "line" and ac.line_point is not None and ac.line_dir is not None:
        ts = _collect_line_clip_parameters(ac.line_point, ac.line_dir, verts, loop, tol)
        intervals = _all_line_segment_intervals(
            ts, ac.line_point, ac.line_dir, hc, verts, loop
        )
        if not intervals:
            return []
        for t0, t1 in intervals:
            length = abs(t1 - t0)
            if length < tol:
                continue
            n_samp = _adaptive_sample_count(length, _fit_support_diag([ac.fit_a, ac.fit_b]))
            if ref_step and ref_step > 1e-15:
                n_samp = _sample_count_from_reference_step(length, ref_step)
            n_samp = max(3, n_samp)
            pts = _sample_analytic_between(ac, t0, t1, hc, n_samp)
            if pts.shape[0] < 2:
                continue
            v0, v1 = _assign_segment_boundary_vertices(
                pair_key,
                vertex_labels,
                loop,
                verts,
                pts[0],
                pts[-1],
                preferred_corners=pref,
            )
            if int(v0) == int(v1) and int(v0) >= 0:
                continue
            conf = "high" if len(ts) >= 2 else "medium"
            seg = BoundedCurveSegment(
                analytic=curve,
                t_start=t0,
                t_end=t1,
                curve_points=pts,
                boundary_vertex_indices=(v0, v1),
                clip_confidence=conf,
                start_xyz=pts[0].copy(),
                end_xyz=pts[-1].copy(),
            )
            if _segment_is_usable_partial_clip(seg):
                seg = _rebuild_segment_toward_hole_interior(
                    seg,
                    verts,
                    loop,
                    hc,
                    loop_mean_edge,
                )
            segments.append(seg)
        return segments

    if (
        ac.kind == "circle"
        and ac.circle_center is not None
        and ac.circle_normal is not None
        and ac.circle_radius is not None
    ):
        angles = _collect_circle_clip_angles(
            ac.circle_center,
            ac.circle_normal,
            ac.circle_radius,
            verts,
            loop,
            tol,
        )
        interval = _select_circle_arc_interval(
            angles,
            ac.circle_center,
            ac.circle_normal,
            ac.circle_radius,
            hc,
            verts,
            loop,
        )
        if interval is None:
            return []
        a0, a1 = interval
        u_axis, v_axis = _circle_plane_basis(ac.circle_normal)
        start_pt = _circle_point_at_angle(
            ac.circle_center, ac.circle_radius, u_axis, v_axis, a0
        )
        end_pt = _circle_point_at_angle(
            ac.circle_center, ac.circle_radius, u_axis, v_axis, a1
        )
        length = abs(ac.circle_radius * (a1 - a0))
        n_samp = _adaptive_sample_count(length, _fit_support_diag([ac.fit_a, ac.fit_b]))
        if ref_step and ref_step > 1e-15:
            n_samp = _sample_count_from_reference_step(length, ref_step)
        n_samp = max(3, n_samp)
        pts = _sample_circle_arc(
            ac.circle_center,
            ac.circle_normal,
            ac.circle_radius,
            start_pt,
            end_pt,
            hc,
            max(2, n_samp),
            short_arc_only=True,
        )
        if pts.shape[0] < 2:
            return []
        v0, v1 = _assign_segment_boundary_vertices(
            pair_key,
            vertex_labels,
            loop,
            verts,
            pts[0],
            pts[-1],
            preferred_corners=pref,
        )
        if int(v0) == int(v1) and int(v0) >= 0:
            return []
        conf = "high" if len(angles) >= 2 else "medium"
        seg = BoundedCurveSegment(
            analytic=curve,
            t_start=a0,
            t_end=a1,
            curve_points=pts,
            boundary_vertex_indices=(v0, v1),
            clip_confidence=conf,
            start_xyz=pts[0].copy(),
            end_xyz=pts[-1].copy(),
        )
        if _segment_is_usable_partial_clip(seg):
            seg = _rebuild_segment_toward_hole_interior(
                seg,
                verts,
                loop,
                hc,
                loop_mean_edge,
            )
        segments.append(seg)
        return segments

    return []


def bounded_segment_to_intersection_curve(
    segment: BoundedCurveSegment,
    *,
    intersection_sampling_reference_step: Optional[float] = None,
    vertices: Optional[np.ndarray] = None,
) -> IntersectionCurve:
    """将裁剪段转为 ``IntersectionCurve``（补洞/debug 兼容）。"""
    ac = segment.analytic
    v0, v1 = segment.boundary_vertex_indices
    start = segment.start_xyz if segment.start_xyz is not None else segment.curve_points[0]
    end = segment.end_xyz if segment.end_xyz is not None else segment.curve_points[-1]
    if vertices is not None:
        if int(v0) >= 0:
            start = np.asarray(vertices[int(v0)], dtype=np.float64)
        if int(v1) >= 0:
            end = np.asarray(vertices[int(v1)], dtype=np.float64)
    guide = 0.5 * (start + end)
    ref = intersection_sampling_reference_step
    n_pts = int(len(segment.curve_points))
    if ref and ref > 1e-15:
        length = float(np.sum(np.linalg.norm(np.diff(segment.curve_points, axis=0), axis=1)))
        if length > ref * 1.5 and n_pts < 3:
            ref = ref  # recover_curve will resample
    return recover_curve_between_points(
        ac.fit_a,
        ac.fit_b,
        start,
        end,
        guide,
        endpoint_vertex_indices=(int(v0), int(v1)),
        intersection_sampling_reference_step=ref,
    )


def _segment_is_usable_partial_clip(segment: BoundedCurveSegment) -> bool:
    """裁剪得到的一角点 + 一虚拟端点且置信度足够，可直接作 layout 辐射段。"""
    v0, v1 = (
        int(segment.boundary_vertex_indices[0]),
        int(segment.boundary_vertex_indices[1]),
    )
    if not ((v0 >= 0 and v1 < 0) or (v1 >= 0 and v0 < 0)):
        return False
    return str(segment.clip_confidence) in {"high", "medium"}


def _partial_clip_corner_vertex(segment: BoundedCurveSegment) -> Optional[int]:
    v0, v1 = (
        int(segment.boundary_vertex_indices[0]),
        int(segment.boundary_vertex_indices[1]),
    )
    if v0 >= 0 and v1 < 0:
        return v0
    if v1 >= 0 and v0 < 0:
        return v1
    return None


def _hole_inward_junction_on_analytic(
    analytic: AnalyticCurve,
    from_xyz: np.ndarray,
    hole_center: np.ndarray,
    *,
    vertices: Optional[np.ndarray] = None,
    loop: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """
    从边界角点沿解析交线朝孔心方向的汇交点。

    直线：角点参数 → 孔心投影参数，取孔心侧端点（圆柱双枝时以孔心锚定正确分枝）。
    圆：短弧走向孔心在交圆上的投影。
    一般曲线：两面交线系统上孔心的投影。
    """
    hc = np.asarray(hole_center, dtype=np.float64).reshape(3)
    from_p = np.asarray(from_xyz, dtype=np.float64).reshape(3)
    fa = analytic.fit_a
    fb = analytic.fit_b

    line = _analytic_curve_as_line(analytic, hc)
    if line is not None:
        lp = np.asarray(line[0], dtype=np.float64)
        ld = np.asarray(line[1], dtype=np.float64)
        from_on = lp + float(np.dot(from_p - lp, ld)) * ld
        vec_h = hc - from_on
        march_dir = ld if float(np.dot(ld, vec_h)) >= 0.0 else -ld
        t_star = max(0.0, float(np.dot(hc - from_on, march_dir)))
        junction = from_on + t_star * march_dir
        if vertices is not None and loop is not None:
            verts = np.asarray(vertices, dtype=np.float64)
            loop_pts = _loop_points(verts, loop)
            diag = max(float(np.linalg.norm(np.ptp(loop_pts, axis=0))), 1e-12)
            tol = max(1e-8 * diag, 1e-9)
            ts = _collect_line_clip_parameters(lp, ld, verts, loop, tol)
            t_from = float(np.dot(from_on - lp, ld))
            sign = 1.0 if float(np.dot(march_dir, ld)) >= 0.0 else -1.0
            inward = [
                float(t)
                for t in ts
                if sign * (float(t) - t_from) > 1e-9
                and sign * (float(t) - float(np.dot(junction - lp, ld))) <= 1e-9
            ]
            if inward and _point_in_loop_polygon_3d(verts, loop, hc):
                t_boundary = max(inward) if sign > 0 else min(inward)
                boundary_pt = lp + t_boundary * ld
                if float(np.linalg.norm(boundary_pt - hc)) < float(
                    np.linalg.norm(junction - hc)
                ):
                    junction = 0.92 * junction + 0.08 * boundary_pt
        return junction

    if (
        analytic.kind == "circle"
        and analytic.circle_center is not None
        and analytic.circle_normal is not None
        and analytic.circle_radius is not None
    ):
        center = np.asarray(analytic.circle_center, dtype=np.float64)
        normal = np.asarray(analytic.circle_normal, dtype=np.float64)
        radius = float(analytic.circle_radius)
        from_on = project_point_to_surface_pair(fa, fb, from_p)
        hc_on = project_point_to_surface_pair(fa, fb, hc)
        u_axis, v_axis = _circle_plane_basis(normal)
        a_from = _angle_on_circle(from_on, center, u_axis, v_axis)
        delta = _circle_arc_sweep_delta(
            center,
            normal,
            radius,
            from_on,
            hc_on,
            hc,
            16,
            short_arc_only=True,
        )
        return _circle_point_at_angle(center, radius, u_axis, v_axis, a_from + delta)

    return _project_to_surface_system([fa, fb], hc, iterations=12)


def _rebuild_segment_toward_hole_interior(
    segment: BoundedCurveSegment,
    vertices: np.ndarray,
    loop: Sequence[int],
    hole_center: np.ndarray,
    loop_mean_edge: Optional[float],
) -> BoundedCurveSegment:
    """将一角点 + 虚拟端点裁剪段重建为角点 → 孔内汇交点辐射段。"""
    corner = _partial_clip_corner_vertex(segment)
    if corner is None:
        return segment
    corner_xyz = np.asarray(vertices[int(corner)], dtype=np.float64)
    junction = _hole_inward_junction_on_analytic(
        segment.analytic,
        corner_xyz,
        hole_center,
        vertices=vertices,
        loop=loop,
    )
    rebuilt = _segment_corner_to_junction(
        segment.analytic,
        int(corner),
        junction,
        vertices,
        hole_center,
        loop_mean_edge=loop_mean_edge,
    )
    return rebuilt if rebuilt is not None else segment


def _junction_near_hole_on_analytic(
    analytic: AnalyticCurve,
    hole_center: np.ndarray,
) -> np.ndarray:
    """沿解析交线取最靠近孔心的点，避免圆柱双分支选到孔外远枝。"""
    hc = np.asarray(hole_center, dtype=np.float64).reshape(3)
    line = _general_analytic_as_line(analytic, hc)
    if line is not None:
        lp = np.asarray(line[0], dtype=np.float64)
        ld = np.asarray(line[1], dtype=np.float64)
        ln = float(np.linalg.norm(ld))
        if ln > 1e-15:
            ld = ld / ln
            return lp + float(np.dot(hc - lp, ld)) * ld
    if (
        analytic.kind == "line"
        and analytic.line_point is not None
        and analytic.line_dir is not None
    ):
        lp = np.asarray(analytic.line_point, dtype=np.float64)
        ld = np.asarray(analytic.line_dir, dtype=np.float64)
        ln = float(np.linalg.norm(ld))
        if ln > 1e-15:
            ld = ld / ln
            return lp + float(np.dot(hc - lp, ld)) * ld
    return hc.copy()


def _pair_needs_junction_completion(
    segments: Sequence[BoundedCurveSegment],
) -> bool:
    if not segments:
        return True
    if any(_segment_is_usable_partial_clip(seg) for seg in segments):
        return False
    if any(
        int(seg.boundary_vertex_indices[0]) >= 0
        and int(seg.boundary_vertex_indices[1]) >= 0
        for seg in segments
    ):
        return False
    return True


def _segment_corner_to_junction(
    analytic: AnalyticCurve,
    corner_vertex: int,
    junction_point: np.ndarray,
    vertices: np.ndarray,
    hole_center: np.ndarray,
    *,
    loop_mean_edge: Optional[float] = None,
) -> Optional[BoundedCurveSegment]:
    """单 transition corner → 汇交点 辐射段（四面/多面汇交）。"""
    start = np.asarray(vertices[int(corner_vertex)], dtype=np.float64)
    end = np.asarray(junction_point, dtype=np.float64)
    guide = np.asarray(hole_center, dtype=np.float64)
    ic = recover_curve_between_points(
        analytic.fit_a,
        analytic.fit_b,
        start,
        end,
        guide,
        endpoint_vertex_indices=(int(corner_vertex), -1),
        intersection_sampling_reference_step=loop_mean_edge,
    )
    pts = np.asarray(ic.curve_points, dtype=np.float64)
    if pts.shape[0] < 2:
        return None
    return BoundedCurveSegment(
        analytic=analytic,
        t_start=0.0,
        t_end=1.0,
        curve_points=pts,
        boundary_vertex_indices=(int(corner_vertex), -1),
        clip_confidence="corner_junction",
        start_xyz=start.copy(),
        end_xyz=end.copy(),
    )


def _segments_form_spurious_boundary_fan(
    pairs: Sequence[Tuple[int, int]],
    segments_by_pair: Mapping[Tuple[int, int], Sequence[BoundedCurveSegment]],
) -> bool:
    """
    多 pair 裁剪段若共用同一孔边界顶点向外辐射，说明三面汇交被误收成边界扇形。
    """
    pair_list = [tuple(sorted((int(a), int(b)))) for a, b in pairs]
    if len(pair_list) < 3:
        return False
    endpoint_sets: List[Set[int]] = []
    for pair in pair_list:
        segs = list(segments_by_pair.get(pair, []))
        mesh_mesh = [
            seg
            for seg in segs
            if int(seg.boundary_vertex_indices[0]) >= 0
            and int(seg.boundary_vertex_indices[1]) >= 0
        ]
        if not mesh_mesh:
            return False
        e0, e1 = (
            int(mesh_mesh[0].boundary_vertex_indices[0]),
            int(mesh_mesh[0].boundary_vertex_indices[1]),
        )
        endpoint_sets.append({e0, e1})
    shared = set.intersection(*endpoint_sets)
    if len(shared) != 1:
        return False
    hub = int(next(iter(shared)))
    others = sorted(
        {
            int(next(v for v in es if int(v) != hub))
            for es in endpoint_sets
        }
    )
    return len(others) == len(endpoint_sets) and len(others) >= 3


def _corner_for_junction_segment(
    pair: Tuple[int, int],
    vertex_labels: Mapping[int, Sequence[int]],
    loop: Sequence[int],
    *,
    arc_corner_hints: Optional[Mapping[Tuple[int, int], Sequence[int]]] = None,
) -> Optional[int]:
    loop_set = {int(v) for v in loop}
    key = tuple(sorted((int(pair[0]), int(pair[1]))))
    if arc_corner_hints is not None:
        pref = arc_corner_hints.get(key)
        if pref:
            for c in pref:
                if int(c) in loop_set:
                    return int(c)
    corners = [
        int(c)
        for c in _transition_corners_for_pair(pair, vertex_labels)
        if int(c) in loop_set
    ]
    return int(corners[0]) if corners else None


def _complete_segments_with_junction(
    patch_surface_fits: Mapping[int, SurfaceFit],
    pairs: Sequence[Tuple[int, int]],
    segments_by_pair: Mapping[Tuple[int, int], List[BoundedCurveSegment]],
    vertex_labels: Mapping[int, Sequence[int]],
    vertices: np.ndarray,
    loop: Sequence[int],
    hole_center: np.ndarray,
    loop_mean_edge: Optional[float],
    *,
    arc_corner_hints: Optional[Mapping[Tuple[int, int], Sequence[int]]] = None,
) -> Tuple[List[BoundedCurveSegment], Optional[np.ndarray], str]:
    """为缺段或单角点 pair 补 corner→junction 辐射线。"""
    pair_list = [tuple(sorted((int(a), int(b)))) for a, b in pairs]
    hc = np.asarray(hole_center, dtype=np.float64).reshape(3)
    junction_point: Optional[np.ndarray] = None
    junction_confidence = "none"
    force_junction_rays = _segments_form_spurious_boundary_fan(
        pair_list,
        segments_by_pair,
    )
    out: List[BoundedCurveSegment] = []
    per_pair_junctions: List[np.ndarray] = []
    for pair in pair_list:
        existing = list(segments_by_pair.get(pair, []))
        good = [
            seg
            for seg in existing
            if int(seg.boundary_vertex_indices[0]) >= 0
            and int(seg.boundary_vertex_indices[1]) >= 0
        ]
        partial = [
            seg for seg in existing if _segment_is_usable_partial_clip(seg)
        ]
        if good and not force_junction_rays:
            out.extend(good)
            continue
        if partial and not force_junction_rays:
            for seg in partial:
                rebuilt = _rebuild_segment_toward_hole_interior(
                    seg,
                    vertices,
                    loop,
                    hc,
                    loop_mean_edge,
                )
                out.append(rebuilt)
                corner = _partial_clip_corner_vertex(rebuilt)
                if corner is not None:
                    virt_xyz = np.asarray(
                        rebuilt.end_xyz
                        if int(rebuilt.boundary_vertex_indices[0]) == int(corner)
                        else rebuilt.start_xyz,
                        dtype=np.float64,
                    )
                    per_pair_junctions.append(virt_xyz.reshape(3))
            continue
        if pair[0] not in patch_surface_fits or pair[1] not in patch_surface_fits:
            continue
        fa = patch_surface_fits[pair[0]]
        fb = patch_surface_fits[pair[1]]
        analytic = analytic_intersection(fa, fb)
        if analytic is None:
            continue
        corner = _corner_for_junction_segment(
            pair,
            vertex_labels,
            loop,
            arc_corner_hints=arc_corner_hints,
        )
        if corner is None:
            continue
        corner_xyz = np.asarray(vertices[int(corner)], dtype=np.float64)
        pair_junction = _hole_inward_junction_on_analytic(
            analytic,
            corner_xyz,
            hc,
            vertices=vertices,
            loop=loop,
        )
        per_pair_junctions.append(np.asarray(pair_junction, dtype=np.float64).reshape(3))
        seg = _segment_corner_to_junction(
            analytic,
            corner,
            pair_junction,
            vertices,
            hc,
            loop_mean_edge=loop_mean_edge,
        )
        if seg is not None:
            out.append(seg)
    if per_pair_junctions:
        junction_point = np.mean(np.vstack(per_pair_junctions), axis=0)
        junction_confidence = "medium"
    elif len(pair_list) >= 3:
        junction_point, junction_confidence = estimate_triple_junction(
            patch_surface_fits,
            pair_list,
            hc,
        )
    if not out:
        return [], None, "none"
    if junction_point is None:
        junction_point = hc.copy()
        junction_confidence = "low"
    return out, np.asarray(junction_point, dtype=np.float64), str(junction_confidence)


def _transition_corners_for_pair(
    pair: Tuple[int, int],
    vertex_labels: Mapping[int, Sequence[int]],
) -> List[int]:
    a, b = int(pair[0]), int(pair[1])
    out: List[int] = []
    for vertex, labels in vertex_labels.items():
        ls = {int(x) for x in labels}
        if a in ls and b in ls:
            out.append(int(vertex))
    return out


def _best_transition_corner_pair(
    candidates: Sequence[int],
    vertices: np.ndarray,
    start_xyz: np.ndarray,
    end_xyz: np.ndarray,
) -> Optional[Tuple[int, int]]:
    """在 transition corner 候选中选与几何端点最匹配的一对。"""
    cand = sorted({int(c) for c in candidates})
    if len(cand) < 2:
        return None
    if len(cand) == 2:
        v0, v1 = cand[0], cand[1]
        cost_fwd = float(
            np.linalg.norm(vertices[v0] - start_xyz)
            + np.linalg.norm(vertices[v1] - end_xyz)
        )
        cost_rev = float(
            np.linalg.norm(vertices[v1] - start_xyz)
            + np.linalg.norm(vertices[v0] - end_xyz)
        )
        return (v0, v1) if cost_fwd <= cost_rev else (v1, v0)
    best: Optional[Tuple[int, int]] = None
    best_cost = float("inf")
    for i, v0 in enumerate(cand):
        for v1 in cand[i + 1 :]:
            for a, b in ((v0, v1), (v1, v0)):
                cost = float(
                    np.linalg.norm(vertices[a] - start_xyz)
                    + np.linalg.norm(vertices[b] - end_xyz)
                )
                if cost < best_cost:
                    best_cost = cost
                    best = (int(a), int(b))
    return best


def _assign_segment_boundary_vertices(
    pair: Tuple[int, int],
    vertex_labels: Optional[Mapping[int, Sequence[int]]],
    loop: Sequence[int],
    vertices: np.ndarray,
    start_xyz: np.ndarray,
    end_xyz: np.ndarray,
    *,
    preferred_corners: Optional[Sequence[int]] = None,
) -> Tuple[int, int]:
    """将裁剪段几何端点绑定到该 pair 的 transition corner（禁止全局盲搜）。"""
    if vertex_labels is None:
        return (
            _nearest_loop_vertex(vertices, loop, start_xyz),
            _nearest_loop_vertex(vertices, loop, end_xyz),
        )
    loop_set = {int(v) for v in loop}
    pref = [int(c) for c in (preferred_corners or []) if int(c) in loop_set]
    corners = pref if len(pref) >= 2 else [
        int(c)
        for c in _transition_corners_for_pair(pair, vertex_labels)
        if int(c) in loop_set
    ]
    if len(pref) == 1:
        corners = pref + [
            int(c)
            for c in _transition_corners_for_pair(pair, vertex_labels)
            if int(c) in loop_set and int(c) not in pref
        ]
    if len(corners) >= 2:
        matched = _best_transition_corner_pair(corners, vertices, start_xyz, end_xyz)
        if matched is not None:
            return matched
    if len(corners) == 1:
        c = int(corners[0])
        d0 = float(np.linalg.norm(vertices[c] - start_xyz))
        d1 = float(np.linalg.norm(vertices[c] - end_xyz))
        if d0 <= d1:
            return c, -1
        return -1, c
    return (
        _nearest_loop_vertex(vertices, loop, start_xyz),
        _nearest_loop_vertex(vertices, loop, end_xyz),
    )


def _segment_clip_score(segment: BoundedCurveSegment) -> Tuple[int, int, int, float]:
    v0, v1 = segment.boundary_vertex_indices
    both = int(v0) >= 0 and int(v1) >= 0
    conf_rank = {
        "high": 4,
        "medium": 3,
        "corner_endpoints": 2,
        "corner_junction": 1,
        "low": 0,
        "none": 0,
    }.get(str(segment.clip_confidence), 0)
    n_pts = int(len(segment.curve_points))
    length = 0.0
    if n_pts >= 2:
        length = float(
            np.sum(np.linalg.norm(np.diff(segment.curve_points, axis=0), axis=1))
        )
    return (1 if both else 0, conf_rank, n_pts, length)


def _select_best_segments_per_pair(
    segments_by_pair: Mapping[Tuple[int, int], List[BoundedCurveSegment]],
) -> List[BoundedCurveSegment]:
    out: List[BoundedCurveSegment] = []
    for pair in sorted(segments_by_pair):
        segs = list(segments_by_pair[pair])
        if not segs:
            continue
        if len(segs) == 1:
            out.append(segs[0])
            continue
        out.append(max(segs, key=_segment_clip_score))
    return out


def _ordered_loop_corner_pair(
    corners: Sequence[int],
    loop: Sequence[int],
) -> Optional[Tuple[int, int]]:
    loop_idx = {int(v): i for i, v in enumerate(loop)}
    ordered = [int(v) for v in corners if int(v) in loop_idx]
    if len(ordered) < 2:
        return None
    ordered.sort(key=lambda v: loop_idx[int(v)])
    return int(ordered[0]), int(ordered[-1])


def _segment_from_transition_corners(
    analytic: AnalyticCurve,
    vertices: np.ndarray,
    loop: Sequence[int],
    vertex_labels: Mapping[int, Sequence[int]],
    hole_center: np.ndarray,
    *,
    loop_mean_edge: Optional[float] = None,
    preferred_corners: Optional[Sequence[int]] = None,
) -> Optional[BoundedCurveSegment]:
    """
    clip 失败时：用语义过渡角点作端点，形状仍由 ``recover_curve_between_points`` 解析采样。
    """
    loop_set = {int(v) for v in loop}
    pref = [int(c) for c in (preferred_corners or []) if int(c) in loop_set]
    corners = pref if len(pref) >= 2 else [
        int(c)
        for c in _transition_corners_for_pair(analytic.patch_pair, vertex_labels)
        if int(c) in loop_set
    ]
    if len(corners) < 2:
        return None
    fa = analytic.fit_a
    fb = analytic.fit_b
    v0, v1 = corners[0], corners[1]
    if len(corners) > 2:
        # 先用解析线估计几何端点，再选最优角点对
        probe = analytic_intersection(fa, fb)
        if probe is not None and probe.kind == "line" and probe.line_point is not None and probe.line_dir is not None:
            p0 = np.asarray(probe.line_point, dtype=np.float64)
            d = np.asarray(probe.line_dir, dtype=np.float64)
            start_xyz = p0 - 0.05 * d
            end_xyz = p0 + 0.05 * d
        else:
            start_xyz = np.asarray(vertices[v0], dtype=np.float64)
            end_xyz = np.asarray(vertices[corners[-1]], dtype=np.float64)
        matched = _best_transition_corner_pair(corners, vertices, start_xyz, end_xyz)
        if matched is None:
            return None
        v0, v1 = matched
    start = np.asarray(vertices[int(v0)], dtype=np.float64)
    end = np.asarray(vertices[int(v1)], dtype=np.float64)
    guide = np.asarray(hole_center, dtype=np.float64)
    ic = recover_curve_between_points(
        analytic.fit_a,
        analytic.fit_b,
        start,
        end,
        guide,
        endpoint_vertex_indices=(int(v0), int(v1)),
        intersection_sampling_reference_step=loop_mean_edge,
    )
    pts = np.asarray(ic.curve_points, dtype=np.float64)
    if pts.shape[0] < 2:
        return None
    return BoundedCurveSegment(
        analytic=analytic,
        t_start=0.0,
        t_end=1.0,
        curve_points=pts,
        boundary_vertex_indices=(int(v0), int(v1)),
        clip_confidence="corner_endpoints",
        start_xyz=pts[0].copy(),
        end_xyz=pts[-1].copy(),
    )


def recover_bounded_intersection_curves(
    patch_surface_fits: Mapping[int, SurfaceFit],
    patch_pairs: Iterable[Tuple[int, int]],
    vertices: np.ndarray,
    loop: Sequence[int],
    *,
    hole_center: np.ndarray,
    loop_mean_edge: Optional[float] = None,
    vertex_labels: Optional[Mapping[int, Sequence[int]]] = None,
    arc_corner_hints: Optional[Mapping[Tuple[int, int], Sequence[int]]] = None,
) -> Tuple[List[IntersectionCurve], List[BoundedCurveSegment], List[AnalyticCurve]]:
    """
    对每对相邻解析面：孔腔约束 arrangement → IntersectionCurve。

    几何语义统一委托 ``hole_cavity_arrangement.recover_cavity_restricted_curves``：
    Γ_ij = restrict(S_i ∩ S_j, C)，汇交由 arrangement 节点聚类确定。
    """
    from .hole_cavity_arrangement import recover_cavity_restricted_curves

    result = recover_cavity_restricted_curves(
        patch_surface_fits,
        patch_pairs,
        vertices,
        loop,
        hole_center=hole_center,
        loop_mean_edge=loop_mean_edge,
        vertex_labels=vertex_labels,
        arc_corner_hints=arc_corner_hints,
    )
    return result.curves, result.bounded_segments, result.analytic_curves



def _relevant_fits_for_junction(
    patch_fits: Mapping[int, SurfaceFit],
    pair_keys: Sequence[Tuple[int, int]],
) -> List[SurfaceFit]:
    relevant_labels = sorted({label for pair in pair_keys for label in pair})
    relevant_fits = [
        patch_fits[label]
        for label in relevant_labels
        if label in patch_fits
        and is_analytic_surface_type(patch_fits[label].surface_type)
    ]
    if not relevant_fits:
        relevant_fits = [
            fit
            for fit in patch_fits.values()
            if is_analytic_surface_type(fit.surface_type)
        ]
    if not relevant_fits:
        relevant_fits = list(patch_fits.values())
    return relevant_fits


def _junction_residual_score(
    point: np.ndarray,
    relevant_fits: Sequence[SurfaceFit],
) -> float:
    diag = _fit_support_diag(relevant_fits)
    residuals = [
        abs(_surface_residual_and_gradient(fit, np.asarray(point, dtype=np.float64))[0])
        for fit in relevant_fits
    ]
    if not residuals:
        return float("inf")
    return float(np.sqrt(np.mean(np.square(residuals)))) / diag


def _junction_confidence_from_score(score: float, n_fits: int) -> str:
    if score < 8e-3 and int(n_fits) >= 3:
        return "high"
    if score < 3e-2 and int(n_fits) >= 2:
        return "medium"
    return "low"


def _pick_best_junction_candidate(
    candidates: Sequence[np.ndarray],
    relevant_fits: Sequence[SurfaceFit],
    *,
    guide: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """投影精修后，在残差接近最优的候选里取距 guide 最近者。"""
    guide_pt = np.asarray(guide, dtype=np.float64).reshape(3)
    if not candidates:
        point = _project_to_surface_system(relevant_fits, guide_pt, iterations=12)
        return point, _junction_residual_score(point, relevant_fits)

    evaluated: List[Tuple[np.ndarray, float, float]] = []
    for seed in candidates:
        point = _project_to_surface_system(
            relevant_fits,
            np.asarray(seed, dtype=np.float64).reshape(3),
            iterations=12,
        )
        evaluated.append(
            (
                point,
                _junction_residual_score(point, relevant_fits),
                float(np.linalg.norm(point - guide_pt)),
            )
        )

    best_score = min(score for _pt, score, _dist in evaluated)
    score_cutoff = max(8e-3, float(best_score) * 4.0, float(best_score) + 1e-9)
    pool = [item for item in evaluated if item[1] <= score_cutoff] or list(evaluated)
    point, score, _dist = min(pool, key=lambda item: (item[2], item[1]))
    return point, score


def triple_junction_from_pairwise_curves(
    patch_fits: Mapping[int, SurfaceFit],
    pair_keys: Iterable[Tuple[int, int]],
    guide_point: np.ndarray,
) -> Optional[Tuple[np.ndarray, str]]:
    """
    三面汇交：两两解析求交线，再求交线-交线交点作为 J。

    与有界裁剪 / virtual bridge 共用 ``analytic_intersection`` + ``intersect_analytic_curves``。
    至少 3 个相关 patch 且能构造共享单 patch 的交线对时返回结果，否则 ``None``。
    """
    guide = np.asarray(guide_point, dtype=np.float64).reshape(3)
    pair_keys_list = [tuple(sorted((int(a), int(b)))) for a, b in pair_keys]
    relevant_labels = sorted({label for pair in pair_keys_list for label in pair})
    if len(relevant_labels) < 3:
        return None

    relevant_fits = _relevant_fits_for_junction(patch_fits, pair_keys_list)
    if len(relevant_fits) < 3:
        return None

    curves_by_pair: Dict[Tuple[int, int], AnalyticCurve] = {}
    for a_label, b_label in pair_keys_list:
        if a_label not in patch_fits or b_label not in patch_fits:
            continue
        ac = analytic_intersection(patch_fits[a_label], patch_fits[b_label])
        if ac is not None:
            curves_by_pair[(int(a_label), int(b_label))] = ac

    if len(curves_by_pair) < 2:
        return None

    raw_candidates: List[np.ndarray] = []
    plane_fits = [
        patch_fits[label]
        for label in relevant_labels
        if label in patch_fits and patch_fits[label].surface_type == "plane"
    ]
    if len(plane_fits) >= 3:
        a_rows: List[np.ndarray] = []
        b_vals: List[float] = []
        for fit in plane_fits:
            normal = np.asarray(fit.surface_params["normal"], dtype=np.float64)
            point = np.asarray(fit.surface_params["point"], dtype=np.float64)
            a_rows.append(normal)
            b_vals.append(float(np.dot(normal, point)))
        try:
            junction, _, _, _ = np.linalg.lstsq(np.array(a_rows), np.array(b_vals), rcond=None)
            raw_candidates.append(np.asarray(junction, dtype=np.float64).reshape(3))
        except np.linalg.LinAlgError:
            pass

    pair_list = sorted(curves_by_pair.keys())
    for i, pair_a in enumerate(pair_list):
        for pair_b in pair_list[i + 1 :]:
            if len(set(pair_a) & set(pair_b)) != 1:
                continue
            junction_pt = intersect_analytic_curves(
                curves_by_pair[pair_a],
                curves_by_pair[pair_b],
                guide_point=guide,
            )
            if junction_pt is not None:
                raw_candidates.append(np.asarray(junction_pt, dtype=np.float64).reshape(3))

    if not raw_candidates:
        return None

    diag = _fit_support_diag(relevant_fits)
    merge_tol = max(1e-9, 1e-6 * diag)
    merged: List[np.ndarray] = []
    for candidate in raw_candidates:
        if any(
            float(np.linalg.norm(candidate - kept)) <= merge_tol
            for kept in merged
        ):
            continue
        merged.append(candidate)

    best_point, best_score = _pick_best_junction_candidate(merged, relevant_fits, guide=guide)
    if not np.isfinite(best_score):
        return None
    return best_point, _junction_confidence_from_score(best_score, len(relevant_fits))


def _estimate_triple_junction_numeric_fallback(
    patch_fits: Mapping[int, SurfaceFit],
    pair_keys: Sequence[Tuple[int, int]],
    guide_point: np.ndarray,
    relevant_fits: Sequence[SurfaceFit],
) -> Tuple[np.ndarray, str]:
    """数值最小二乘回退：孔心 + 双曲面投影种子 + 多曲面迭代。"""
    guide = np.asarray(guide_point, dtype=np.float64).reshape(3)
    candidates: List[np.ndarray] = [guide.copy()]
    plane_fits = [fit for fit in relevant_fits if fit.surface_type == "plane"]
    if len(plane_fits) >= 3:
        a_rows: List[np.ndarray] = []
        b_vals: List[float] = []
        for fit in plane_fits:
            normal = np.asarray(fit.surface_params["normal"], dtype=np.float64)
            point = np.asarray(fit.surface_params["point"], dtype=np.float64)
            a_rows.append(normal)
            b_vals.append(float(np.dot(normal, point)))
        try:
            junction, _, _, _ = np.linalg.lstsq(np.array(a_rows), np.array(b_vals), rcond=None)
            candidates.append(np.asarray(junction, dtype=np.float64).reshape(3))
        except np.linalg.LinAlgError:
            pass
    for a_label, b_label in pair_keys:
        if a_label not in patch_fits or b_label not in patch_fits:
            continue
        fit_a = patch_fits[a_label]
        fit_b = patch_fits[b_label]
        candidates.append(
            _project_to_surface_system([fit_a, fit_b], guide, iterations=10)
        )

    best_point, best_score = _pick_best_junction_candidate(candidates, relevant_fits, guide=guide)
    return best_point, _junction_confidence_from_score(best_score, len(relevant_fits))


def estimate_triple_junction(
    patch_fits: Mapping[int, SurfaceFit],
    pair_keys: Iterable[Tuple[int, int]],
    guide_point: np.ndarray,
) -> Tuple[np.ndarray, str]:
    guide = np.asarray(guide_point, dtype=np.float64).reshape(3)
    pair_keys_list = [tuple(sorted((int(a), int(b)))) for a, b in pair_keys]
    relevant_fits = _relevant_fits_for_junction(patch_fits, pair_keys_list)

    pairwise = triple_junction_from_pairwise_curves(
        patch_fits,
        pair_keys_list,
        guide,
    )
    if pairwise is not None:
        return pairwise

    return _estimate_triple_junction_numeric_fallback(
        patch_fits,
        pair_keys_list,
        guide,
        relevant_fits,
    )
