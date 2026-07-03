#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
局部 patch 解析面拟合。

当前优先支持：
- plane
- cylinder
- sphere
- freeform_fallback
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:  # scipy is optional; the fitter falls back to the existing robust search.
    from scipy.optimize import least_squares
except Exception:  # pragma: no cover - environment dependent
    least_squares = None


ANALYTIC_SURFACE_TYPES = {"plane", "cylinder", "sphere", "cone"}
TRANSITION_SURFACE_TYPE = "transition_surface"


def is_analytic_surface_type(surface_type: str) -> bool:
    return str(surface_type) in ANALYTIC_SURFACE_TYPES


def is_transition_surface_type(surface_type: str) -> bool:
    return str(surface_type) in {TRANSITION_SURFACE_TYPE, "freeform_fallback"}


@dataclass
class SurfaceFit:
    """局部 patch 的解析面拟合结果。"""

    patch_label: int
    surface_type: str
    surface_params: Dict[str, object]
    fit_residual: float
    fit_score: float
    fit_confidence: str
    support_face_indices: List[int]
    support_vertex_indices: List[int]
    support_points: np.ndarray
    fit_diagnostics: Dict[str, object] = field(default_factory=dict)


def _unique_patch_vertices(
    faces: np.ndarray, face_indices: Sequence[int]
) -> List[int]:
    verts = sorted({int(v) for fi in face_indices for v in faces[int(fi)]})
    return verts


def _patch_support_points(
    vertices: np.ndarray, faces: np.ndarray, face_indices: Sequence[int]
) -> Tuple[List[int], np.ndarray]:
    vertex_indices = _unique_patch_vertices(faces, face_indices)
    pts = vertices[np.array(vertex_indices, dtype=np.int64)]
    return vertex_indices, pts


def _patch_support_face_normals(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_indices: Sequence[int],
) -> np.ndarray:
    normals: List[np.ndarray] = []
    for fi in face_indices:
        tri = faces[int(fi)]
        p0, p1, p2 = (
            np.asarray(vertices[int(tri[0])], dtype=np.float64),
            np.asarray(vertices[int(tri[1])], dtype=np.float64),
            np.asarray(vertices[int(tri[2])], dtype=np.float64),
        )
        n = np.cross(p1 - p0, p2 - p0)
        ln = float(np.linalg.norm(n))
        if ln < 1e-12:
            continue
        normals.append(n / ln)
    if not normals:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(normals, dtype=np.float64)


def _bbox_diag(points: np.ndarray) -> float:
    if len(points) == 0:
        return 1.0
    diag = float(np.linalg.norm(np.ptp(points, axis=0)))
    return max(diag, 1e-12)


def _normal_spread(face_normals: np.ndarray) -> float:
    normals = np.asarray(face_normals, dtype=np.float64)
    if normals.ndim != 2 or normals.shape[0] == 0:
        return 0.0
    mean = np.mean(normals, axis=0)
    ln = float(np.linalg.norm(mean))
    if ln < 1e-12:
        return 1.0
    mean = mean / ln
    dots = np.clip(np.abs(normals @ mean.reshape(3, 1)).reshape(-1), 0.0, 1.0)
    angles = np.arccos(dots)
    spread = float(np.quantile(angles, 0.75) / np.deg2rad(45.0))
    return float(np.clip(spread, 0.0, 1.0))


def _safe_normalize(vec: np.ndarray) -> np.ndarray:
    nrm = float(np.linalg.norm(vec))
    if nrm < 1e-15:
        return np.zeros_like(vec, dtype=np.float64)
    return np.asarray(vec, dtype=np.float64) / nrm


def _canonical_direction(vec: np.ndarray) -> np.ndarray:
    out = _safe_normalize(vec)
    if out.shape[0] == 3:
        if abs(float(out[2])) > 1e-12:
            return out if float(out[2]) >= 0.0 else -out
        if abs(float(out[1])) > 1e-12:
            return out if float(out[1]) >= 0.0 else -out
        if abs(float(out[0])) > 1e-12:
            return out if float(out[0]) >= 0.0 else -out
    return out


def _fit_plane_ls(points: np.ndarray) -> Tuple[Dict[str, object], float]:
    centroid = np.mean(points, axis=0)
    centered = points - centroid.reshape(1, 3)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal = normal / (float(np.linalg.norm(normal)) + 1e-15)
    dist = np.abs(centered @ normal.reshape(3, 1)).reshape(-1)
    residual = float(np.sqrt(np.mean(dist * dist)))
    return {"point": centroid, "normal": normal}, residual


def _fit_plane_from_indices(
    points: np.ndarray, ids: Sequence[int]
) -> Tuple[Optional[Dict[str, object]], float]:
    ids = [int(i) for i in ids]
    if len(ids) < 3:
        return None, float("inf")
    sample = points[np.array(ids, dtype=np.int64)]
    p0, p1, p2 = sample[:3]
    normal = np.cross(p1 - p0, p2 - p0)
    if float(np.linalg.norm(normal)) < 1e-12:
        return None, float("inf")
    normal = _canonical_direction(normal)
    params = {"point": np.mean(sample, axis=0), "normal": normal}
    residuals = _plane_residuals(points, params)
    return params, float(np.sqrt(np.mean(residuals * residuals)))


def _fit_sphere_ls(points: np.ndarray) -> Tuple[Dict[str, object], float]:
    a = np.hstack([2.0 * points, np.ones((len(points), 1), dtype=np.float64)])
    b = np.sum(points * points, axis=1)
    sol, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
    center = sol[:3]
    radius2 = float(sol[3] + np.dot(center, center))
    radius = float(np.sqrt(max(radius2, 1e-12)))
    dist = np.linalg.norm(points - center.reshape(1, 3), axis=1)
    residual = float(np.sqrt(np.mean((dist - radius) ** 2)))
    return {"center": center, "radius": radius}, residual


def _fit_sphere_from_indices(
    points: np.ndarray, ids: Sequence[int]
) -> Tuple[Optional[Dict[str, object]], float]:
    ids = [int(i) for i in ids]
    if len(ids) < 4:
        return None, float("inf")
    sample = points[np.array(ids[:4], dtype=np.int64)]
    p0 = sample[0]
    a = 2.0 * (sample[1:] - p0.reshape(1, 3))
    b = np.sum(sample[1:] * sample[1:], axis=1) - float(np.dot(p0, p0))
    try:
        center = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return None, float("inf")
    radius = float(np.mean(np.linalg.norm(sample - center.reshape(1, 3), axis=1)))
    if radius < 1e-12:
        return None, float("inf")
    params = {"center": center, "radius": radius}
    residuals = _sphere_residuals(points, params)
    return params, float(np.sqrt(np.mean(residuals * residuals)))


def _orthonormal_basis_perp_to_axis(axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """平面 ⊥ axis 上的一组标准正交基 (e1, e2)。"""
    a = _canonical_direction(axis)
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(a, ref))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    e1 = np.cross(a, ref)
    ln1 = float(np.linalg.norm(e1))
    if ln1 < 1e-15:
        return np.zeros(3), np.zeros(3)
    e1 = e1 / ln1
    e2 = np.cross(a, e1)
    e2 = e2 / (float(np.linalg.norm(e2)) + 1e-15)
    return e1, e2


def _refine_cylinder_axis_local(
    points: np.ndarray,
    axis: np.ndarray,
    *,
    spans: Tuple[float, float, float] = (0.14, 0.06, 0.025),
    grid_n: int = 5,
) -> Tuple[Optional[Dict[str, object]], float]:
    """
    在轴线切空间做小范围网格搜索 + 每步 Kåsa 圆拟合，降低「轴方向略偏 → 鼓包」问题。
    """
    axis = _canonical_direction(axis)
    if float(np.linalg.norm(axis)) < 1e-12:
        return None, float("inf")
    params0, rmse0 = _fit_cylinder_kasa_in_plane(points, axis)
    if params0 is None:
        return None, float("inf")
    best_p, best_r = params0, rmse0
    a = np.asarray(best_p["axis"], dtype=np.float64)
    a = a / (float(np.linalg.norm(a)) + 1e-15)
    for span in spans:
        e1, e2 = _orthonormal_basis_perp_to_axis(a)
        if float(np.linalg.norm(e1)) < 1e-15:
            break
        if grid_n < 2:
            continue
        da_list = np.linspace(-span, span, int(grid_n))
        for da in da_list:
            for db in da_list:
                raw = a + float(da) * e1 + float(db) * e2
                ln = float(np.linalg.norm(raw))
                if ln < 1e-12:
                    continue
                a_try = _canonical_direction(raw)
                pk, rk = _fit_cylinder_kasa_in_plane(points, a_try)
                if pk is not None and rk < best_r - 1e-9:
                    best_p, best_r = pk, rk
                    a = np.asarray(best_p["axis"], dtype=np.float64)
                    a = a / (float(np.linalg.norm(a)) + 1e-15)
    return best_p, best_r


def _fit_cylinder_kasa_in_plane(
    points: np.ndarray, axis: np.ndarray
) -> Tuple[Optional[Dict[str, object]], float]:
    """
    真圆柱侧面：点在垂直于轴的平面上的投影落在一圆上。
    用 Kåsa 代数圆拟合求圆心（轴上一点）与半径，避免「轴线过质心」带来的偏移误差。
    """
    axis = _canonical_direction(axis)
    n = int(points.shape[0])
    if n < 4:
        return None, float("inf")
    e1, e2 = _orthonormal_basis_perp_to_axis(axis)
    if float(np.linalg.norm(e1)) < 1e-15:
        return None, float("inf")
    c = np.mean(points, axis=0)
    x = points - c.reshape(1, 3)
    u = (x @ e1.reshape(3, 1)).reshape(-1)
    v = (x @ e2.reshape(3, 1)).reshape(-1)
    rhs = u * u + v * v
    a_mat = np.column_stack([2.0 * u, 2.0 * v, np.ones(n, dtype=np.float64)])
    try:
        coef, _, rank, _ = np.linalg.lstsq(a_mat, rhs, rcond=1e-9)
    except np.linalg.LinAlgError:
        return None, float("inf")
    if rank < 3:
        return None, float("inf")
    cx, cy = float(coef[0]), float(coef[1])
    k = float(coef[2])
    r2 = cx * cx + cy * cy + k
    if r2 <= 1e-20:
        return None, float("inf")
    R = float(np.sqrt(r2))
    if R < 1e-12:
        return None, float("inf")
    axis_point = c + cx * e1 + cy * e2
    params = {"point": axis_point, "axis": axis, "radius": R}
    residuals = _cylinder_residuals(points, params)
    rmse = float(np.sqrt(np.mean(residuals * residuals)))
    return params, rmse


def _fit_axisymmetric_shape(
    points: np.ndarray, axis: np.ndarray, shape_type: str
) -> Tuple[Optional[Dict[str, object]], float]:
    if shape_type not in {"cylinder", "cone"}:
        return None, float("inf")
    axis = _canonical_direction(axis)
    if float(np.linalg.norm(axis)) < 1e-12:
        return None, float("inf")

    centroid = np.mean(points, axis=0)
    centered = points - centroid.reshape(1, 3)
    axial = centered @ axis.reshape(3, 1)
    foot = centroid.reshape(1, 3) + axial * axis.reshape(1, 3)
    radial = points - foot
    radial_dist = np.linalg.norm(radial, axis=1)
    axial_1d = axial.reshape(-1)

    if shape_type == "cylinder":
        radius_med = float(np.median(radial_dist))
        if radius_med < 1e-12:
            return None, float("inf")
        params_med = {"point": centroid, "axis": axis, "radius": radius_med}
        res_med = _cylinder_residuals(points, params_med)
        rmse_med = float(np.sqrt(np.mean(res_med * res_med)))

        params_k, rmse_k = _fit_cylinder_kasa_in_plane(points, axis)
        if params_k is not None:
            params_r, rmse_r = _refine_cylinder_axis_local(points, params_k["axis"])
            if params_r is not None and rmse_r < rmse_k:
                params_k, rmse_k = params_r, rmse_r
            if rmse_k <= rmse_med * 1.15:
                return params_k, rmse_k
        return params_med, rmse_med

    # cone：径向/轴向相对「过质心且平行于 axis 的直线」在质心不在锥轴上时会失真。
    # 先用与圆柱相同的 Kåsa 轴上一点作锚点，再对 r–t 做线性拟合。
    anchor = centroid
    params_k, _ = _fit_cylinder_kasa_in_plane(points, axis)
    if params_k is not None:
        anchor = np.asarray(params_k["point"], dtype=np.float64)
        axis = _safe_normalize(np.asarray(params_k["axis"], dtype=np.float64))
    rel_a = points - anchor.reshape(1, 3)
    axial_1d = (rel_a @ axis.reshape(3, 1)).reshape(-1)
    foot = anchor.reshape(1, 3) + axial_1d.reshape(-1, 1) * axis.reshape(1, 3)
    radial_dist = np.linalg.norm(points - foot, axis=1)

    order = np.argsort(axial_1d)
    axial_sorted = axial_1d[order]
    radial_sorted = radial_dist[order]
    a = np.column_stack([axial_sorted, np.ones_like(axial_sorted)])
    sol, _, _, _ = np.linalg.lstsq(a, radial_sorted, rcond=None)
    slope = float(sol[0])
    intercept = float(sol[1])
    if abs(slope) < 1e-7:
        return None, float("inf")
    half_angle = float(np.arctan(abs(slope)))
    if half_angle < np.deg2rad(1.2) or half_angle > np.deg2rad(87.0):
        return None, float("inf")
    apex_offset = -intercept / slope
    apex = anchor + apex_offset * axis
    height_sign = 1.0 if slope >= 0.0 else -1.0
    params = {
        "apex": apex,
        "axis": axis * height_sign,
        "half_angle": half_angle,
    }
    residuals = _cone_residuals(points, params)
    return params, float(np.sqrt(np.mean(residuals * residuals)))


def _axes_from_face_normals(face_normals: np.ndarray) -> List[np.ndarray]:
    """
    圆柱/圆锥侧面：面法向近似落在与轴线垂直的平面内，
    法向协方差的最小特征方向 ≈ 轴线方向（对球面则三向接近各向同性，不会稳定占优）。
    """
    normals = np.asarray(face_normals, dtype=np.float64)
    if normals.ndim != 2 or normals.shape[0] < 3 or normals.shape[1] != 3:
        return []
    mean = np.mean(normals, axis=0)
    centered = normals - mean.reshape(1, 3)
    _, s, vh = np.linalg.svd(centered, full_matrices=False)
    if s.size < 3 or float(s[0]) < 1e-15:
        return []
    # 圆柱：s2 << s0,s1 时第三主轴为轴线（略放宽，避免漏掉宽条带圆柱）
    ratio = float(s[2] / (s[0] + 1e-15))
    if ratio > 0.58:
        return []
    out: List[np.ndarray] = []
    for i in (2, 1, 0):
        ax = _canonical_direction(vh[i])
        if float(np.linalg.norm(ax)) < 1e-12:
            continue
        if not any(abs(float(np.dot(ax, ref))) > 0.995 for ref in out):
            out.append(ax)
    return out


def _candidate_axes(
    points: np.ndarray,
    rng: np.random.Generator,
    face_normals: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    centered = points - np.mean(points, axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axes: List[np.ndarray] = []
    for i in range(3):
        ax = _canonical_direction(vh[i])
        if float(np.linalg.norm(ax)) > 1e-12:
            axes.append(ax)
    if face_normals is not None:
        for ax in _axes_from_face_normals(face_normals):
            if not any(abs(float(np.dot(ax, ref))) > 0.995 for ref in axes):
                axes.append(ax)
    n = int(points.shape[0])
    if n >= 2:
        sample_count = min(18, n * (n - 1) // 2)
        for _ in range(sample_count):
            i, j = rng.choice(n, size=2, replace=False)
            diff = points[int(j)] - points[int(i)]
            if float(np.linalg.norm(diff)) < 1e-10:
                continue
            axes.append(_canonical_direction(diff))
    # 圆锥轴未必落在 PCA/面法向候选里；补充随机方向以提高锥面识别率。
    n_rand = min(40, max(16, n // 2))
    for _ in range(int(n_rand)):
        raw = rng.standard_normal(3)
        ln = float(np.linalg.norm(raw))
        if ln < 1e-12:
            continue
        axes.append(_canonical_direction(raw))
    unique: List[np.ndarray] = []
    for axis in axes:
        if float(np.linalg.norm(axis)) < 1e-12:
            continue
        if not any(abs(float(np.dot(axis, ref))) > 0.995 for ref in unique):
            unique.append(axis)
    return unique


def _plane_residuals(points: np.ndarray, params: Mapping[str, object]) -> np.ndarray:
    point = np.asarray(params["point"], dtype=np.float64)
    normal = _safe_normalize(np.asarray(params["normal"], dtype=np.float64))
    return np.abs((points - point.reshape(1, 3)) @ normal.reshape(3, 1)).reshape(-1)


def _sphere_residuals(points: np.ndarray, params: Mapping[str, object]) -> np.ndarray:
    center = np.asarray(params["center"], dtype=np.float64)
    radius = float(params["radius"])
    dist = np.linalg.norm(points - center.reshape(1, 3), axis=1)
    return np.abs(dist - radius)


def _cylinder_residuals(points: np.ndarray, params: Mapping[str, object]) -> np.ndarray:
    axis_point = np.asarray(params["point"], dtype=np.float64)
    axis = _safe_normalize(np.asarray(params["axis"], dtype=np.float64))
    radius = float(params["radius"])
    rel = points - axis_point.reshape(1, 3)
    axial = rel @ axis.reshape(3, 1)
    foot = axis_point.reshape(1, 3) + axial * axis.reshape(1, 3)
    radial = np.linalg.norm(points - foot, axis=1)
    return np.abs(radial - radius)


def _cone_residuals(points: np.ndarray, params: Mapping[str, object]) -> np.ndarray:
    apex = np.asarray(params["apex"], dtype=np.float64)
    axis = _safe_normalize(np.asarray(params["axis"], dtype=np.float64))
    half_angle = float(params["half_angle"])
    rel = points - apex.reshape(1, 3)
    axial = rel @ axis.reshape(3, 1)
    foot = apex.reshape(1, 3) + axial * axis.reshape(1, 3)
    radial = np.linalg.norm(points - foot, axis=1)
    target = np.abs(axial.reshape(-1)) * np.tan(half_angle)
    return np.abs(radial - target)


def _robust_model_score(
    residuals: np.ndarray,
    diag: float,
    n_params: int,
) -> Tuple[float, float, float]:
    if residuals.size == 0:
        return float("inf"), 0.0, float("inf")
    q50 = float(np.quantile(residuals, 0.5))
    q80 = float(np.quantile(residuals, 0.8))
    threshold = max(0.02 * diag, 2.5 * q50, 1e-12)
    inlier_ratio = float(np.mean(residuals <= threshold))
    robust_res = 0.65 * q80 + 0.35 * q50
    score = robust_res / max(diag, 1e-12) + 0.06 * (1.0 - inlier_ratio) + 3e-4 * n_params
    return score, inlier_ratio, robust_res


def _confidence_from_metrics(
    score: float, inlier_ratio: float, n_points: int
) -> str:
    if n_points < 5:
        return "low"
    if score < 8e-3 and inlier_ratio > 0.8:
        return "high"
    if score < 2.5e-2 and inlier_ratio > 0.55:
        return "medium"
    return "low"


def _evaluate_candidate(
    surface_type: str,
    params: Mapping[str, object],
    points: np.ndarray,
    diag: float,
) -> Tuple[float, float, float]:
    if surface_type == "plane":
        residuals = _plane_residuals(points, params)
        return _robust_model_score(residuals, diag, 4)
    if surface_type == "sphere":
        residuals = _sphere_residuals(points, params)
        return _robust_model_score(residuals, diag, 4)
    if surface_type == "cylinder":
        residuals = _cylinder_residuals(points, params)
        return _robust_model_score(residuals, diag, 5)
    if surface_type == "cone":
        residuals = _cone_residuals(points, params)
        return _robust_model_score(residuals, diag, 5)
    return float("inf"), 0.0, float("inf")


def _rmse_for_surface(
    surface_type: str,
    params: Mapping[str, object],
    points: np.ndarray,
) -> float:
    if surface_type == "plane":
        residuals = _plane_residuals(points, params)
    elif surface_type == "sphere":
        residuals = _sphere_residuals(points, params)
    elif surface_type == "cylinder":
        residuals = _cylinder_residuals(points, params)
    elif surface_type == "cone":
        residuals = _cone_residuals(points, params)
    else:
        return float("inf")
    return float(np.sqrt(np.mean(residuals * residuals))) if residuals.size else float("inf")


def _refine_candidate_scipy(
    surface_type: str,
    params: Mapping[str, object],
    points: np.ndarray,
    diag: float,
) -> Tuple[Dict[str, object], float, float, float, Dict[str, object]]:
    """Local robust nonlinear refinement for the selected analytic primitive."""
    diagnostics: Dict[str, object] = {
        "enabled": bool(os.environ.get("CAD_HOLE_FIT_REFINE", "1") != "0"),
        "available": bool(least_squares is not None),
        "accepted": False,
        "surface_type": str(surface_type),
    }
    original = dict(params)
    before_score, before_inlier, _ = _evaluate_candidate(surface_type, original, points, diag)
    before_rmse = _rmse_for_surface(surface_type, original, points)
    diagnostics.update(
        {
            "score_before": float(before_score),
            "rmse_before": float(before_rmse),
            "inlier_ratio_before": float(before_inlier),
        }
    )
    if (
        least_squares is None
        or os.environ.get("CAD_HOLE_FIT_REFINE", "1") == "0"
        or surface_type not in {"sphere", "cylinder", "cone"}
        or points.shape[0] < 5
    ):
        return original, float(before_rmse), float(before_score), float(before_inlier), diagnostics

    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    scale = max(float(diag), 1e-12)
    min_radius_log = float(np.log(max(1e-9 * scale, 1e-12)))
    max_radius_log = float(np.log(max(1e3 * scale, 1e-9)))

    def _axis_from_raw(raw: np.ndarray) -> np.ndarray:
        axis = _safe_normalize(np.asarray(raw, dtype=np.float64).reshape(3))
        if float(np.linalg.norm(axis)) < 1e-12:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return axis

    def _accept(refined: Dict[str, object], result) -> Tuple[Dict[str, object], float, float, float, Dict[str, object]]:
        after_score, after_inlier, _ = _evaluate_candidate(surface_type, refined, pts, diag)
        after_rmse = _rmse_for_surface(surface_type, refined, pts)
        diagnostics.update(
            {
                "success": bool(getattr(result, "success", False)),
                "nfev": int(getattr(result, "nfev", 0)),
                "score_after": float(after_score),
                "rmse_after": float(after_rmse),
                "inlier_ratio_after": float(after_inlier),
            }
        )
        score_not_worse = after_score <= before_score + 1e-8
        rmse_improved = after_rmse <= before_rmse * 0.995
        score_nearly_stable = after_score <= before_score * 1.005 + 1e-8
        if np.isfinite(after_score) and (score_not_worse or (rmse_improved and score_nearly_stable)):
            diagnostics["accepted"] = True
            return refined, float(after_rmse), float(after_score), float(after_inlier), diagnostics
        diagnostics["reject_reason"] = "score_not_improved"
        return original, float(before_rmse), float(before_score), float(before_inlier), diagnostics

    try:
        if surface_type == "sphere":
            c0 = np.asarray(original["center"], dtype=np.float64)
            r0 = max(float(original["radius"]), 1e-12)
            x0 = np.r_[c0, np.clip(np.log(r0), min_radius_log, max_radius_log)]

            def residual(x: np.ndarray) -> np.ndarray:
                c = x[:3]
                r = float(np.exp(np.clip(x[3], min_radius_log, max_radius_log)))
                return (np.linalg.norm(pts - c.reshape(1, 3), axis=1) - r) / scale

            lb = np.r_[np.full(3, -np.inf), min_radius_log]
            ub = np.r_[np.full(3, np.inf), max_radius_log]
            res = least_squares(
                residual,
                x0,
                bounds=(lb, ub),
                loss="soft_l1",
                f_scale=0.01,
                max_nfev=48,
            )
            refined = {
                "center": res.x[:3],
                "radius": float(np.exp(np.clip(res.x[3], min_radius_log, max_radius_log))),
            }
            return _accept(refined, res)

        if surface_type == "cylinder":
            p0 = np.asarray(original["point"], dtype=np.float64)
            a0 = _safe_normalize(np.asarray(original["axis"], dtype=np.float64))
            r0 = max(float(original["radius"]), 1e-12)
            x0 = np.r_[p0, a0, np.clip(np.log(r0), min_radius_log, max_radius_log)]

            def residual(x: np.ndarray) -> np.ndarray:
                p = x[:3]
                a = _axis_from_raw(x[3:6])
                r = float(np.exp(np.clip(x[6], min_radius_log, max_radius_log)))
                rel = pts - p.reshape(1, 3)
                axial = (rel @ a.reshape(3, 1)).reshape(-1)
                foot = p.reshape(1, 3) + axial.reshape(-1, 1) * a.reshape(1, 3)
                radial = np.linalg.norm(pts - foot, axis=1)
                return (radial - r) / scale

            lb = np.r_[np.full(6, -np.inf), min_radius_log]
            ub = np.r_[np.full(6, np.inf), max_radius_log]
            res = least_squares(
                residual,
                x0,
                bounds=(lb, ub),
                loss="soft_l1",
                f_scale=0.01,
                max_nfev=56,
            )
            refined = {
                "point": res.x[:3],
                "axis": _canonical_direction(res.x[3:6]),
                "radius": float(np.exp(np.clip(res.x[6], min_radius_log, max_radius_log))),
            }
            return _accept(refined, res)

        if surface_type == "cone":
            apex0 = np.asarray(original["apex"], dtype=np.float64)
            axis0 = _safe_normalize(np.asarray(original["axis"], dtype=np.float64))
            angle0 = float(np.clip(float(original["half_angle"]), np.deg2rad(1.2), np.deg2rad(87.0)))
            x0 = np.r_[apex0, axis0, angle0]
            lb = np.r_[
                np.full(3, -np.inf),
                np.full(3, -np.inf),
                np.deg2rad(1.2),
            ]
            ub = np.r_[
                np.full(3, np.inf),
                np.full(3, np.inf),
                np.deg2rad(87.0),
            ]

            def residual(x: np.ndarray) -> np.ndarray:
                apex = x[:3]
                axis = _axis_from_raw(x[3:6])
                angle = float(x[6])
                rel = pts - apex.reshape(1, 3)
                axial = (rel @ axis.reshape(3, 1)).reshape(-1)
                foot = apex.reshape(1, 3) + axial.reshape(-1, 1) * axis.reshape(1, 3)
                radial = np.linalg.norm(pts - foot, axis=1)
                target = np.abs(axial) * np.tan(angle)
                return (radial - target) / scale

            res = least_squares(
                residual,
                x0,
                bounds=(lb, ub),
                loss="soft_l1",
                f_scale=0.01,
                max_nfev=64,
            )
            refined = {
                "apex": res.x[:3],
                "axis": _canonical_direction(res.x[3:6]),
                "half_angle": float(res.x[6]),
            }
            return _accept(refined, res)
    except Exception as exc:
        diagnostics["error"] = f"{type(exc).__name__}: {exc}"
        return original, float(before_rmse), float(before_score), float(before_inlier), diagnostics

    return original, float(before_rmse), float(before_score), float(before_inlier), diagnostics


def _axisymmetric_normal_spread_weight(points: np.ndarray) -> float:
    """
    三角片支撑上的法向往往分散（normal_spread 大），但顶点仍近似落在同一平面上。
    此时应对「罚平面 / 奖圆柱」的项降权，避免平面片被错判成柱面。

    返回 [0,1]：越大表示越应保留原有 normal_spread 启发式（旋转面、弯面等）。
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 5:
        return 1.0
    c = np.mean(pts, axis=0)
    x = pts - c
    try:
        _, s, _ = np.linalg.svd(x, full_matrices=False)
    except np.linalg.LinAlgError:
        return 1.0
    if s.size < 3 or float(s[0]) < 1e-15:
        return 1.0
    flatness = float(s[-1] / s[0])
    diag = _bbox_diag(pts)
    plane_params, _ = _fit_plane_ls(pts)
    pr = _plane_residuals(pts, plane_params)
    plane_rel = float(np.sqrt(np.mean(pr * pr)) / max(diag, 1e-12))
    # 点云很扁且到 LSQ 平面距离相对包围盒很小 → 权重压低
    w_flat = float(np.clip((flatness - 0.006) / (0.088 - 0.006), 0.0, 1.0))
    w_pr = float(np.clip((plane_rel - 0.0035) / (0.032 - 0.0035), 0.0, 1.0))
    return max(w_flat, w_pr)


def _fit_best_model_robust(
    points: np.ndarray,
    rng: np.random.Generator,
    *,
    normal_spread: float = 0.0,
    face_normals: Optional[np.ndarray] = None,
) -> Tuple[str, Dict[str, object], float, float, float, Dict[str, object]]:
    diag = _bbox_diag(points)
    nsw = _axisymmetric_normal_spread_weight(points)
    candidates: List[Tuple[str, Dict[str, object], float, float, float]] = []

    plane_params, plane_res = _fit_plane_ls(points)
    plane_score, plane_inlier_ratio, plane_robust_res = _evaluate_candidate(
        "plane", plane_params, points, diag
    )
    if normal_spread > 0.18:
        plane_score += 0.25 * normal_spread * nsw
    candidates.append(("plane", plane_params, plane_res, plane_score, plane_inlier_ratio))
    for _ in range(min(24, max(6, points.shape[0] // 2))):
        ids = rng.choice(points.shape[0], size=3, replace=False)
        params, residual = _fit_plane_from_indices(points, ids)
        if params is None:
            continue
        score, inlier_ratio, _ = _evaluate_candidate("plane", params, points, diag)
        if normal_spread > 0.18:
            score += 0.25 * normal_spread * nsw
        candidates.append(("plane", params, residual, score, inlier_ratio))

    if points.shape[0] >= 4:
        sphere_params, sphere_res = _fit_sphere_ls(points)
        score, inlier_ratio, _ = _evaluate_candidate("sphere", sphere_params, points, diag)
        candidates.append(("sphere", sphere_params, sphere_res, score + 2e-4, inlier_ratio))
        for _ in range(min(20, max(6, points.shape[0] // 2))):
            ids = rng.choice(points.shape[0], size=4, replace=False)
            params, residual = _fit_sphere_from_indices(points, ids)
            if params is None:
                continue
            score, inlier_ratio, _ = _evaluate_candidate("sphere", params, points, diag)
            candidates.append(("sphere", params, residual, score + 2e-4, inlier_ratio))

    if points.shape[0] >= 5:
        for axis in _candidate_axes(points, rng, face_normals):
            cyl_params, cyl_res = _fit_axisymmetric_shape(points, axis, "cylinder")
            if cyl_params is not None:
                score, inlier_ratio, _ = _evaluate_candidate(
                    "cylinder", cyl_params, points, diag
                )
                if normal_spread > 0.18:
                    score -= 0.03 * min(normal_spread, 0.75) * nsw
                candidates.append(
                    ("cylinder", cyl_params, cyl_res, score + 4e-4, inlier_ratio)
                )
            cone_params, cone_res = _fit_axisymmetric_shape(points, axis, "cone")
            if cone_params is not None:
                score, inlier_ratio, _ = _evaluate_candidate(
                    "cone", cone_params, points, diag
                )
                if normal_spread > 0.18:
                    score -= 0.025 * min(normal_spread, 0.75) * nsw
                candidates.append(("cone", cone_params, cone_res, score + 5e-4, inlier_ratio))

    candidates_sorted = sorted(candidates, key=lambda item: item[3])
    best_type, best_params, best_residual, best_score, best_inlier_ratio = candidates_sorted[0]
    second_distinct = next(
        (
            item for item in candidates_sorted[1:]
            if item[0] != best_type
        ),
        None,
    )
    diagnostics = {
        "normal_spread": float(normal_spread),
        "normal_spread_axis_weight": float(nsw),
        "best_score": float(best_score),
        "best_type": str(best_type),
    }
    if second_distinct is not None:
        diagnostics["second_type"] = str(second_distinct[0])
        diagnostics["score_gap"] = float(second_distinct[3] - best_score)

    # 圆台/锥面侧面：Kåsa 锚点下圆锥与圆柱残差常接近，综合分会偏向圆柱；若二者分数接近且半角合理则取圆锥。
    if best_type == "cylinder":
        cone_items = [c for c in candidates if c[0] == "cone"]
        if cone_items:
            best_cone_entry = min(cone_items, key=lambda item: item[3])
            _, cone_params, cone_res, cone_score, cone_inlier = best_cone_entry
            gap = float(cone_score - best_score)
            ha = float(cone_params.get("half_angle", 0.0))
            cyl_res = float(best_residual)
            cone_ok = float(cone_res) < max(cyl_res * 6.5, cyl_res + 0.004)
            if (
                gap < 0.022
                and np.deg2rad(1.2) < ha < np.deg2rad(82.0)
                and cone_ok
            ):
                diagnostics["cylinder_to_cone_preference"] = True
                diagnostics["cone_score_gap_vs_cylinder"] = gap
                best_type = "cone"
                best_params = cone_params
                best_residual = float(cone_res)
                best_score = float(cone_score)
                best_inlier_ratio = float(cone_inlier)

    if best_type == "plane" and normal_spread > 0.32:
        axisymmetric = [
            item for item in candidates_sorted
            if item[0] in {"cylinder", "cone"}
        ]
        if axisymmetric:
            alt_type, alt_params, alt_residual, alt_score, alt_inlier_ratio = axisymmetric[0]
            if alt_score < best_score + 0.12:
                diagnostics["plane_degeneracy_override"] = str(alt_type)
                best_type, best_params, best_residual, best_score, best_inlier_ratio = (
                    alt_type,
                    alt_params,
                    alt_residual,
                    alt_score,
                    alt_inlier_ratio,
                )

    # 法向高度分散时球面得分常更优；仅当圆柱几何残差与球面同一量级时才改为圆柱，避免鼓包。
    if best_type == "sphere" and normal_spread > 0.42:
        cyl_item = next((c for c in candidates_sorted if c[0] == "cylinder"), None)
        sph_item = next((c for c in candidates_sorted if c[0] == "sphere"), None)
        if cyl_item is not None and sph_item is not None:
            gap = float(cyl_item[3] - best_score)
            cyl_rmse = float(cyl_item[2])
            sph_rmse = float(sph_item[2])
            geom_ok = cyl_rmse < 1.38 * max(sph_rmse, 1e-9) and cyl_rmse < 0.22 * max(
                diag, 1e-12
            )
            if gap < 0.016 and geom_ok:
                diagnostics["sphere_to_cylinder_override"] = "high_normal_spread_close_cylinder"
                best_type, best_params, best_residual, best_score, best_inlier_ratio = (
                    cyl_item[0],
                    cyl_item[1],
                    cyl_item[2],
                    cyl_item[3],
                    cyl_item[4],
                )

    refined_params, refined_residual, refined_score, refined_inlier_ratio, refine_diag = (
        _refine_candidate_scipy(best_type, best_params, points, diag)
    )
    diagnostics["scipy_refinement"] = refine_diag
    if bool(refine_diag.get("accepted", False)):
        best_params = refined_params
        best_residual = float(refined_residual)
        best_score = float(refined_score)
        best_inlier_ratio = float(refined_inlier_ratio)

    if (
        best_score > 5e-2
        or (
            best_type == "plane"
            and normal_spread > 0.22
            and second_distinct is not None
            and second_distinct[3] - best_score < 2.5e-2
        )
    ):
        transition_params = {
            "anchor": np.mean(points, axis=0),
            "point": np.mean(points, axis=0),
            "normal": _fit_plane_ls(points)[0]["normal"],
        }
        diagnostics["transition_reason"] = (
            "high_residual"
            if best_score > 5e-2
            else "ambiguous_or_nonplanar_patch"
        )
        return (
            TRANSITION_SURFACE_TYPE,
            transition_params,
            float(best_residual),
            float(best_score),
            float(best_inlier_ratio),
            diagnostics,
        )

    return (
        best_type,
        best_params,
        float(best_residual),
        float(best_score),
        float(best_inlier_ratio),
        diagnostics,
    )


def _select_rng_seed(
    patch_label: int, support_vertex_indices: Sequence[int], n_points: int
) -> int:
    seed = int(patch_label) * 1_000_003 + int(n_points) * 97
    for idx in support_vertex_indices[:16]:
        seed = (seed * 1_315_423_911 + int(idx) + 17) % (2**32)
    return seed


def fit_patch_surface(
    patch_label: int,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_indices: Sequence[int],
) -> SurfaceFit:
    support_vertex_indices, support_points = _patch_support_points(
        vertices,
        faces,
        face_indices,
    )
    support_face_normals = _patch_support_face_normals(vertices, faces, face_indices)
    if len(support_points) < 3:
        return SurfaceFit(
            patch_label=int(patch_label),
            surface_type="freeform_fallback",
            surface_params={"anchor": np.mean(support_points, axis=0) if len(support_points) else np.zeros(3, dtype=np.float64)},
            fit_residual=float("inf"),
            fit_score=float("inf"),
            fit_confidence="low",
            support_face_indices=[int(x) for x in face_indices],
            support_vertex_indices=support_vertex_indices,
            support_points=support_points,
            fit_diagnostics={},
        )

    rng = np.random.default_rng(
        _select_rng_seed(int(patch_label), support_vertex_indices, len(support_points))
    )
    surface_type, surface_params, residual, score, inlier_ratio, diagnostics = _fit_best_model_robust(
        support_points,
        rng,
        normal_spread=_normal_spread(support_face_normals),
        face_normals=support_face_normals,
    )
    confidence = _confidence_from_metrics(score, inlier_ratio, len(support_points))
    return SurfaceFit(
        patch_label=int(patch_label),
        surface_type=surface_type,
        surface_params=surface_params,
        fit_residual=float(residual),
        fit_score=float(score),
        fit_confidence=confidence,
        support_face_indices=[int(x) for x in face_indices],
        support_vertex_indices=support_vertex_indices,
        support_points=support_points,
        fit_diagnostics=diagnostics,
    )


def fit_patch_surfaces(
    vertices: np.ndarray,
    faces: np.ndarray,
    patch_face_indices: Mapping[int, Sequence[int]],
) -> Dict[int, SurfaceFit]:
    out: Dict[int, SurfaceFit] = {}
    for patch_label, face_indices in patch_face_indices.items():
        out[int(patch_label)] = fit_patch_surface(
            int(patch_label),
            vertices,
            faces,
            face_indices,
        )
    return out


def project_point_to_surface(fit: SurfaceFit, point: np.ndarray) -> np.ndarray:
    surface_type = fit.surface_type
    params = fit.surface_params
    p = np.asarray(point, dtype=np.float64)

    if surface_type in {"plane", TRANSITION_SURFACE_TYPE, "freeform_fallback"}:
        plane_point = np.asarray(params["point"], dtype=np.float64)
        normal = np.asarray(params["normal"], dtype=np.float64)
        signed = float(np.dot(p - plane_point, normal))
        return p - signed * normal

    if surface_type == "sphere":
        center = np.asarray(params["center"], dtype=np.float64)
        radius = float(params["radius"])
        direction = p - center
        ln = float(np.linalg.norm(direction))
        if ln < 1e-15:
            direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            ln = 1.0
        return center + radius * direction / ln

    if surface_type == "cylinder":
        axis_point = np.asarray(params["point"], dtype=np.float64)
        axis = _safe_normalize(np.asarray(params["axis"], dtype=np.float64))
        radius = float(params["radius"])
        rel = p - axis_point
        axial = float(np.dot(rel, axis))
        foot = axis_point + axial * axis
        radial = p - foot
        lr = float(np.linalg.norm(radial))
        if lr < 1e-15:
            radial = np.cross(axis, np.array([1.0, 0.0, 0.0], dtype=np.float64))
            if float(np.linalg.norm(radial)) < 1e-15:
                radial = np.cross(axis, np.array([0.0, 1.0, 0.0], dtype=np.float64))
            lr = float(np.linalg.norm(radial))
        return foot + radius * radial / lr

    if surface_type == "cone":
        apex = np.asarray(params["apex"], dtype=np.float64)
        axis = _safe_normalize(np.asarray(params["axis"], dtype=np.float64))
        half_angle = float(params["half_angle"])
        rel = p - apex
        axial = float(np.dot(rel, axis))
        foot = apex + axial * axis
        radial = p - foot
        lr = float(np.linalg.norm(radial))
        if lr < 1e-15:
            radial = np.cross(axis, np.array([1.0, 0.0, 0.0], dtype=np.float64))
            if float(np.linalg.norm(radial)) < 1e-15:
                radial = np.cross(axis, np.array([0.0, 1.0, 0.0], dtype=np.float64))
            lr = float(np.linalg.norm(radial))
        target_r = abs(axial) * np.tan(half_angle)
        target_r = max(target_r, 1e-12)
        axial_sign = 1.0 if axial >= 0.0 else -1.0
        return apex + axial_sign * abs(axial) * axis + target_r * radial / lr

    return p.copy()


def project_point_to_surface_pair(
    fit_a: SurfaceFit,
    fit_b: SurfaceFit,
    point: np.ndarray,
    iterations: int = 6,
) -> np.ndarray:
    p = np.asarray(point, dtype=np.float64).copy()
    for _ in range(max(1, int(iterations))):
        p = project_point_to_surface(fit_a, p)
        p = project_point_to_surface(fit_b, p)
    return p
