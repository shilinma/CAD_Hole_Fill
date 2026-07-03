#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
局部解析面参数化。

为每个子孔提供：
- 边界点 3D -> 2D 参数域
- 2D 参数点 -> 3D 曲面点
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .surface_fitting import (
    SurfaceFit,
    TRANSITION_SURFACE_TYPE,
    project_point_to_surface,
)


@dataclass
class SurfaceParameterization:
    patch_label: int
    kind: str
    uv_boundary_points: np.ndarray
    parameter_data: Dict[str, object]


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


def _align_frame_with_reference(
    normal: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    reference_normal: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = np.asarray(normal, dtype=np.float64)
    u = np.asarray(u_axis, dtype=np.float64)
    v = np.asarray(v_axis, dtype=np.float64)
    if reference_normal is not None:
        ref = np.asarray(reference_normal, dtype=np.float64)
        if float(np.linalg.norm(ref)) > 1e-12 and float(np.dot(n, ref)) < 0.0:
            n = -n
            u = -u
    if float(np.dot(np.cross(u, v), n)) < 0.0:
        v = -v
    return n, u, v


def _parameterize_plane(
    fit: SurfaceFit,
    points: np.ndarray,
    kind: str,
    reference_normal: Optional[np.ndarray],
) -> SurfaceParameterization:
    plane_point = np.asarray(fit.surface_params.get("point", np.mean(points, axis=0)), dtype=np.float64)
    normal = np.asarray(fit.surface_params.get("normal", np.array([0.0, 0.0, 1.0], dtype=np.float64)), dtype=np.float64)
    u_axis, v_axis = _orthonormal_basis(normal)
    normal, u_axis, v_axis = _align_frame_with_reference(normal, u_axis, v_axis, reference_normal)
    rel = points - plane_point.reshape(1, 3)
    uv = np.column_stack([rel @ u_axis, rel @ v_axis])
    return SurfaceParameterization(
        patch_label=int(fit.patch_label),
        kind=kind,
        uv_boundary_points=uv,
        parameter_data={
            "origin": plane_point,
            "u_axis": u_axis,
            "v_axis": v_axis,
            "normal": normal,
        },
    )

def _orient_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _point_on_segment_2d(
    a: np.ndarray, b: np.ndarray, p: np.ndarray, eps: float
) -> bool:
    if abs(_orient_2d(a, b, p)) > eps:
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
    o1 = _orient_2d(a, b, c)
    o2 = _orient_2d(a, b, d)
    o3 = _orient_2d(c, d, a)
    o4 = _orient_2d(c, d, b)

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


def _uv_closed_polygon_is_simple(uv: np.ndarray) -> bool:
    """检测闭合折线在 UV 上是否自交（与 hole_patch_triangulation 中逻辑一致）。"""
    uv = np.asarray(uv, dtype=np.float64)
    n = int(uv.shape[0])
    if n < 3:
        return True
    diag = max(float(np.linalg.norm(np.ptp(uv, axis=0))), 1.0)
    eps = max(1e-12, 1e-10 * diag)
    loop: Sequence[int] = list(range(n))
    for i in range(n):
        a0 = int(loop[i])
        a1 = int(loop[(i + 1) % n])
        pa0, pa1 = uv[a0], uv[a1]
        for j in range(i + 1, n):
            if (j + 1) % n == i or (i + 1) % n == j:
                continue
            b0 = int(loop[j])
            b1 = int(loop[(j + 1) % n])
            if len({a0, a1, b0, b1}) < 4:
                continue
            if _segments_intersect_2d(pa0, pa1, uv[b0], uv[b1], eps):
                return False
    return True


def _unwrap_angles(theta: np.ndarray) -> np.ndarray:
    if len(theta) == 0:
        return theta
    out = theta.copy()
    for i in range(1, len(out)):
        delta = out[i] - out[i - 1]
        while delta > np.pi:
            out[i] -= 2.0 * np.pi
            delta = out[i] - out[i - 1]
        while delta < -np.pi:
            out[i] += 2.0 * np.pi
            delta = out[i] - out[i - 1]
    return out


def _enumerate_cylinder_theta_unwraps(principal_theta: np.ndarray) -> List[np.ndarray]:
    """
    闭合孔边界在柱面上的角向可以有多种 2π 缠绕与展开方式。
    只在「与主值 atan2 一致 mod 2π」的前提下换分支，不改变各顶点径向方向的几何含义。
    """
    th = np.asarray(principal_theta, dtype=np.float64).reshape(-1)
    n = int(th.shape[0])
    if n == 0:
        return []
    raw: List[np.ndarray] = []

    raw.append(_unwrap_angles(th.copy()))

    ext_dup = np.concatenate([th, th[:1]])
    raw.append(np.unwrap(ext_dup)[:-1])

    for w in range(-5, 6):
        if w == 0:
            continue
        extended = np.concatenate([th, [th[0] + 2.0 * np.pi * float(w)]])
        uu = np.unwrap(extended)
        raw.append(uu[:-1])

    out: List[np.ndarray] = []
    for arr in raw:
        a = np.asarray(arr, dtype=np.float64)
        if any(np.allclose(a, b, rtol=0.0, atol=1e-8) for b in out):
            continue
        out.append(a)
    return out


def _pick_best_cylinder_uv(
    theta_candidates: Sequence[np.ndarray],
    z: np.ndarray,
    radius: float,
) -> Tuple[np.ndarray, bool, np.ndarray]:
    """
    在 (u=Rθ, z) 上优先选边界多边形不自交的候选；否则选 u=Rθ 方向跨度最小者。
    返回 (uv, 是否简单, 对应的 θ 序列)。
    """
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    best_simple: Optional[Tuple[float, np.ndarray, np.ndarray]] = None
    best_any: Optional[Tuple[float, np.ndarray, np.ndarray]] = None
    r = max(float(radius), 1e-12)

    for theta_c in theta_candidates:
        tc = np.asarray(theta_c, dtype=np.float64).reshape(-1)
        if tc.shape[0] != z.shape[0]:
            continue
        u_coord = r * tc
        uv = np.column_stack([u_coord, z])
        span_u = float(np.ptp(u_coord))
        ok = _uv_closed_polygon_is_simple(uv)
        if ok:
            if best_simple is None or span_u < best_simple[0]:
                best_simple = (span_u, uv, tc.copy())
        if best_any is None or span_u < best_any[0]:
            best_any = (span_u, uv, tc.copy())

    if best_simple is not None:
        return best_simple[1], True, best_simple[2]
    if best_any is not None:
        return best_any[1], False, best_any[2]
    tc0 = np.asarray(theta_candidates[0], dtype=np.float64).reshape(-1)
    uv0 = np.column_stack([r * tc0, z])
    return uv0, False, tc0


def _cylinder_build_local_frame(
    fit: SurfaceFit,
    points: np.ndarray,
    reference_normal: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    axis_point = np.asarray(fit.surface_params["point"], dtype=np.float64)
    axis = np.asarray(fit.surface_params["axis"], dtype=np.float64)
    axis = axis / (float(np.linalg.norm(axis)) + 1e-15)
    radius = float(fit.surface_params["radius"])

    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    seed = pts[0] - axis_point
    seed = seed - float(np.dot(seed, axis)) * axis
    if float(np.linalg.norm(seed)) < 1e-12:
        seed = np.cross(axis, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        if float(np.linalg.norm(seed)) < 1e-12:
            seed = np.cross(axis, np.array([0.0, 1.0, 0.0], dtype=np.float64))
    x_axis = seed / (float(np.linalg.norm(seed)) + 1e-15)
    y_axis = np.cross(axis, x_axis)
    y_axis = y_axis / (float(np.linalg.norm(y_axis)) + 1e-15)
    radial_ref = np.mean(pts - axis_point.reshape(1, 3), axis=0)
    radial_ref = radial_ref - float(np.dot(radial_ref, axis)) * axis
    ref_normal = radial_ref if float(np.linalg.norm(radial_ref)) > 1e-12 else reference_normal
    _normal, x_axis, y_axis = _align_frame_with_reference(
        np.cross(x_axis, y_axis), x_axis, y_axis, ref_normal
    )
    return axis_point, axis, radius, x_axis, y_axis


def _try_parameterize_cylinder_rtheta(
    fit: SurfaceFit,
    points: np.ndarray,
    reference_normal: Optional[np.ndarray],
) -> Optional[SurfaceParameterization]:
    axis_point, axis, radius, x_axis, y_axis = _cylinder_build_local_frame(
        fit, points, reference_normal
    )
    rel = np.asarray(points, dtype=np.float64).reshape(-1, 3) - axis_point.reshape(1, 3)
    z = rel @ axis
    radial = rel - z.reshape(-1, 1) * axis.reshape(1, 3)
    principal_theta = np.arctan2(radial @ y_axis, radial @ x_axis)
    theta_candidates = _enumerate_cylinder_theta_unwraps(principal_theta)
    if not theta_candidates:
        theta_candidates = [_unwrap_angles(principal_theta.copy())]
    uv, ok_r, _ = _pick_best_cylinder_uv(theta_candidates, z, radius)
    if not ok_r:
        return None
    return SurfaceParameterization(
        patch_label=int(fit.patch_label),
        kind="cylinder",
        uv_boundary_points=uv,
        parameter_data={
            "axis_point": axis_point,
            "axis": axis,
            "radius": radius,
            "x_axis": x_axis,
            "y_axis": y_axis,
        },
    )


def _cylinder_projected_tangent_frame(
    fit: SurfaceFit,
    points: np.ndarray,
    reference_normal: Optional[np.ndarray],
    anchor_override: Optional[np.ndarray] = None,
) -> Optional[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]
]:
    """
    边界点投回柱面后，在投影像中取 anchor，构造「周向 × 轴向」切平面 (e_theta, e_ax)。
    返回
    (projected, anchor, e_theta, e_ax, radius, axis_point, x_axis, y_axis) 或失败时 None。
    """
    axis_point, axis, radius, x_axis, y_axis = _cylinder_build_local_frame(
        fit, points, reference_normal
    )
    r = max(radius, 1e-12)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    projected = np.vstack(
        [
            np.asarray(project_point_to_surface(fit, pts[i]), dtype=np.float64).reshape(3)
            for i in range(pts.shape[0])
        ]
    )
    if anchor_override is not None:
        anchor = np.asarray(
            project_point_to_surface(
                fit, np.asarray(anchor_override, dtype=np.float64).reshape(3)
            ),
            dtype=np.float64,
        ).reshape(3)
    else:
        anchor = np.mean(projected, axis=0)
        anchor = np.asarray(project_point_to_surface(fit, anchor), dtype=np.float64).reshape(
            3
        )
    radial = anchor - axis_point - float(np.dot(anchor - axis_point, axis)) * axis
    ln = float(np.linalg.norm(radial))
    if ln < 1e-12 * max(r, 1.0):
        mean_rp = np.mean(
            projected - axis_point.reshape(1, 3)
            - ((projected - axis_point.reshape(1, 3)) @ axis).reshape(-1, 1)
            * axis.reshape(1, 3),
            axis=0,
        )
        ln = float(np.linalg.norm(mean_rp))
        if ln < 1e-15:
            refn = (
                reference_normal
                if reference_normal is not None
                and float(np.linalg.norm(reference_normal)) > 1e-12
                else x_axis
            )
            radial = np.asarray(refn, dtype=np.float64).reshape(3)
            radial = radial - float(np.dot(radial, axis)) * axis
            ln = float(np.linalg.norm(radial))
        else:
            radial = mean_rp
    # 孔洞可能绕轴对称，「平均」径向抵消；取边界投影像中最大模长径向或解析框架 x_axis
    if ln < 1e-12 * max(r, 1.0):
        best_ln = ln
        best_rv = radial
        for i in range(int(projected.shape[0])):
            rv = projected[i] - axis_point
            rv = rv - float(np.dot(rv, axis)) * axis
            li = float(np.linalg.norm(rv))
            if li > best_ln:
                best_ln = li
                best_rv = rv
        if best_ln > 1e-15:
            radial = best_rv
            ln = best_ln
    if ln < 1e-12 * max(r, 1.0):
        radial = np.asarray(x_axis, dtype=np.float64).reshape(3)
        radial = radial - float(np.dot(radial, axis)) * axis
        ln = float(np.linalg.norm(radial))
    if ln < 1e-15:
        return None
    n_rad = radial / ln
    e_ax = axis
    e_theta = np.cross(e_ax, n_rad)
    lte = float(np.linalg.norm(e_theta))
    if lte < 1e-15:
        return None
    e_theta = e_theta / lte
    n_surf = np.cross(e_theta, e_ax)
    lns = float(np.linalg.norm(n_surf))
    if lns < 1e-15:
        return None
    n_surf = n_surf / lns
    n_surf, e_theta, e_ax = _align_frame_with_reference(
        n_surf, e_theta, e_ax, reference_normal
    )
    return projected, anchor, e_theta, e_ax, radius, axis_point, x_axis, y_axis


def _parameterize_cylinder_tangent(
    fit: SurfaceFit,
    points: np.ndarray,
    reference_normal: Optional[np.ndarray],
) -> Optional[SurfaceParameterization]:
    """
    (Rθ,z) 无可展简单域时：将边界先投回柱面，在平均位置附近用切平面
    (周向×轴向) 做 UV，类似 sphere_tangent；抬升为试探点再 project。
    """
    fr = _cylinder_projected_tangent_frame(fit, points, reference_normal)
    if fr is None:
        return None
    projected, anchor, e_theta, e_ax, radius, axis_point, x_axis, y_axis = fr
    rel = projected - anchor.reshape(1, 3)
    uv = np.column_stack([rel @ e_theta, rel @ e_ax])
    if not _uv_closed_polygon_is_simple(uv):
        return None
    return SurfaceParameterization(
        patch_label=int(fit.patch_label),
        kind="cylinder_tangent",
        uv_boundary_points=uv,
        parameter_data={
            "axis_point": axis_point,
            "axis": e_ax,
            "radius": radius,
            "x_axis": x_axis,
            "y_axis": y_axis,
            "anchor": anchor,
            "u_axis": e_theta,
            "v_axis": e_ax,
        },
    )


def _parameterize_ls_cylinder_aligned(
    fit: SurfaceFit,
    points: np.ndarray,
    reference_normal: Optional[np.ndarray],
) -> Optional[SurfaceParameterization]:
    """
    柱面拟合下的最小二乘工作域：在柱面法向为径向的切平面上展开（周向×轴向），
    避免边界面「绕过轴」时 PCA 法向几乎平行于轴，导致 UV 极度扁平与抬升自交。
    使用 local_plane 抬升（平面点再 project 回柱面），与 sphere 的切平面回退一致。
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = int(pts.shape[0])
    projected = np.vstack(
        [
            np.asarray(project_point_to_surface(fit, pts[i]), dtype=np.float64).reshape(3)
            for i in range(n)
        ]
    )
    anchor_trials: List[Optional[np.ndarray]] = [None]
    if n > 1:
        n_anc = min(n, 24)
        for j in range(n_anc):
            k = int(round(j * float(n - 1) / float(max(n_anc - 1, 1))))
            anchor_trials.append(projected[int(k)])
    for ao in anchor_trials:
        fr = _cylinder_projected_tangent_frame(
            fit, points, reference_normal, anchor_override=ao
        )
        if fr is None:
            continue
        projected2, anchor, e_theta, e_ax, _, _, _, _ = fr
        rel = projected2 - anchor.reshape(1, 3)
        uv = np.column_stack([rel @ e_theta, rel @ e_ax])
        if _uv_closed_polygon_is_simple(uv):
            return SurfaceParameterization(
                patch_label=int(fit.patch_label),
                kind="cylinder_local_plane",
                uv_boundary_points=uv,
                parameter_data={
                    "origin": anchor,
                    "u_axis": e_theta,
                    "v_axis": e_ax,
                    "normal": np.cross(e_theta, e_ax),
                },
            )
    return None


def _cylinder_rtheta_u_span_ok(uv: np.ndarray, radius: float) -> bool:
    """(Rθ,z) 中 u=Rθ 跨度过大时，即使 2D 折线不自交，周向仍容易在 3D 上重叠。"""
    u_coord = np.asarray(uv, dtype=np.float64).reshape(-1, 2)[:, 0]
    span_u = float(np.ptp(u_coord))
    r = max(float(radius), 1e-12)
    # 略大于 2πR 留一点 unwrap 余量；再大则改用笔直切平面回退
    return span_u <= 1.28 * (2.0 * np.pi * r)


def _parameterize_ls_plane_for_lift(
    fit: SurfaceFit,
    points: np.ndarray,
    reference_normal: Optional[np.ndarray],
) -> SurfaceParameterization:
    """
    边界点在最小二乘拟合平面上投影得到简单 UV；抬升仍用 `project_point_to_surface`
    回到原解析面，用于圆柱/圆锥展开自交时的安全回退。

    柱面时先在解析柱面上投影点再做 PCA，并在法向与轴线近平行时改用「径向」法向，
    避免边界沿母线拉长时最小奇异方向落在轴向上导致 UV 严重扭曲与自交。
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n0 = int(pts.shape[0])

    if fit.surface_type == "cylinder" and n0 >= 1:
        pts_ls = np.vstack(
            [
                np.asarray(project_point_to_surface(fit, pts[i]), dtype=np.float64).reshape(
                    3
                )
                for i in range(n0)
            ]
        )
    else:
        pts_ls = pts

    centroid = np.mean(pts_ls, axis=0)
    if pts_ls.shape[0] >= 3:
        centered = pts_ls - centroid.reshape(1, 3)
        if fit.surface_type == "cylinder":
            _ap, axis_raw, _, _, _ = _cylinder_build_local_frame(
                fit, pts, reference_normal
            )
            axis_u = axis_raw / (float(np.linalg.norm(axis_raw)) + 1e-15)
            ax_comp = (centered @ axis_u).reshape(-1, 1) * axis_u.reshape(1, 3)
            tang = centered - ax_comp
            _, _, vh = np.linalg.svd(tang, full_matrices=False)
            normal = np.asarray(vh[-1], dtype=np.float64)
            ln_t = float(np.linalg.norm(normal))
            if ln_t < 1e-12 or abs(float(np.dot(normal / (ln_t + 1e-15), axis_u))) > 0.95:
                _, _, vh2 = np.linalg.svd(centered, full_matrices=False)
                normal = np.asarray(vh2[-1], dtype=np.float64)
        else:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            normal = np.asarray(vh[-1], dtype=np.float64)
    else:
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    ln = float(np.linalg.norm(normal))
    if ln < 1e-12:
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        normal = normal / ln

    if fit.surface_type == "cylinder":
        axis_point, axis_p, _r, x_ax, _y_ax = _cylinder_build_local_frame(
            fit, pts, reference_normal
        )
        axis_u = axis_p / (float(np.linalg.norm(axis_p)) + 1e-15)
        if abs(float(np.dot(normal, axis_u))) > 0.58:
            cen = np.mean(pts_ls, axis=0)
            radial = cen - axis_point - float(np.dot(cen - axis_point, axis_u)) * axis_u
            lr = float(np.linalg.norm(radial))
            if lr > 1e-12 * max(float(fit.surface_params.get("radius", 1.0)), 1e-6):
                normal = radial / lr
            else:
                radial2 = np.cross(axis_u, x_ax)
                if float(np.linalg.norm(radial2)) > 1e-12:
                    normal = radial2 / float(np.linalg.norm(radial2))

    if reference_normal is not None and float(np.linalg.norm(reference_normal)) > 1e-12:
        ref = np.asarray(reference_normal, dtype=np.float64)
        ref = ref / float(np.linalg.norm(ref))
        if float(np.dot(normal, ref)) < 0.0:
            normal = -normal
    u_axis, v_axis = _orthonormal_basis(normal)
    normal, u_axis, v_axis = _align_frame_with_reference(
        normal, u_axis, v_axis, reference_normal
    )
    rel = pts_ls - centroid.reshape(1, 3)
    uv = np.column_stack([rel @ u_axis, rel @ v_axis])
    if (
        fit.surface_type == "cylinder"
        and not _uv_closed_polygon_is_simple(uv)
    ):
        alt = _parameterize_ls_cylinder_aligned(fit, pts, reference_normal)
        if alt is not None:
            return alt
    return SurfaceParameterization(
        patch_label=int(fit.patch_label),
        kind="local_plane",
        uv_boundary_points=uv,
        parameter_data={
            "origin": centroid,
            "u_axis": u_axis,
            "v_axis": v_axis,
            "normal": normal,
        },
    )


def _parameterize_sphere(
    fit: SurfaceFit,
    points: np.ndarray,
    reference_normal: Optional[np.ndarray],
) -> SurfaceParameterization:
    center = np.asarray(fit.surface_params["center"], dtype=np.float64)
    radius = float(fit.surface_params["radius"])
    anchor = np.mean(points, axis=0) - center
    if float(np.linalg.norm(anchor)) < 1e-12:
        anchor = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    normal = anchor / (float(np.linalg.norm(anchor)) + 1e-15)
    u_axis, v_axis = _orthonormal_basis(normal)
    normal, u_axis, v_axis = _align_frame_with_reference(normal, u_axis, v_axis, reference_normal)

    projected = np.vstack([project_point_to_surface(fit, p) for p in points])
    rel = projected - center.reshape(1, 3)
    uv = np.column_stack([rel @ u_axis, rel @ v_axis])
    return SurfaceParameterization(
        patch_label=int(fit.patch_label),
        kind="sphere_tangent",
        uv_boundary_points=uv,
        parameter_data={
            "center": center,
            "radius": radius,
            "origin_dir": normal,
            "u_axis": u_axis,
            "v_axis": v_axis,
        },
    )


def _parameterize_cone(
    fit: SurfaceFit,
    points: np.ndarray,
    reference_normal: Optional[np.ndarray],
) -> SurfaceParameterization:
    apex = np.asarray(fit.surface_params["apex"], dtype=np.float64)
    axis = np.asarray(fit.surface_params["axis"], dtype=np.float64)
    axis = axis / (float(np.linalg.norm(axis)) + 1e-15)
    half_angle = float(fit.surface_params["half_angle"])

    projected = np.vstack([project_point_to_surface(fit, p) for p in points])
    rel = projected - apex.reshape(1, 3)
    axial = rel @ axis
    radial = rel - axial.reshape(-1, 1) * axis.reshape(1, 3)

    seed = radial[0] if len(radial) > 0 else np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if float(np.linalg.norm(seed)) < 1e-12:
        seed = np.cross(axis, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        if float(np.linalg.norm(seed)) < 1e-12:
            seed = np.cross(axis, np.array([0.0, 1.0, 0.0], dtype=np.float64))
    x_axis = seed / (float(np.linalg.norm(seed)) + 1e-15)
    y_axis = np.cross(axis, x_axis)
    y_axis = y_axis / (float(np.linalg.norm(y_axis)) + 1e-15)
    radial_ref = np.mean(radial, axis=0)
    ref_normal = radial_ref if float(np.linalg.norm(radial_ref)) > 1e-12 else reference_normal
    _normal, x_axis, y_axis = _align_frame_with_reference(
        np.cross(x_axis, y_axis), x_axis, y_axis, ref_normal
    )

    theta = np.arctan2(radial @ y_axis, radial @ x_axis)
    theta = _unwrap_angles(theta)
    mean_radius = float(np.mean(np.linalg.norm(radial, axis=1))) if len(radial) else 1.0
    mean_radius = max(mean_radius, 1e-6)
    uv = np.column_stack([mean_radius * theta, axial])
    return SurfaceParameterization(
        patch_label=int(fit.patch_label),
        kind="cone",
        uv_boundary_points=uv,
        parameter_data={
            "apex": apex,
            "axis": axis,
            "half_angle": half_angle,
            "x_axis": x_axis,
            "y_axis": y_axis,
            "mean_radius": mean_radius,
        },
    )


def parameterize_boundary(
    fit: SurfaceFit,
    boundary_points: np.ndarray,
    reference_normal: Optional[np.ndarray] = None,
) -> SurfaceParameterization:
    points = np.asarray(boundary_points, dtype=np.float64)
    if fit.surface_type == "plane":
        pl = _parameterize_plane(fit, points, "plane", reference_normal)
        # 解析平面基底下边界折线可能呈 8 字自交（三重交等），换边界最小二乘切平面常可展开成简单多边形
        if not _uv_closed_polygon_is_simple(pl.uv_boundary_points):
            return _parameterize_ls_plane_for_lift(fit, points, reference_normal)
        return pl
    if fit.surface_type in {TRANSITION_SURFACE_TYPE, "freeform_fallback"}:
        return _parameterize_plane(fit, points, "plane", reference_normal)
    if fit.surface_type == "cylinder":
        rtheta = _try_parameterize_cylinder_rtheta(fit, points, reference_normal)
        if rtheta is not None:
            r_rad = float(rtheta.parameter_data["radius"])
            uv_r = rtheta.uv_boundary_points
            if (
                _cylinder_rtheta_u_span_ok(uv_r, r_rad)
                and _uv_closed_polygon_is_simple(uv_r)
            ):
                return rtheta
        tang = _parameterize_cylinder_tangent(fit, points, reference_normal)
        if tang is not None and _uv_closed_polygon_is_simple(tang.uv_boundary_points):
            return tang
        ls_cyl = _parameterize_ls_cylinder_aligned(fit, points, reference_normal)
        if ls_cyl is not None:
            return ls_cyl
        return _parameterize_ls_plane_for_lift(fit, points, reference_normal)
    if fit.surface_type == "sphere":
        return _parameterize_sphere(fit, points, reference_normal)
    if fit.surface_type == "cone":
        cone = _parameterize_cone(fit, points, reference_normal)
        if not _uv_closed_polygon_is_simple(cone.uv_boundary_points):
            return _parameterize_ls_plane_for_lift(fit, points, reference_normal)
        return cone
    return _parameterize_plane(fit, points, "local_plane", reference_normal)


def lift_parameter_point(
    fit: SurfaceFit,
    parameterization: SurfaceParameterization,
    uv: np.ndarray,
) -> np.ndarray:
    uv = np.asarray(uv, dtype=np.float64)
    kind = parameterization.kind
    data = parameterization.parameter_data

    if kind in {"plane", "local_plane", "cylinder_local_plane"}:
        origin = np.asarray(data["origin"], dtype=np.float64)
        u_axis = np.asarray(data["u_axis"], dtype=np.float64)
        v_axis = np.asarray(data["v_axis"], dtype=np.float64)
        p = origin + uv[0] * u_axis + uv[1] * v_axis
        return project_point_to_surface(fit, p)

    if kind == "cylinder":
        axis_point = np.asarray(data["axis_point"], dtype=np.float64)
        axis = np.asarray(data["axis"], dtype=np.float64)
        radius = float(data["radius"])
        x_axis = np.asarray(data["x_axis"], dtype=np.float64)
        y_axis = np.asarray(data["y_axis"], dtype=np.float64)
        theta = float(uv[0]) / max(radius, 1e-12)
        z = float(uv[1])
        radial = np.cos(theta) * x_axis + np.sin(theta) * y_axis
        p = axis_point + z * axis + radius * radial
        return project_point_to_surface(fit, p)

    if kind == "cylinder_tangent":
        anchor = np.asarray(data["anchor"], dtype=np.float64).reshape(3)
        u_t = np.asarray(data["u_axis"], dtype=np.float64)
        v_t = np.asarray(data["v_axis"], dtype=np.float64)
        p = anchor + float(uv[0]) * u_t + float(uv[1]) * v_t
        return project_point_to_surface(fit, p)

    if kind == "sphere_tangent":
        center = np.asarray(data["center"], dtype=np.float64)
        radius = float(data["radius"])
        origin_dir = np.asarray(data["origin_dir"], dtype=np.float64)
        u_axis = np.asarray(data["u_axis"], dtype=np.float64)
        v_axis = np.asarray(data["v_axis"], dtype=np.float64)
        p = center + radius * origin_dir + uv[0] * u_axis + uv[1] * v_axis
        direction = p - center
        ln = float(np.linalg.norm(direction))
        if ln < 1e-15:
            direction = origin_dir
            ln = 1.0
        p = center + radius * direction / ln
        return project_point_to_surface(fit, p)

    if kind == "cone":
        apex = np.asarray(data["apex"], dtype=np.float64)
        axis = np.asarray(data["axis"], dtype=np.float64)
        half_angle = float(data["half_angle"])
        x_axis = np.asarray(data["x_axis"], dtype=np.float64)
        y_axis = np.asarray(data["y_axis"], dtype=np.float64)
        mean_radius = max(float(data["mean_radius"]), 1e-6)
        theta = float(uv[0]) / mean_radius
        axial = float(uv[1])
        radial_len = abs(axial) * np.tan(half_angle)
        radial = np.cos(theta) * x_axis + np.sin(theta) * y_axis
        p = apex + axial * axis + radial_len * radial
        return project_point_to_surface(fit, p)

    return project_point_to_surface(fit, np.asarray(uv, dtype=np.float64))
