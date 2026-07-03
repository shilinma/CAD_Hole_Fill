#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
孔洞局部结构分析（孔域剖分管线）

整体框架（因果链，非并列 stage 补丁）
====================================

L1  边界符号化 — 孔环邻域聚类、解析拟合、孔边弧与特征点
    输出：``BoundaryArc``、``patch_surface_fits``、``HoleScale``

L2  孔域所有权 — 聚类 K 与补洞 M 一次性定稿（``FillOwnershipSnapshot``）
    特征恢复：``hole_cavity_arrangement`` — Γ_ij = restrict(S_i∩S_j, C)
    证据：退化条带 / 内侧支撑探针 → active vs support
    layout：|M|=1 移除全部交线；|M|>1 仅保留 active–active 交线；降级 active–support 角点

L3  剖分构造 — 孔边弧 + 定稿交线拼子环（``curve_arc_partition`` / ``opening_carrier``）
    命题 1 推论：active 弧被 support 条带隔开时须沿 ``degenerate_label_paths`` 走孔边
    楔形闸门（命题 3/4）→ ``O4_non_wedge``；support 桥缺失 → ``O3_coverage_gap``

L4  曲面实现 — ``run_cad_fill``（本模块之外）
    命题（L4 seam）：孔环顶点为子孔接缝唯一 mesh 索引；禁止投影漂移复制顶点

L5  验证 — ``fill_gate``、``partition_obstacles``（L3 窄腰）；L4 残余环显式失败

拓扑命题（开发契约）
------------------
1. active–support 交线不得作 cell–cell seam → 从 layout 移除。
2. 降级点仍在孔环 ``L`` 上，但不得作 seam 角点。
3. 楔形 𝒲_ℓ：``endpoints(Γ_ℓk)`` 与弧端点/可证虚拟汇交拼环 ⇒ ``∂Ω_ℓ`` 可构造。
4. 端点无法配对且拼环失败 ⇒ 𝒲_ℓ 内无解 → ``O4_non_wedge``。
5. support 嵌套或多链汇交超出单楔形 ⇒ arrangement 或显式失败（非 per-case 绕过）。

L3 障碍 taxonomy
----------------
- ``O1_cycle_open``：子孔环不闭 / active label 缺失
- ``O2_missing_intersection``：layout 交线缺失（退化）
- ``O3_coverage_gap``：support 桥未登记或 L4 覆盖缺口
- ``O4_non_wedge``：命题 4 否定

硬规则（禁止补丁文化）
----------------------
- 每层只实现本层数学角色；禁止用下层容错掩盖上层缺口
- 禁止 label-sorted 假多边形进 fill
- 禁止 L4 residual 静默补带掩盖 L3/L4 失败
- 禁止 per-case 分支（``if case_00xx``）
- 改动须能回答「对应哪条命题推论」；答不出则视为补丁，不应合并
- ``fill_pipeline_stage`` 仅作兼容诊断字段

批处理见 ``hole_analysis_diagnostics``；完整契约见 ``.cursor/skills/cad-hole-fill-research/reference.md``。
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field, replace
from itertools import combinations
from typing import Dict, FrozenSet, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import numpy as np

from .hole_analysis_diagnostics import build_analysis_diagnostics
from .hole_analysis_types import (
    MULTI_PATCH,
    PARTITION_OBSTACLE_O1,
    PARTITION_OBSTACLE_O3,
    PARTITION_OBSTACLE_O4,
    PUBLIC_HOLE_TYPES,
    SINGLE_PATCH,
    AnalysisDiagnostics,
    BoundaryArc,
    FILL_STRATEGY_CURVE_ARC_PARTITION,
    FILL_STRATEGY_OPENING_CARRIER,
    FILL_STRATEGY_WHOLE_LOOP,
    FillOwnershipSnapshot,
    FillPatchClassification,
    FillGateResult,
    FillPlan,
    HoleAnalysis,
    HoleScale,
    HoleType,
    PartitionObstacle,
    PreparedSubhole,
    build_fill_plan,
    infer_fill_strategy,
)
from .hole_detector import HalfEdgeMesh
from .surface_fitting import (
    SurfaceFit,
    _bbox_diag,
    _confidence_from_metrics,
    _evaluate_candidate,
    _fit_plane_ls,
    _patch_support_points,
    _plane_residuals,
    fit_patch_surface,
    fit_patch_surfaces,
    is_analytic_surface_type,
    is_transition_surface_type,
    project_point_to_surface,
)
from .surface_parameterization import SurfaceParameterization, parameterize_boundary
from .hole_patch_triangulation import assess_patch_boundary_readiness, sanitize_closed_ring
from .surface_intersections import (
    AnalyticCurve,
    BoundedCurveSegment,
    IntersectionCurve,
    analytic_intersection,
    feature_curve_sample_count,
    intersect_analytic_curves,
    recover_curve_between_points,
    _point_in_loop_polygon_3d,
)
from .feature_graph import (
    FeatureArrangement,
    PatchCell,
    polygon_area_3d,
    validate_patch_cell,
)


# L3 剖分阶段名（写入 diagnostics["fill_pipeline_stage"]）
FILL_STAGE_OPENING_CARRIER = "L3_opening_carrier_boundary"
FILL_STAGE_CURVE_ARC_PARTITION = "L3_curve_arc_partition"
FILL_STAGE_EXPORT_PREPARED = "L3_export_prepared"
CLOSURE_CURVE_ARC_PARTITION = "curve_arc_partition"

__all__ = [
    "HoleAnalyzer",
    "HoleAnalysis",
    "AnalysisDiagnostics",
    "PreparedSubhole",
    "BoundaryArc",
    "FillPatchClassification",
    "FillGateResult",
    "FillPlan",
    "FillValidationError",
    "HoleType",
    "SINGLE_PATCH",
    "MULTI_PATCH",
    "PUBLIC_HOLE_TYPES",
    "FILL_STRATEGY_WHOLE_LOOP",
    "FILL_STRATEGY_OPENING_CARRIER",
    "FILL_STRATEGY_CURVE_ARC_PARTITION",
    "infer_fill_strategy",
    "build_fill_plan",

]


# ---------------------------------------------------------------------------
# S0 几何感知（邻域聚类、面拟合、孔边弧、特征点）
# ---------------------------------------------------------------------------


def _compute_face_normals_and_areas(
    vertices: np.ndarray, faces: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    normals = np.zeros((len(faces), 3), dtype=np.float64)
    areas = np.zeros(len(faces), dtype=np.float64)
    for fi, tri in enumerate(faces):
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        p0, p1, p2 = vertices[i0], vertices[i1], vertices[i2]
        n = np.cross(p1 - p0, p2 - p0)
        area2 = float(np.linalg.norm(n))
        if area2 > 1e-15:
            normals[fi] = n / area2
            areas[fi] = 0.5 * area2
    return normals, areas


def _build_vertex_to_faces(n_vertices: int, faces: np.ndarray) -> List[List[int]]:
    out: List[List[int]] = [[] for _ in range(n_vertices)]
    for fi, tri in enumerate(faces):
        for vi in tri:
            out[int(vi)].append(fi)
    return out


def _build_face_edge_neighbors(faces: np.ndarray) -> List[Set[int]]:
    neighbors: List[Set[int]] = [set() for _ in range(len(faces))]
    edge_to_faces: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for fi, tri in enumerate(faces):
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            edge_to_faces[key].append(fi)

    for incident in edge_to_faces.values():
        if len(incident) < 2:
            continue
        for fi in incident:
            neighbors[fi].update(fj for fj in incident if fj != fi)
    return neighbors


def _collect_neighborhood_faces(
    loop: Sequence[int],
    vertex_to_faces: Sequence[Sequence[int]],
    face_neighbors: Sequence[Set[int]],
    rings: int,
) -> List[int]:
    seed_faces: Set[int] = set()
    for vi in loop:
        seed_faces.update(vertex_to_faces[int(vi)])

    if rings <= 1:
        return sorted(seed_faces)

    visited = set(seed_faces)
    frontier = deque((fi, 1) for fi in seed_faces)
    while frontier:
        fi, depth = frontier.popleft()
        if depth >= rings:
            continue
        for nb in face_neighbors[fi]:
            if nb in visited:
                continue
            visited.add(nb)
            frontier.append((nb, depth + 1))
    return sorted(visited)


def _build_patch_face_indices(
    face_labels: Mapping[int, int]
) -> Dict[int, List[int]]:
    patch_face_indices: Dict[int, List[int]] = defaultdict(list)
    for fi, label in face_labels.items():
        patch_face_indices[int(label)].append(int(fi))
    return {
        label: sorted(indices)
        for label, indices in sorted(patch_face_indices.items(), key=lambda x: x[0])
    }


def _boundary_seed_faces(
    loop: Sequence[int], vertex_to_faces: Sequence[Sequence[int]]
) -> List[int]:
    seed_faces: Set[int] = set()
    for vi in loop:
        seed_faces.update(int(fi) for fi in vertex_to_faces[int(vi)])
    return sorted(seed_faces)


def _boundary_face_distances(
    face_indices: Sequence[int],
    face_neighbors: Sequence[Set[int]],
    seed_faces: Sequence[int],
) -> Dict[int, int]:
    face_set = set(int(fi) for fi in face_indices)
    dist: Dict[int, int] = {int(fi): 10**9 for fi in face_set}
    frontier = deque()
    for fi in seed_faces:
        if int(fi) not in face_set:
            continue
        dist[int(fi)] = 0
        frontier.append(int(fi))
    while frontier:
        fi = frontier.popleft()
        cur = dist[fi]
        for nb in face_neighbors[fi]:
            nb = int(nb)
            if nb not in face_set or dist[nb] <= cur + 1:
                continue
            dist[nb] = cur + 1
            frontier.append(nb)
    return dist


def _fit_distance_to_point(fit: SurfaceFit, point: np.ndarray) -> float:
    proj = project_point_to_surface(fit, point)
    return float(np.linalg.norm(proj - point))


def _fit_support_diag(fit: SurfaceFit) -> float:
    pts = np.asarray(getattr(fit, "support_points", np.zeros((0, 3))), dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return 1.0
    return _bbox_diag(pts)


def _axis_line_offset(pa: np.ndarray, aa: np.ndarray, pb: np.ndarray, ab: np.ndarray) -> float:
    aa = np.asarray(aa, dtype=np.float64)
    ab = np.asarray(ab, dtype=np.float64)
    la = float(np.linalg.norm(aa))
    lb = float(np.linalg.norm(ab))
    if la < 1e-12 or lb < 1e-12:
        return float("inf")
    aa = aa / la
    ab = ab / lb
    delta = np.asarray(pb, dtype=np.float64) - np.asarray(pa, dtype=np.float64)
    cross = np.cross(aa, ab)
    cross_len = float(np.linalg.norm(cross))
    if cross_len < 1e-8:
        # Same direction: only perpendicular offset matters; axis points may slide along axis.
        perp = delta - float(np.dot(delta, aa)) * aa
        return float(np.linalg.norm(perp))
    return abs(float(np.dot(delta, cross / cross_len)))


def _fit_compatibility_score(fit_a: SurfaceFit, fit_b: SurfaceFit) -> float:
    if is_transition_surface_type(fit_a.surface_type) or is_transition_surface_type(fit_b.surface_type):
        return 0.15 + abs(float(fit_a.fit_score) - float(fit_b.fit_score))
    if fit_a.surface_type != fit_b.surface_type:
        return float("inf")
    if fit_a.surface_type == "plane":
        na = np.asarray(fit_a.surface_params.get("normal"), dtype=np.float64)
        nb = np.asarray(fit_b.surface_params.get("normal"), dtype=np.float64)
        if float(np.linalg.norm(na)) < 1e-12 or float(np.linalg.norm(nb)) < 1e-12:
            return float("inf")
        angle_term = 1.0 - abs(float(np.dot(na, nb)) / (float(np.linalg.norm(na)) * float(np.linalg.norm(nb))))
        pa = np.asarray(fit_a.surface_params.get("point"), dtype=np.float64)
        pb = np.asarray(fit_b.surface_params.get("point"), dtype=np.float64)
        offset_term = _fit_distance_to_point(fit_a, pb) + _fit_distance_to_point(fit_b, pa)
        return angle_term + offset_term
    if fit_a.surface_type == "cylinder":
        aa = np.asarray(fit_a.surface_params.get("axis"), dtype=np.float64)
        ab = np.asarray(fit_b.surface_params.get("axis"), dtype=np.float64)
        if float(np.linalg.norm(aa)) < 1e-12 or float(np.linalg.norm(ab)) < 1e-12:
            return float("inf")
        angle_term = 1.0 - abs(float(np.dot(aa, ab)) / (float(np.linalg.norm(aa)) * float(np.linalg.norm(ab))))
        pa = np.asarray(fit_a.surface_params.get("point"), dtype=np.float64)
        pb = np.asarray(fit_b.surface_params.get("point"), dtype=np.float64)
        ra = float(fit_a.surface_params.get("radius", 0.0))
        rb = float(fit_b.surface_params.get("radius", 0.0))
        scale = max(_fit_support_diag(fit_a), _fit_support_diag(fit_b), abs(ra), abs(rb), 1e-12)
        radius_term = abs(ra - rb) / max(abs(ra), abs(rb), scale, 1e-12)
        offset_term = _axis_line_offset(pa, aa, pb, ab) / scale
        return (
            angle_term
            + 0.75 * radius_term
            + 0.35 * offset_term
            + abs(float(fit_a.fit_score) - float(fit_b.fit_score))
        )
    if fit_a.surface_type == "cone":
        aa = np.asarray(fit_a.surface_params.get("axis"), dtype=np.float64)
        ab = np.asarray(fit_b.surface_params.get("axis"), dtype=np.float64)
        if float(np.linalg.norm(aa)) < 1e-12 or float(np.linalg.norm(ab)) < 1e-12:
            return float("inf")
        angle_term = 1.0 - abs(float(np.dot(aa, ab)) / (float(np.linalg.norm(aa)) * float(np.linalg.norm(ab))))
        apex_a = np.asarray(fit_a.surface_params.get("apex"), dtype=np.float64)
        apex_b = np.asarray(fit_b.surface_params.get("apex"), dtype=np.float64)
        ha = float(fit_a.surface_params.get("half_angle", 0.0))
        hb = float(fit_b.surface_params.get("half_angle", 0.0))
        scale = max(_fit_support_diag(fit_a), _fit_support_diag(fit_b), 1e-12)
        apex_term = float(np.linalg.norm(apex_a - apex_b)) / scale
        half_angle_term = abs(ha - hb) / max(np.deg2rad(10.0), abs(ha), abs(hb), 1e-12)
        return (
            angle_term
            + 0.35 * apex_term
            + 0.5 * half_angle_term
            + abs(float(fit_a.fit_score) - float(fit_b.fit_score))
        )
    if fit_a.surface_type == "sphere":
        ca = np.asarray(fit_a.surface_params.get("center"), dtype=np.float64)
        cb = np.asarray(fit_b.surface_params.get("center"), dtype=np.float64)
        ra = float(fit_a.surface_params.get("radius", 0.0))
        rb = float(fit_b.surface_params.get("radius", 0.0))
        return float(np.linalg.norm(ca - cb)) + abs(ra - rb)
    return abs(float(fit_a.fit_score) - float(fit_b.fit_score))


def _should_merge_patch_fits(fit_a: SurfaceFit, fit_b: SurfaceFit) -> bool:
    if is_transition_surface_type(fit_a.surface_type) or is_transition_surface_type(fit_b.surface_type):
        return False
    if fit_a.surface_type != fit_b.surface_type:
        return False
    compat = _fit_compatibility_score(fit_a, fit_b)
    if fit_a.surface_type == "plane":
        return compat < 0.08
    if fit_a.surface_type == "sphere":
        return compat < 0.12
    if fit_a.surface_type in {"cylinder", "cone"}:
        return compat < 0.08
    return compat < 0.04


def _cluster_faces_by_normal_connectivity(
    face_indices: Sequence[int],
    face_neighbors: Sequence[Set[int]],
    face_normals: np.ndarray,
    max_patches: int,
    normal_angle_deg: float,
    boundary_seed_faces: Optional[Sequence[int]] = None,
) -> Dict[int, int]:
    if not face_indices:
        return {}

    base_cos_thresh = float(np.cos(np.deg2rad(normal_angle_deg)))
    face_set = set(int(fi) for fi in face_indices)
    boundary_dist = _boundary_face_distances(
        face_indices,
        face_neighbors,
        boundary_seed_faces or [],
    )
    raw_labels: Dict[int, int] = {}
    cluster_id = 0

    for seed in face_indices:
        seed = int(seed)
        if seed in raw_labels:
            continue
        raw_labels[seed] = cluster_id
        frontier = deque([seed])
        while frontier:
            fi = frontier.popleft()
            ni = face_normals[fi]
            for nb in face_neighbors[fi]:
                if nb not in face_set or nb in raw_labels:
                    continue
                nj = face_normals[nb]
                local_dist = min(boundary_dist.get(int(fi), 0), boundary_dist.get(int(nb), 0))
                cos_thresh = base_cos_thresh
                if local_dist <= 1:
                    cos_thresh = max(cos_thresh, float(np.cos(np.deg2rad(max(8.0, 0.7 * normal_angle_deg)))))
                elif local_dist >= 3:
                    cos_thresh = min(cos_thresh, float(np.cos(np.deg2rad(1.15 * normal_angle_deg))))
                if float(np.dot(ni, nj)) >= cos_thresh:
                    raw_labels[nb] = cluster_id
                    frontier.append(nb)
        cluster_id += 1

    unique_raw = sorted(set(raw_labels.values()))
    if len(unique_raw) <= max(2 * max_patches, max_patches + 1):
        remap = {old: new for new, old in enumerate(unique_raw)}
        return {fi: remap[label] for fi, label in raw_labels.items()}

    clusters: Dict[int, List[int]] = {label: [] for label in unique_raw}
    for fi, label in raw_labels.items():
        clusters[label].append(fi)

    cluster_normals: Dict[int, np.ndarray] = {}
    cluster_weights: Dict[int, float] = {}
    for label, members in clusters.items():
        n = np.sum(face_normals[np.array(members, dtype=np.int64)], axis=0)
        ln = float(np.linalg.norm(n))
        if ln > 1e-15:
            cluster_normals[label] = n / ln
        else:
            cluster_normals[label] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        cluster_weights[label] = float(len(members))

    keep = sorted(unique_raw, key=lambda x: cluster_weights[x], reverse=True)[:max_patches]
    keep_set = set(keep)
    remap = {old: new for new, old in enumerate(keep)}
    labels: Dict[int, int] = {}
    for fi, raw in raw_labels.items():
        if raw in keep_set:
            labels[fi] = remap[raw]
            continue

        nf = face_normals[fi]
        best_raw = keep[0]
        best_score = -float("inf")
        for cand in keep:
            score = float(np.dot(nf, cluster_normals[cand]))
            if score > best_score:
                best_score = score
                best_raw = cand
        labels[fi] = remap[best_raw]
    return labels


def _refine_patch_partition_with_surface_fit(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_labels: Mapping[int, int],
    max_patches: int,
) -> Tuple[Dict[int, int], Dict[int, List[int]], Dict[int, SurfaceFit]]:
    labels = {int(fi): int(label) for fi, label in face_labels.items()}
    if not labels:
        return {}, {}, {}
    face_neighbors = _build_face_edge_neighbors(faces)

    changed = True
    while changed:
        changed = False
        patch_face_indices = _build_patch_face_indices(labels)
        patch_surface_fits = fit_patch_surfaces(vertices, faces, patch_face_indices)
        ordered_labels = sorted(patch_face_indices)

        for i, src_label in enumerate(ordered_labels):
            src_fit = patch_surface_fits.get(int(src_label))
            if src_fit is None:
                continue
            for dst_label in ordered_labels[i + 1 :]:
                dst_fit = patch_surface_fits.get(int(dst_label))
                if dst_fit is None:
                    continue
                if not _should_merge_patch_fits(src_fit, dst_fit):
                    continue
                merged_faces = sorted(
                    patch_face_indices.get(int(src_label), [])
                    + patch_face_indices.get(int(dst_label), [])
                )
                merged_fit = fit_patch_surfaces(
                    vertices,
                    faces,
                    {int(src_label): merged_faces},
                ).get(int(src_label))
                if merged_fit is None or is_transition_surface_type(merged_fit.surface_type):
                    continue
                for fi, label in list(labels.items()):
                    if int(label) == int(dst_label):
                        labels[int(fi)] = int(src_label)
                changed = True
                break
            if changed:
                break
        if changed:
            continue

        if len(patch_face_indices) <= max_patches:
            break

        ordered = sorted(
            patch_face_indices.items(),
            key=lambda item: (
                len(item[1]),
                patch_surface_fits[item[0]].fit_score if item[0] in patch_surface_fits else float("inf"),
            ),
        )
        src_label = int(ordered[0][0])
        src_fit = patch_surface_fits.get(src_label)
        dst_label = None
        best_score = float("inf")
        for cand_label, _ in ordered[1:]:
            cand_fit = patch_surface_fits.get(int(cand_label))
            if src_fit is None or cand_fit is None:
                continue
            # 禁止跨 surface_type 并 patch：仅当两侧拟合类型一致时才允许合并
            if src_fit.surface_type != cand_fit.surface_type:
                continue
            compat = _fit_compatibility_score(src_fit, cand_fit)
            if (
                is_transition_surface_type(cand_fit.surface_type)
                and src_fit is not None
                and is_analytic_surface_type(src_fit.surface_type)
            ):
                compat += 0.08
            if compat < best_score:
                best_score = compat
                dst_label = int(cand_label)
        if dst_label is None:
            # 无法在不跨类型的情况下减少 patch 数，保留当前划分（可能 > max_patches）
            break
        for fi, label in list(labels.items()):
            if int(label) == src_label:
                labels[int(fi)] = int(dst_label)

        # 再做一轮按拟合距离的局部重分配，让过分割出的面更贴近稳定 patch。
        patch_face_indices = _build_patch_face_indices(labels)
        patch_surface_fits = fit_patch_surfaces(vertices, faces, patch_face_indices)
        reassign_changed = False
        for fi in sorted(labels):
            tri = faces[int(fi)]
            face_center = np.mean(vertices[np.array(tri, dtype=np.int64)], axis=0)
            current_label = int(labels[fi])
            candidate_labels = {current_label}
            for nb in face_neighbors[int(fi)]:
                nb = int(nb)
                if nb in labels:
                    candidate_labels.add(int(labels[nb]))
            if len(candidate_labels) <= 1:
                continue
            current_fit = patch_surface_fits.get(current_label)
            best_label = current_label
            best_score = (
                _fit_distance_to_point(current_fit, face_center) + 0.2 * float(current_fit.fit_score)
                if current_fit is not None
                else float("inf")
            )
            for cand_label in sorted(candidate_labels):
                cand_fit = patch_surface_fits.get(int(cand_label))
                if cand_fit is None:
                    continue
                score = _fit_distance_to_point(cand_fit, face_center) + 0.2 * float(cand_fit.fit_score)
                if (
                    is_transition_surface_type(cand_fit.surface_type)
                    and cand_label != current_label
                ):
                    score += 0.06
                if score + 1e-10 < best_score:
                    best_score = score
                    best_label = int(cand_label)
            if best_label != current_label:
                labels[int(fi)] = best_label
                reassign_changed = True
        if reassign_changed:
            changed = True

    patch_face_indices = _build_patch_face_indices(labels)
    patch_surface_fits = fit_patch_surfaces(vertices, faces, patch_face_indices)
    remap = {old: new for new, old in enumerate(sorted(patch_face_indices))}
    refined_labels = {fi: remap[int(label)] for fi, label in labels.items()}
    refined_patch_face_indices = _build_patch_face_indices(refined_labels)
    refined_patch_surface_fits = fit_patch_surfaces(vertices, faces, refined_patch_face_indices)
    return refined_labels, refined_patch_face_indices, refined_patch_surface_fits


def _patch_support_layers(
    seed_faces: Sequence[int],
    face_neighbors: Sequence[Set[int]],
    *,
    allowed_faces: Optional[Set[int]] = None,
    max_depth: int = 10,
    max_faces: int = 220,
) -> List[List[int]]:
    seed = sorted(set(int(fi) for fi in seed_faces))
    if not seed:
        return []
    visited = set(seed)
    frontier = set(seed)
    layers: List[List[int]] = [seed]
    for _ in range(max(1, int(max_depth))):
        next_frontier: Set[int] = set()
        for fi in frontier:
            for nb in face_neighbors[int(fi)]:
                nb = int(nb)
                if nb in visited:
                    continue
                if allowed_faces is not None and nb not in allowed_faces:
                    continue
                next_frontier.add(nb)
        if not next_frontier:
            break
        ordered = sorted(next_frontier)
        if len(visited) + len(ordered) > max_faces:
            ordered = ordered[: max(0, max_faces - len(visited))]
        if not ordered:
            break
        visited.update(ordered)
        layers.append(sorted(visited))
        frontier = set(ordered)
        if len(visited) >= max_faces:
            break
    return layers


def _semiglobal_fit_objective(
    fit: SurfaceFit,
    *,
    support_size: int,
    seed_size: int,
    depth: int,
) -> float:
    objective = float(fit.fit_score)
    if is_transition_surface_type(fit.surface_type):
        objective += 0.02
    if fit.surface_type == "plane" and depth >= 1:
        objective += 0.0025 * min(depth, 4)
    if is_analytic_surface_type(fit.surface_type) and support_size > seed_size:
        objective -= 0.0015 * min(depth, 4)
    return objective


def _upgrade_patch_surface_fits_semiglobal(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_neighbors: Sequence[Set[int]],
    patch_face_indices: Mapping[int, Sequence[int]],
    patch_surface_fits: Mapping[int, SurfaceFit],
) -> Dict[int, SurfaceFit]:
    upgraded: Dict[int, SurfaceFit] = {}
    for patch_label, seed_faces in patch_face_indices.items():
        seed = sorted(set(int(fi) for fi in seed_faces))
        current = patch_surface_fits.get(int(patch_label))
        if current is None or not seed:
            continue
        best_fit = current
        best_obj = _semiglobal_fit_objective(
            current,
            support_size=len(seed),
            seed_size=len(seed),
            depth=0,
        )
        for depth, support_faces in enumerate(
            _patch_support_layers(seed, face_neighbors),
            start=0,
        ):
            trial_fit = fit_patch_surface(
                int(patch_label),
                vertices,
                faces,
                support_faces,
            )
            trial_obj = _semiglobal_fit_objective(
                trial_fit,
                support_size=len(support_faces),
                seed_size=len(seed),
                depth=depth,
            )
            if trial_obj + 1e-10 < best_obj:
                best_fit = trial_fit
                best_obj = trial_obj
        upgraded[int(patch_label)] = best_fit
    return upgraded


def _relabel_surface_fit(fit: SurfaceFit, patch_label: int) -> SurfaceFit:
    return replace(
        fit,
        patch_label=int(patch_label),
        fit_diagnostics=dict(fit.fit_diagnostics),
    )


def _dominant_face_cluster_on_arc(
    arc: BoundaryArc,
    face_labels: Mapping[int, int],
    boundary_edge_supports: Sequence[Sequence[int]],
) -> Optional[int]:
    """弧段边界边邻接面片中，出现次数最多的初始聚类标签。"""
    counts: Counter = Counter()
    for edge_idx in arc.edge_indices:
        edge_idx = int(edge_idx)
        incident_faces = (
            [int(fi) for fi in boundary_edge_supports[edge_idx]]
            if 0 <= edge_idx < len(boundary_edge_supports)
            else []
        )
        for fi in incident_faces:
            if fi in face_labels:
                counts[int(face_labels[int(fi)])] += 1
    if not counts:
        return None
    return int(counts.most_common(1)[0][0])


def _arc_seed_faces(
    arc: BoundaryArc,
    face_labels: Mapping[int, int],
    boundary_edge_supports: Sequence[Sequence[int]],
) -> List[int]:
    seed: Set[int] = set()
    target_label = (
        int(arc.source_face_patch_label)
        if arc.source_face_patch_label is not None
        else _dominant_face_cluster_on_arc(arc, face_labels, boundary_edge_supports)
    )
    if target_label is None:
        target_label = int(arc.patch_label)
    fallback: Set[int] = set()
    for edge_idx in arc.edge_indices:
        edge_idx = int(edge_idx)
        incident_faces = (
            [int(fi) for fi in boundary_edge_supports[edge_idx]]
            if 0 <= edge_idx < len(boundary_edge_supports)
            else []
        )
        for fi in incident_faces:
            if fi in face_labels and int(face_labels[fi]) == target_label:
                seed.add(int(fi))
            elif fi in face_labels:
                fallback.add(int(fi))
    if seed:
        return sorted(seed)
    if fallback:
        return sorted(fallback)
    return sorted(seed)


def _arc_fit_objective(
    fit: SurfaceFit,
    arc_points: np.ndarray,
    *,
    support_size: int,
    seed_size: int,
    depth: int,
) -> float:
    diag = max(
        float(np.linalg.norm(np.ptp(arc_points, axis=0))) if len(arc_points) else 0.0,
        float(np.linalg.norm(np.ptp(fit.support_points, axis=0))) if len(fit.support_points) else 0.0,
        1e-12,
    )
    arc_residual = 0.0
    if len(arc_points) > 0:
        arc_residual = float(
            np.mean([_fit_distance_to_point(fit, p) for p in np.asarray(arc_points, dtype=np.float64)])
        ) / diag
    objective = float(fit.fit_score) + 0.45 * arc_residual
    if (
        fit.surface_type in {"sphere", "cylinder"}
        and seed_size < 4
        and support_size < max(18, 6 * seed_size)
    ):
        # With only one or two boundary seed faces, a sphere can interpolate a
        # cone patch almost exactly; a short cone strip may also look
        # cylindrical. Prefer a semiglobal support before accepting the local
        # axisymmetric model.
        objective += 0.06
    if is_transition_surface_type(fit.surface_type):
        objective += 0.025
    if is_analytic_surface_type(fit.surface_type) and support_size > seed_size:
        objective -= 0.002 * min(depth, 5)
    return objective


def _fit_boundary_arc_surface(
    patch_label: int,
    arc: BoundaryArc,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_labels: Mapping[int, int],
    boundary_edge_supports: Sequence[Sequence[int]],
    face_neighbors: Sequence[Set[int]],
    allowed_faces: Optional[Set[int]],
    base_fit: Optional[SurfaceFit],
) -> Optional[SurfaceFit]:
    seed_faces = _arc_seed_faces(
        arc,
        face_labels,
        boundary_edge_supports,
    )
    if not seed_faces:
        return _relabel_surface_fit(base_fit, patch_label) if base_fit is not None else None

    eff_allowed = allowed_faces
    if allowed_faces is not None and face_labels is not None:
        src_lbl = (
            int(arc.source_face_patch_label)
            if arc.source_face_patch_label is not None
            else int(arc.patch_label)
        )
        if len(seed_faces) >= 4:
            eff_allowed = {
                int(fi)
                for fi in allowed_faces
                if int(face_labels.get(int(fi), -1)) == src_lbl
            }

    arc_vertices = [int(v) for v in arc.vertex_indices]
    arc_points = vertices[np.array(arc_vertices, dtype=np.int64)]

    if base_fit is not None:
        best_fit = _relabel_surface_fit(base_fit, patch_label)
        best_obj = _arc_fit_objective(
            best_fit,
            arc_points,
            support_size=len(best_fit.support_face_indices),
            seed_size=len(seed_faces),
            depth=0,
        )
    else:
        best_fit = fit_patch_surface(int(patch_label), vertices, faces, seed_faces)
        best_obj = _arc_fit_objective(
            best_fit,
            arc_points,
            support_size=len(seed_faces),
            seed_size=len(seed_faces),
            depth=0,
        )
        if best_fit.surface_type == "sphere" and len(seed_faces) < 4:
            best_obj += 0.06

    for depth, support_faces in enumerate(
        _patch_support_layers(
            seed_faces,
            face_neighbors,
            allowed_faces=eff_allowed,
            max_depth=8,
            max_faces=120,
        ),
        start=0,
    ):
        trial_fit = fit_patch_surface(int(patch_label), vertices, faces, support_faces)
        trial_obj = _arc_fit_objective(
            trial_fit,
            arc_points,
            support_size=len(support_faces),
            seed_size=len(seed_faces),
            depth=depth,
        )
        if trial_obj + 1e-10 < best_obj:
            best_fit = trial_fit
            best_obj = trial_obj

    if (
        base_fit is not None
        and best_fit.surface_type == "sphere"
        and len(seed_faces) < 4
        and base_fit.surface_type in {"cylinder", "cone"}
    ):
        best_fit = _relabel_surface_fit(base_fit, patch_label)

    # 弧段支持域扩张时可能混入少量非共面三角，综合分会偏向圆柱；若点云仍近似落在同一平面上则保留平面。
    if (
        base_fit is not None
        and base_fit.surface_type == "plane"
        and best_fit.surface_type != "plane"
    ):
        vidx, pts = _patch_support_points(vertices, faces, best_fit.support_face_indices)
        if len(pts) >= 3:
            diag = _bbox_diag(pts)
            plane_params, plane_res = _fit_plane_ls(pts)
            residuals = _plane_residuals(pts, plane_params)
            rmse = float(np.sqrt(np.mean(residuals * residuals)))
            if rmse < 0.022 * max(diag, 1e-12):
                score, inlier_ratio, _ = _evaluate_candidate("plane", plane_params, pts, diag)
                conf = _confidence_from_metrics(score, inlier_ratio, len(pts))
                best_fit = SurfaceFit(
                    patch_label=int(best_fit.patch_label),
                    surface_type="plane",
                    surface_params=plane_params,
                    fit_residual=float(plane_res),
                    fit_score=float(score),
                    fit_confidence=conf,
                    support_face_indices=[int(x) for x in best_fit.support_face_indices],
                    support_vertex_indices=vidx,
                    support_points=pts,
                    fit_diagnostics=dict(best_fit.fit_diagnostics),
                )

    best_fit.fit_diagnostics = dict(best_fit.fit_diagnostics)
    best_fit.fit_diagnostics["arc_seed_faces"] = len(seed_faces)
    best_fit.fit_diagnostics["arc_edge_count"] = len(arc.edge_indices)
    return best_fit


def _arc_semiglobal_surface_fits(
    vertices: np.ndarray,
    faces: np.ndarray,
    loop: Sequence[int],
    boundary_half_edges: Sequence[int],
    mesh: HalfEdgeMesh,
    face_labels: Mapping[int, int],
    vertex_to_faces: Sequence[Sequence[int]],
    boundary_edge_supports: Sequence[Sequence[int]],
    face_neighbors: Sequence[Set[int]],
    boundary_arcs: Sequence[BoundaryArc],
    patch_surface_fits: Mapping[int, SurfaceFit],
    allowed_faces: Optional[Set[int]] = None,
) -> Tuple[List[BoundaryArc], Dict[int, SurfaceFit], List[int]]:
    if not boundary_arcs:
        return [], dict(patch_surface_fits), []

    next_label = max((int(label) for label in patch_surface_fits.keys()), default=-1) + 1
    arc_fits: Dict[int, SurfaceFit] = {}
    relabeled_arcs: List[BoundaryArc] = []
    edge_labels = [0] * len(loop)
    for arc in boundary_arcs:
        for edge_idx in arc.edge_indices:
            if 0 <= int(edge_idx) < len(edge_labels):
                edge_labels[int(edge_idx)] = int(arc.patch_label)
    for arc in boundary_arcs:
        new_label = int(next_label)
        next_label += 1
        src_cluster = (
            int(arc.source_face_patch_label)
            if arc.source_face_patch_label is not None
            else _dominant_face_cluster_on_arc(arc, face_labels, boundary_edge_supports)
        )
        fit = _fit_boundary_arc_surface(
            new_label,
            arc,
            vertices,
            faces,
            face_labels,
            boundary_edge_supports,
            face_neighbors,
            allowed_faces,
            patch_surface_fits.get(int(src_cluster)) if src_cluster is not None else None,
        )
        if fit is None:
            relabeled_arcs.append(
                BoundaryArc(
                    patch_label=int(arc.patch_label),
                    edge_indices=[int(x) for x in arc.edge_indices],
                    vertex_indices=[int(x) for x in arc.vertex_indices],
                    source_face_patch_label=src_cluster,
                )
            )
            continue
        arc_fits[new_label] = fit
        src_part = int(src_cluster) if src_cluster is not None else int(arc.patch_label)
        relabeled_arcs.append(
            BoundaryArc(
                patch_label=new_label,
                edge_indices=[int(x) for x in arc.edge_indices],
                vertex_indices=[int(x) for x in arc.vertex_indices],
                source_face_patch_label=src_part,
            )
        )
        for edge_idx in arc.edge_indices:
            edge_labels[int(edge_idx)] = new_label

    merged_fits = dict(patch_surface_fits)
    merged_fits.update(arc_fits)
    edge_labels, relabeled_arcs, merged_fits = _merge_short_arc_transition_pairs(
        vertices,
        faces,
        edge_labels,
        relabeled_arcs,
        merged_fits,
    )
    merged_fits = _prefer_boundary_plane_for_unreliable_axisymmetric_fits(
        vertices,
        relabeled_arcs,
        merged_fits,
    )
    return relabeled_arcs, merged_fits, edge_labels


def _prefer_boundary_plane_for_unreliable_axisymmetric_fits(
    vertices: np.ndarray,
    boundary_arcs: Sequence[BoundaryArc],
    patch_surface_fits: Mapping[int, SurfaceFit],
) -> Dict[int, SurfaceFit]:
    """
    If a cylinder/cone was selected from noisy support faces but the actual
    boundary arc for that label is stably planar, prefer the boundary plane.

    This keeps true curved cases (non-planar boundary arcs or low residual
    axisymmetric fits) unchanged while correcting local support contamination.
    """
    out = dict(patch_surface_fits)
    arcs_by_label: Dict[int, List[BoundaryArc]] = defaultdict(list)
    for arc in boundary_arcs:
        arcs_by_label[int(arc.patch_label)].append(arc)

    for label, arcs_for_label in arcs_by_label.items():
        fit = out.get(int(label))
        if fit is None or fit.surface_type not in {"cylinder", "cone"}:
            continue
        fit_rel = float(fit.fit_residual) / max(float(_fit_support_diag(fit)), 1e-12)
        if fit_rel <= 1e-3:
            continue
        vertex_ids: List[int] = []
        for arc in arcs_for_label:
            vertex_ids.extend(int(v) for v in arc.vertex_indices)
        if len(vertex_ids) < 3:
            continue
        pts = np.asarray(vertices[np.asarray(vertex_ids, dtype=np.int64)], dtype=np.float64)
        centroid = np.mean(pts, axis=0)
        centered = pts - centroid.reshape(1, 3)
        _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
        normal = np.asarray(vh[-1], dtype=np.float64)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1e-12:
            continue
        normal = normal / normal_norm
        residuals = centered @ normal
        plane_res = float(np.sqrt(np.mean(residuals * residuals)))
        diag = max(float(np.linalg.norm(np.ptp(pts, axis=0))), 1e-12)
        plane_rel = plane_res / diag
        if plane_rel > 1e-5:
            continue
        out[int(label)] = replace(
            fit,
            surface_type="plane",
            surface_params={"point": centroid, "normal": normal},
            fit_residual=plane_res,
            fit_score=0.0012 + plane_rel,
            fit_confidence="medium",
            support_points=pts,
            support_vertex_indices=[int(v) for v in vertex_ids],
        )
    return out


def _merge_short_arc_transition_pairs(
    vertices: np.ndarray,
    faces: np.ndarray,
    boundary_edge_labels: Sequence[int],
    boundary_arcs: Sequence[BoundaryArc],
    patch_surface_fits: Mapping[int, SurfaceFit],
) -> Tuple[List[int], List[BoundaryArc], Dict[int, SurfaceFit]]:
    """
    合并被特征点切成两段的短过渡面弧。

    单段短圆柱/圆锥面只有一两个种子三角时常会被局部平面拟合吸收；若两段
    短弧的支持面合并后形成稳定解析过渡面，则它们表示同一 CAD 面。
    """
    labels = [int(x) for x in boundary_edge_labels]
    arcs = list(boundary_arcs)
    fits = dict(patch_surface_fits)
    if len(arcs) < 2:
        return labels, arcs, fits

    parent = {int(arc.patch_label): int(arc.patch_label) for arc in arcs}

    def find(x: int) -> int:
        while parent[int(x)] != int(x):
            parent[int(x)] = parent[parent[int(x)]]
            x = parent[int(x)]
        return int(x)

    def union(a: int, b: int) -> int:
        ra, rb = find(int(a)), find(int(b))
        if ra == rb:
            return ra
        root = min(ra, rb)
        parent[max(ra, rb)] = root
        return root

    merged_fit_by_root: Dict[int, SurfaceFit] = {}
    for i, arc_a in enumerate(arcs):
        fit_a = fits.get(int(arc_a.patch_label))
        if fit_a is None or fit_a.surface_type != "plane":
            continue
        if len(arc_a.edge_indices) > 2 or len(fit_a.support_face_indices) > 6:
            continue
        for arc_b in arcs[i + 1 :]:
            fit_b = fits.get(int(arc_b.patch_label))
            if fit_b is None or fit_b.surface_type != "plane":
                continue
            if len(arc_b.edge_indices) > 2 or len(fit_b.support_face_indices) > 6:
                continue
            if int(arc_a.patch_label) == int(arc_b.patch_label):
                continue
            support_faces = sorted(
                set(int(x) for x in fit_a.support_face_indices)
                | set(int(x) for x in fit_b.support_face_indices)
            )
            if len(support_faces) < 4:
                continue
            merged_fit = fit_patch_surface(
                min(int(arc_a.patch_label), int(arc_b.patch_label)),
                vertices,
                faces,
                support_faces,
            )
            if merged_fit.surface_type not in {"cylinder", "cone"}:
                continue
            diag = max(_fit_support_diag(merged_fit), 1e-12)
            if float(merged_fit.fit_residual) > 1e-4 * diag:
                continue
            if float(merged_fit.fit_score) > 0.006:
                continue
            root = union(int(arc_a.patch_label), int(arc_b.patch_label))
            merged_fit_by_root[int(root)] = _relabel_surface_fit(merged_fit, int(root))

    remap = {label: find(label) for label in parent}
    if all(int(k) == int(v) for k, v in remap.items()):
        return labels, arcs, fits

    out_labels = [int(remap.get(int(label), int(label))) for label in labels]
    out_arcs = [
        BoundaryArc(
            patch_label=int(remap.get(int(arc.patch_label), int(arc.patch_label))),
            edge_indices=[int(x) for x in arc.edge_indices],
            vertex_indices=[int(x) for x in arc.vertex_indices],
            source_face_patch_label=arc.source_face_patch_label,
        )
        for arc in arcs
    ]
    out_fits: Dict[int, SurfaceFit] = {}
    for label, fit in fits.items():
        root = int(remap.get(int(label), int(label)))
        if root in merged_fit_by_root:
            out_fits[root] = merged_fit_by_root[root]
        elif root not in out_fits:
            out_fits[root] = _relabel_surface_fit(fit, root)
    return out_labels, out_arcs, out_fits


def _arc_fit_subset(
    boundary_arcs: Sequence[BoundaryArc],
    patch_surface_fits: Mapping[int, SurfaceFit],
) -> Dict[int, SurfaceFit]:
    used_labels = {int(arc.patch_label) for arc in boundary_arcs}
    return {
        int(label): fit
        for label, fit in patch_surface_fits.items()
        if int(label) in used_labels
    }


def _merge_equivalent_boundary_arc_labels(
    boundary_edge_labels: Sequence[int],
    boundary_arcs: Sequence[BoundaryArc],
    patch_surface_fits: Mapping[int, SurfaceFit],
) -> Tuple[List[int], List[BoundaryArc], Dict[int, SurfaceFit]]:
    labels = [int(x) for x in boundary_edge_labels]
    arc_labels = sorted({int(arc.patch_label) for arc in boundary_arcs})
    arc_by_label = {int(arc.patch_label): arc for arc in boundary_arcs}
    parent = {label: label for label in arc_labels}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        parent[max(ra, rb)] = min(ra, rb)

    for i, src in enumerate(arc_labels):
        fit_a = patch_surface_fits.get(int(src))
        if fit_a is None or not is_analytic_surface_type(fit_a.surface_type):
            continue
        for dst in arc_labels[i + 1 :]:
            fit_b = patch_surface_fits.get(int(dst))
            if fit_b is None or not is_analytic_surface_type(fit_b.surface_type):
                continue
            if fit_a.surface_type != fit_b.surface_type:
                continue
            if _fit_compatibility_score(fit_a, fit_b) < 0.08:
                a_arc = arc_by_label.get(int(src))
                b_arc = arc_by_label.get(int(dst))
                if (
                    a_arc is not None
                    and b_arc is not None
                    and a_arc.source_face_patch_label is not None
                    and b_arc.source_face_patch_label is not None
                    and int(a_arc.source_face_patch_label)
                    != int(b_arc.source_face_patch_label)
                ):
                    continue
                union(int(src), int(dst))

    remap = {label: find(int(label)) for label in arc_labels}
    merged_labels = [remap.get(int(label), int(label)) for label in labels]
    merged_arcs = [
        BoundaryArc(
            patch_label=remap.get(int(arc.patch_label), int(arc.patch_label)),
            edge_indices=[int(x) for x in arc.edge_indices],
            vertex_indices=[int(x) for x in arc.vertex_indices],
            source_face_patch_label=arc.source_face_patch_label,
        )
        for arc in boundary_arcs
    ]
    merged_fits: Dict[int, SurfaceFit] = {}
    for label in arc_labels:
        canonical = remap[int(label)]
        if canonical in merged_fits:
            continue
        fit = patch_surface_fits.get(canonical) or patch_surface_fits.get(int(label))
        if fit is not None:
            merged_fits[int(canonical)] = _relabel_surface_fit(fit, int(canonical))
    return merged_labels, merged_arcs, merged_fits


def _canonical_surface_label_map(
    labels: Iterable[int],
    patch_surface_fits: Mapping[int, SurfaceFit],
) -> Dict[int, int]:
    """把局部 patch/arc label 合并成几何意义上的稳定 surface id。"""
    labs = sorted(set(int(x) for x in labels))
    if not labs:
        return {}
    parent = {label: label for label in labs}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(int(a)), find(int(b))
        if ra == rb:
            return
        parent[max(ra, rb)] = min(ra, rb)

    for i, src in enumerate(labs):
        fit_a = patch_surface_fits.get(int(src))
        if fit_a is None or not is_analytic_surface_type(fit_a.surface_type):
            continue
        for dst in labs[i + 1 :]:
            fit_b = patch_surface_fits.get(int(dst))
            if fit_b is None or not is_analytic_surface_type(fit_b.surface_type):
                continue
            if _should_merge_patch_fits(fit_a, fit_b):
                union(int(src), int(dst))

    canonical = {label: find(label) for label in labs}
    remap = {old: new for new, old in enumerate(sorted(set(canonical.values())))}
    return {label: remap[canonical[label]] for label in labs}


def _relabel_boundary_by_surface_id(
    loop: Sequence[int],
    boundary_edge_labels: Sequence[int],
    boundary_arcs: Sequence[BoundaryArc],
    patch_surface_fits: Mapping[int, SurfaceFit],
) -> Tuple[List[int], List[BoundaryArc], Dict[int, SurfaceFit], Dict[int, int]]:
    labels = sorted(
        set(int(x) for x in boundary_edge_labels)
        | {int(arc.patch_label) for arc in boundary_arcs}
    )
    label_to_surface = _canonical_surface_label_map(labels, patch_surface_fits)
    if not label_to_surface:
        return [int(x) for x in boundary_edge_labels], list(boundary_arcs), dict(patch_surface_fits), {}

    surface_edge_labels = [
        int(label_to_surface.get(int(label), int(label)))
        for label in boundary_edge_labels
    ]
    # 保留特征点切分的弧拓扑，仅重映射 patch_label，不按 surface id 重新切弧。
    surface_arcs = [
        BoundaryArc(
            patch_label=int(label_to_surface.get(int(arc.patch_label), int(arc.patch_label))),
            edge_indices=[int(x) for x in arc.edge_indices],
            vertex_indices=[int(x) for x in arc.vertex_indices],
            source_face_patch_label=arc.source_face_patch_label,
        )
        for arc in boundary_arcs
    ]

    surface_fits: Dict[int, SurfaceFit] = {}
    for old_label, sid in sorted(label_to_surface.items()):
        if sid in surface_fits:
            continue
        fit = patch_surface_fits.get(int(old_label))
        if fit is not None:
            surface_fits[int(sid)] = _relabel_surface_fit(fit, int(sid))
    source_to_surface: Dict[int, int] = {}
    for arc in boundary_arcs:
        if arc.source_face_patch_label is None:
            continue
        source_to_surface[int(arc.source_face_patch_label)] = int(
            label_to_surface.get(int(arc.patch_label), int(arc.patch_label))
        )
    return surface_edge_labels, surface_arcs, surface_fits, source_to_surface


def _build_boundary_half_edge_lookup(mesh: HalfEdgeMesh) -> Dict[Tuple[int, int], int]:
    lookup: Dict[Tuple[int, int], int] = {}
    for he_idx, he in enumerate(mesh.half_edges):
        if he.twin != -1 or he.face < 0:
            continue
        face = mesh.faces[int(he.face)]
        if len(face) < 3:
            continue
        try:
            pos = face.index(int(he.origin))
        except ValueError:
            continue
        tip = int(face[(pos + 1) % len(face)])
        lookup[(int(he.origin), tip)] = he_idx
        lookup.setdefault((tip, int(he.origin)), he_idx)
    return lookup


def _boundary_half_edges_for_loop(mesh: HalfEdgeMesh, loop: Sequence[int]) -> List[int]:
    lookup = _build_boundary_half_edge_lookup(mesh)
    out: List[int] = []
    n = len(loop)
    for i in range(n):
        u = int(loop[i])
        v = int(loop[(i + 1) % n])
        out.append(lookup.get((u, v), -1))
    return out


def _boundary_edge_incident_faces(
    loop: Sequence[int],
    boundary_half_edges: Sequence[int],
    edge_idx: int,
    mesh: HalfEdgeMesh,
    vertex_to_faces: Sequence[Sequence[int]],
) -> List[int]:
    n = len(loop)
    if n == 0:
        return []
    edge_idx = int(edge_idx)
    out: Set[int] = set()
    he_idx = int(boundary_half_edges[edge_idx]) if edge_idx < len(boundary_half_edges) else -1
    if he_idx >= 0:
        face_idx = int(mesh.half_edges[he_idx].face)
        if face_idx >= 0:
            out.add(face_idx)

    u = int(loop[edge_idx % n])
    v = int(loop[(edge_idx + 1) % n])
    shared = set(int(fi) for fi in vertex_to_faces[u]).intersection(
        int(fi) for fi in vertex_to_faces[v]
    )
    for fi in shared:
        out.add(int(fi))
    return sorted(out)


def _boundary_edge_incident_face_supports(
    loop: Sequence[int],
    boundary_half_edges: Sequence[int],
    mesh: HalfEdgeMesh,
    vertex_to_faces: Sequence[Sequence[int]],
) -> List[List[int]]:
    return [
        _boundary_edge_incident_faces(
            loop,
            boundary_half_edges,
            edge_idx,
            mesh,
            vertex_to_faces,
        )
        for edge_idx in range(len(loop))
    ]


def _flatten_boundary_edge_supports(
    boundary_edge_supports: Sequence[Sequence[int]],
) -> List[int]:
    out: Set[int] = set()
    for faces_for_edge in boundary_edge_supports:
        out.update(int(fi) for fi in faces_for_edge)
    return sorted(out)


def _boundary_vertex_patch_labels(
    loop: Sequence[int], boundary_edge_labels: Sequence[int]
) -> Dict[int, List[int]]:
    n = len(loop)
    out: Dict[int, List[int]] = {}
    for i, vi in enumerate(loop):
        labels = sorted(
            {
                int(boundary_edge_labels[(i - 1) % n]),
                int(boundary_edge_labels[i]),
            }
        )
        out[int(vi)] = labels
    return out


def _surface_face_labels_from_fit_distance(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_labels: Mapping[int, int],
    cluster_surface_fits: Mapping[int, SurfaceFit],
    surface_fits: Mapping[int, SurfaceFit],
    source_to_surface_id: Mapping[int, int],
) -> Dict[int, int]:
    cluster_hint: Dict[int, int] = {
        int(src): int(dst)
        for src, dst in source_to_surface_id.items()
        if int(dst) in surface_fits
    }
    out: Dict[int, int] = {}
    for fi, label in face_labels.items():
        tri = faces[int(fi)]
        center = np.mean(vertices[np.asarray(tri, dtype=np.int64)], axis=0)
        cluster_fit = cluster_surface_fits.get(int(label))
        best_sid: Optional[int] = None
        best_score = float("inf")
        for sid, surface_fit in surface_fits.items():
            score = _fit_distance_to_point(surface_fit, center)
            score += 0.10 * float(surface_fit.fit_score)
            if cluster_fit is not None:
                if cluster_fit.surface_type == surface_fit.surface_type:
                    score += 0.05 * _fit_compatibility_score(cluster_fit, surface_fit)
                elif not is_transition_surface_type(cluster_fit.surface_type):
                    score += 0.25
            if int(cluster_hint.get(int(label), -1)) == int(sid):
                score -= 0.03
            if score < best_score:
                best_score = score
                best_sid = int(sid)
        if best_sid is not None:
            out[int(fi)] = best_sid
    return out


def _augment_boundary_vertex_labels_from_surface_faces(
    base_labels: Mapping[int, Sequence[int]],
    loop: Sequence[int],
    vertex_to_faces: Sequence[Sequence[int]],
    surface_face_labels: Mapping[int, int],
) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {
        int(v): sorted(set(int(x) for x in labels))
        for v, labels in base_labels.items()
    }
    for vi in loop:
        labels = set(out.get(int(vi), []))
        if len(labels) >= 2:
            continue
        for fi in vertex_to_faces[int(vi)]:
            sid = surface_face_labels.get(int(fi))
            if sid is not None:
                labels.add(int(sid))
        if labels:
            out[int(vi)] = sorted(labels)
    return out


def _force_boundary_arc_seed_surface_labels(
    surface_face_labels: MutableMapping[int, int],
    boundary_edge_supports: Sequence[Sequence[int]],
    boundary_arcs: Sequence[BoundaryArc],
) -> None:
    if not boundary_arcs:
        return
    for arc in boundary_arcs:
        sid = int(arc.patch_label)
        for edge_idx in arc.edge_indices:
            ei = int(edge_idx)
            if not (0 <= ei < len(boundary_edge_supports)):
                continue
            for fi in boundary_edge_supports[ei]:
                surface_face_labels[int(fi)] = sid


def _interior_face_on_undirected_edge(
    vertex_to_faces: List[List[int]],
    faces: np.ndarray,
    u: int,
    v: int,
) -> Optional[int]:
    """返回包含无向边 ``(u,v)`` 的mesh三角面索引（邻域中首个即可）。"""
    uu, vv = int(u), int(v)
    for fi in vertex_to_faces[uu]:
        a, b, c = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
        if vv in (a, b, c):
            return int(fi)
    return None


def _triangle_unit_normal(
    vertices: np.ndarray,
    faces: np.ndarray,
    fi: int,
) -> np.ndarray:
    tri = faces[int(fi)]
    p0 = vertices[int(tri[0])]
    p1 = vertices[int(tri[1])]
    p2 = vertices[int(tri[2])]
    cr = np.cross(p1 - p0, p2 - p0)
    ln = float(np.linalg.norm(cr))
    if ln < 1e-15:
        return np.zeros(3, dtype=np.float64)
    return cr / ln


def _feature_point_candidates(
    vertices: np.ndarray,
    faces: np.ndarray,
    loop: Sequence[int],
    vertex_to_faces: List[List[int]],
    feature_point_normal_deg: float,
) -> List[int]:
    """
    孔边界特征点候选：仅依据法向折角，不依赖 patch 标签突变。

    对环上每个顶点，取沿孔洞环相邻两条边界边各自一侧的内部三角面；若其法向夹角
    不小于 ``feature_point_normal_deg``，则记为特征点（环上索引）。
    """
    n = len(loop)
    thr = float(feature_point_normal_deg)
    out: List[int] = []
    for i in range(n):
        v_prev = int(loop[(i - 1) % n])
        v_cur = int(loop[i])
        v_next = int(loop[(i + 1) % n])
        fa = _interior_face_on_undirected_edge(
            vertex_to_faces, faces, v_prev, v_cur
        )
        fb = _interior_face_on_undirected_edge(
            vertex_to_faces, faces, v_cur, v_next
        )
        if fa is None or fb is None:
            continue
        if int(fa) == int(fb):
            continue

        na = _triangle_unit_normal(vertices, faces, int(fa))
        nb = _triangle_unit_normal(vertices, faces, int(fb))
        cos_n = float(np.clip(np.dot(na, nb), -1.0, 1.0))
        ang_n = float(np.rad2deg(np.arccos(cos_n)))
        if ang_n >= thr:
            out.append(i)

    return sorted(set(out))


def _feature_edge_candidates(
    loop: Sequence[int],
    feature_point_positions: Sequence[int],
) -> List[Tuple[int, int]]:
    """特征点之间的孔边段（用于特征边候选）。"""
    feature_point_set = set(int(x) for x in feature_point_positions)
    n = len(loop)
    out: List[Tuple[int, int]] = []
    for i in range(n):
        if i in feature_point_set or ((i + 1) % n) in feature_point_set:
            out.append((int(loop[i]), int(loop[(i + 1) % n])))
    return out


def _loop_edge_indices_between(
    start_pos: int,
    end_pos: int,
    n: int,
) -> List[int]:
    """孔环上从 ``start_pos`` 沿正向走到 ``end_pos`` 顶点之前的边下标列（不含离开终点的边）。"""
    if n <= 0:
        return []
    start_pos = int(start_pos) % n
    end_pos = int(end_pos) % n
    if start_pos == end_pos:
        return list(range(n))
    edge_indices: List[int] = []
    pos = start_pos
    while int(pos) != int(end_pos):
        edge_indices.append(int(pos))
        pos = (pos + 1) % n
        if len(edge_indices) > n:
            break
    return edge_indices


def _extract_boundary_arcs_from_feature_points(
    loop: Sequence[int],
    feature_point_positions: Sequence[int],
) -> List[BoundaryArc]:
    """
    按孔环特征点切分边界弧；弧端点均为特征点（环上顶点）。

    无特征点或仅 1 个特征点时退化为整环单弧。
    """
    n = len(loop)
    if n == 0:
        return []
    fp_positions = sorted({int(i) % n for i in feature_point_positions})
    if len(fp_positions) <= 1:
        return [
            BoundaryArc(
                patch_label=0,
                edge_indices=list(range(n)),
                vertex_indices=[int(v) for v in loop] + [int(loop[0])],
                source_face_patch_label=None,
            )
        ]

    arcs: List[BoundaryArc] = []
    for arc_idx, start_pos in enumerate(fp_positions):
        end_pos = int(fp_positions[(arc_idx + 1) % len(fp_positions)])
        edge_indices = _loop_edge_indices_between(start_pos, end_pos, n)
        if not edge_indices:
            continue
        vertex_indices = [int(loop[edge_indices[0]])]
        for edge_idx in edge_indices:
            vertex_indices.append(int(loop[(int(edge_idx) + 1) % n]))
        arcs.append(
            BoundaryArc(
                patch_label=int(arc_idx),
                edge_indices=edge_indices,
                vertex_indices=vertex_indices,
                source_face_patch_label=None,
            )
        )
    return arcs


def _boundary_edge_labels_from_arcs(
    loop: Sequence[int],
    arcs: Sequence[BoundaryArc],
) -> List[int]:
    """由现有 ``BoundaryArc`` 还原孔环每条边的 patch label。"""
    n = len(loop)
    labels = [0] * n
    for arc in arcs:
        for edge_idx in arc.edge_indices:
            if 0 <= int(edge_idx) < n:
                labels[int(edge_idx)] = int(arc.patch_label)
    return labels


def _effective_feature_vertices_after_endpoint_remap(
    feature_vertex_ids: Sequence[int],
    endpoint_remap: Mapping[int, int],
) -> List[int]:
    """layout 端点 refinement 后，用 P 替换 L，未改动的特征点保留。"""
    seen: Set[int] = set()
    out: List[int] = []
    for raw in feature_vertex_ids:
        vid = int(endpoint_remap.get(int(raw), int(raw)))
        if vid in seen:
            continue
        seen.add(vid)
        out.append(vid)
    return sorted(out)


def _loop_positions_for_vertices(
    loop: Sequence[int],
    vertex_ids: Sequence[int],
) -> List[int]:
    """孔环顶点 id → 环上位置（去重排序）。"""
    index_by_vid = {int(v): int(i) for i, v in enumerate(loop)}
    positions: List[int] = []
    for vid in vertex_ids:
        pos = index_by_vid.get(int(vid))
        if pos is None:
            continue
        positions.append(int(pos))
    return sorted(set(positions))


def _extract_boundary_arcs_from_labeled_feature_points(
    loop: Sequence[int],
    feature_point_positions: Sequence[int],
    boundary_edge_labels: Sequence[int],
) -> List[BoundaryArc]:
    """
    按有效特征点（含 refinement 后的 P）重切孔边弧；``patch_label`` 取自 ``boundary_edge_labels``。
    """
    n = len(loop)
    if n == 0:
        return []
    edge_labels = [int(x) for x in boundary_edge_labels]
    if len(edge_labels) != n:
        edge_labels = (edge_labels + [0] * n)[:n]

    fp_positions = sorted({int(i) % n for i in feature_point_positions})
    if len(fp_positions) <= 1:
        patch_label = int(edge_labels[0]) if edge_labels else 0
        return [
            BoundaryArc(
                patch_label=patch_label,
                edge_indices=list(range(n)),
                vertex_indices=[int(v) for v in loop] + [int(loop[0])],
                source_face_patch_label=None,
            )
        ]

    arcs: List[BoundaryArc] = []
    for arc_idx, start_pos in enumerate(fp_positions):
        end_pos = int(fp_positions[(arc_idx + 1) % len(fp_positions)])
        edge_indices = _loop_edge_indices_between(start_pos, end_pos, n)
        if not edge_indices:
                continue
        arc_edge_labels = [int(edge_labels[int(ei)]) for ei in edge_indices]
        patch_label = int(Counter(arc_edge_labels).most_common(1)[0][0])
        vertex_indices = [int(loop[edge_indices[0]])]
        for edge_idx in edge_indices:
            vertex_indices.append(int(loop[(int(edge_idx) + 1) % n]))
        arcs.append(
            BoundaryArc(
                patch_label=patch_label,
                edge_indices=edge_indices,
                vertex_indices=vertex_indices,
                source_face_patch_label=None,
            )
        )
    return arcs


def _infer_public_hole_type(unique_patch_count: int) -> HoleType:
    """对外报告用：仅区分单 patch / 多 patch。"""
    return SINGLE_PATCH if unique_patch_count <= 1 else MULTI_PATCH


def _curves_have_virtual_bridge(curves: Sequence[IntersectionCurve]) -> bool:
    """是否存在两端均为虚拟端点的桥接特征线（替代 template 特判清 junction）。"""
    return any(
        int(c.endpoint_vertex_indices[0]) < 0
        and int(c.endpoint_vertex_indices[1]) < 0
        for c in curves
    )


def _coerce_junction_for_analysis(
    curves: Sequence[IntersectionCurve],
    junction_point: Optional[np.ndarray],
    junction_confidence: str,
) -> Tuple[Optional[np.ndarray], str]:
    """分析出口：存在虚拟 bridge 时不保留汇交点。"""
    if _curves_have_virtual_bridge(curves):
        return None, "none"
    return junction_point, junction_confidence


def _junction_from_curve_virtual_endpoints(
    curves: Sequence[IntersectionCurve],
) -> Optional[np.ndarray]:
    """从已恢复的虚拟端点估计汇交点（孔内半射线出口），避免全局远枝误估。"""
    points: List[np.ndarray] = []
    for curve in curves:
        e0, e1 = (
            int(curve.endpoint_vertex_indices[0]),
            int(curve.endpoint_vertex_indices[1]),
        )
        pts = np.asarray(curve.curve_points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] == 0:
            continue
        if e0 < 0:
            points.append(np.asarray(pts[0], dtype=np.float64).reshape(3))
        if e1 < 0:
            points.append(np.asarray(pts[-1], dtype=np.float64).reshape(3))
    if not points:
        return None
    return np.mean(np.vstack(points), axis=0)


def _effective_boundary_vertex_labels(
    boundary_vertex_labels: Mapping[int, Sequence[int]],
    patch_surface_fits: Mapping[int, SurfaceFit],
) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for vertex, labels in boundary_vertex_labels.items():
        analytic = [
            int(label)
            for label in labels
            if (
                fit := patch_surface_fits.get(int(label))
            ) is not None and is_analytic_surface_type(fit.surface_type)
        ]
        if len(set(analytic)) >= 2:
            out[int(vertex)] = sorted(set(analytic))
        else:
            out[int(vertex)] = sorted(set(int(x) for x in labels))
    return out


def _closed_loop_edges(n_vertices: int) -> np.ndarray:
    return np.array([[i, (i + 1) % n_vertices] for i in range(n_vertices)], dtype=np.int64)


def _open_chain_edges(n_vertices: int) -> np.ndarray:
    if n_vertices < 2:
        return np.zeros((0, 2), dtype=np.int64)
    return np.array([[i, i + 1] for i in range(n_vertices - 1)], dtype=np.int64)


def _loop_centroid(vertices: np.ndarray, loop: Sequence[int]) -> np.ndarray:
    return np.mean(vertices[np.array(loop, dtype=np.int64)], axis=0)


def _polyline_length(pts: np.ndarray) -> float:
    """折线各段长度和。"""
    p = np.asarray(pts, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] < 2 or p.shape[1] < 3:
        return 0.0
    return float(np.sum(np.linalg.norm(p[1:] - p[:-1], axis=1)))


def _feature_curve_guide_point(
    start: np.ndarray,
    end: np.ndarray,
    fallback_center: Optional[np.ndarray] = None,
    *,
    endpoint_vertex_indices: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    交线恢复的参考点。

    角点→虚拟汇交段用孔心，保证沿解析交线朝孔内延伸；
    角点↔角点段用端点中点，避免跨棱边孔被孔心拉偏。
    """
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    if fallback_center is not None and endpoint_vertex_indices is not None:
        v0, v1 = (int(endpoint_vertex_indices[0]), int(endpoint_vertex_indices[1]))
        if (v0 >= 0 and v1 < 0) or (v1 >= 0 and v0 < 0):
            return np.asarray(fallback_center, dtype=np.float64).reshape(3)
        if v0 < 0 and v1 < 0:
            return np.asarray(fallback_center, dtype=np.float64).reshape(3)
    mid = 0.5 * (start + end)
    if fallback_center is None:
        return mid
    center = np.asarray(fallback_center, dtype=np.float64)
    return 0.85 * mid + 0.15 * center


def _polyline_target_spacing(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    seg = np.linalg.norm(points[1:] - points[:-1], axis=1)
    positive = seg[seg > 1e-12]
    if positive.size == 0:
        return 0.0
    return float(np.median(positive))


def _parameterize_subhole(
    patch_label: int,
    closed_points: np.ndarray,
    patch_surface_fits: Mapping[int, SurfaceFit],
) -> SurfaceParameterization:
    """
    对子孔闭合折线做一次边界参数化，返回可供 `lift_parameter_point` 使用的对象。
    """
    fit = patch_surface_fits.get(int(patch_label))
    if fit is None:
        centroid = np.mean(closed_points, axis=0)
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        fit = SurfaceFit(
            patch_label=int(patch_label),
            surface_type="freeform_fallback",
            surface_params={"anchor": centroid, "point": centroid, "normal": normal},
            fit_residual=float("inf"),
            fit_score=float("inf"),
            fit_confidence="low",
            support_face_indices=[],
            support_vertex_indices=[],
            support_points=closed_points,
        )
    ref_normal = _estimate_subhole_reference_normal(
        patch_label,
        closed_points,
        patch_surface_fits,
    )
    return parameterize_boundary(fit, closed_points, reference_normal=ref_normal)


def _estimate_subhole_reference_normal(
    patch_label: int,
    closed_points: np.ndarray,
    patch_surface_fits: Mapping[int, SurfaceFit],
) -> np.ndarray:
    fit = patch_surface_fits.get(int(patch_label))
    if fit is None or len(closed_points) == 0:
        centered = closed_points - np.mean(closed_points, axis=0, keepdims=True)
        if centered.shape[0] >= 3:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            return np.asarray(vh[-1], dtype=np.float64)
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)

    if fit.surface_type == "plane":
        normal = np.asarray(fit.surface_params.get("normal", np.zeros(3)), dtype=np.float64)
        ln = float(np.linalg.norm(normal))
        if ln > 1e-12:
            return normal / ln

    if fit.surface_type == "sphere":
        center = np.asarray(fit.surface_params.get("center", np.zeros(3)), dtype=np.float64)
        normal = np.mean(closed_points - center.reshape(1, 3), axis=0)
        ln = float(np.linalg.norm(normal))
        if ln > 1e-12:
            return normal / ln

    if fit.surface_type in {"cylinder", "cone"}:
        axis = np.asarray(fit.surface_params.get("axis", np.zeros(3)), dtype=np.float64)
        axis_ln = float(np.linalg.norm(axis))
        if axis_ln > 1e-12:
            axis = axis / axis_ln
            origin_key = "point" if fit.surface_type == "cylinder" else "apex"
            origin = np.asarray(fit.surface_params.get(origin_key, np.zeros(3)), dtype=np.float64)
            rel = closed_points - origin.reshape(1, 3)
            axial = rel @ axis
            radial = rel - axial.reshape(-1, 1) * axis.reshape(1, 3)
            normal = np.mean(radial, axis=0)
            ln = float(np.linalg.norm(normal))
            if ln > 1e-12:
                return normal / ln

    centered = closed_points - np.mean(closed_points, axis=0, keepdims=True)
    if centered.shape[0] >= 3:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = np.asarray(vh[-1], dtype=np.float64)
        ln = float(np.linalg.norm(normal))
        if ln > 1e-12:
            return normal / ln
    return np.array([0.0, 0.0, 1.0], dtype=np.float64)


def _arc_chord_monotone_parameterization(
    fit: SurfaceFit,
    closed_points: np.ndarray,
    boundary_sources: Sequence[int],
    reference_normal: np.ndarray,
) -> Optional[SurfaceParameterization]:
    """
    Build a simple 2D domain for one boundary arc closed by one mesh-mesh chord.

    Some cylinder/local-plane projections place the closing chord through the
    projected boundary arc although the 3D topological cell is valid. For L4
    triangulation, the required invariant is a simple ordered work domain; the
    boundary xyz remains authoritative.
    """
    pts = np.asarray(closed_points, dtype=np.float64)
    n = int(pts.shape[0])
    if n < 4:
        return None
    sources = [int(s) for s in boundary_sources]
    if len(sources) != n or any(s < 0 for s in sources):
        return None

    chord = pts[-1] - pts[0]
    chord_len = float(np.linalg.norm(chord))
    if chord_len <= 1e-12:
        return None
    u_axis = chord / chord_len

    normal = np.asarray(reference_normal, dtype=np.float64).reshape(3)
    nrm = float(np.linalg.norm(normal))
    if nrm <= 1e-12:
        centered = pts - np.mean(pts, axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = np.asarray(vh[-1], dtype=np.float64)
        nrm = float(np.linalg.norm(normal))
    if nrm <= 1e-12:
        return None
    normal = normal / nrm
    v_axis = np.cross(normal, u_axis)
    v_len = float(np.linalg.norm(v_axis))
    if v_len <= 1e-12:
        return None
    v_axis = v_axis / v_len

    rel = pts - pts[0].reshape(1, 3)
    x = rel @ u_axis
    y = rel @ v_axis
    sign = 1.0
    if n > 2 and float(np.mean(y[1:-1])) < 0.0:
        sign = -1.0
    y = sign * y
    y[0] = 0.0
    y[-1] = 0.0
    min_height = max(1e-6 * chord_len, 1e-8)
    for i in range(1, n - 1):
        # Preserve relative curvature when projection is useful, but keep the
        # arc strictly on one side of the closing chord.
        if y[i] <= min_height:
            t = max(0.0, min(1.0, x[i] / chord_len))
            y[i] = min_height * (1.0 + np.sin(np.pi * t))
    uv = np.column_stack([x, y])
    if not bool(assess_patch_boundary_readiness(pts, uv).get("ready", False)):
        return None
    return SurfaceParameterization(
        patch_label=int(fit.patch_label),
        kind="local_plane",
        uv_boundary_points=uv,
        parameter_data={
            "origin": pts[0],
            "u_axis": u_axis,
            "v_axis": v_axis,
        },
    )


# ---------------------------------------------------------------------------
# S7 子孔导出（PreparedSubhole 构造）
# ---------------------------------------------------------------------------


def _make_prepared_subhole(
    *,
    patch_label: int,
    boundary_vertex_indices: Sequence[int],
    boundary_points: np.ndarray,
    closed_points: np.ndarray,
    boundary_sources: Sequence[int],
    patch_surface_fits: Mapping[int, SurfaceFit],
    closure_kind: str,
    feature_point_vertex_indices: Tuple[int, int],
    open_as_closed_loop: bool = False,
) -> PreparedSubhole:
    """由闭合折线统一构造 PreparedSubhole（参数化 + 参考法向）。"""
    closed_points = np.asarray(closed_points, dtype=np.float64)
    boundary_points = np.asarray(boundary_points, dtype=np.float64)
    sources_list = [int(s) for s in boundary_sources]
    n_closed = int(closed_points.shape[0])
    if (
        open_as_closed_loop
        and n_closed >= 2
        and int(sources_list[0]) == int(sources_list[-1])
        and float(np.linalg.norm(closed_points[0] - closed_points[-1])) <= 1e-9
    ):
        # 显式闭合环首尾同点：去掉重复闭合点，避免三角化/合并时退化三角形。
        closed_points = closed_points[:-1]
        if int(boundary_points.shape[0]) == n_closed:
            boundary_points = boundary_points[:-1]
        sources_list = sources_list[:-1]
        n_closed = int(closed_points.shape[0])
    closed_points, sources_list, _removed_dup = sanitize_closed_ring(
        closed_points,
        sources_list,
    )
    n_closed = int(closed_points.shape[0])
    if int(boundary_points.shape[0]) != n_closed:
        boundary_points = np.asarray(closed_points, dtype=np.float64)
    boundary_vertex_indices = [int(s) for s in sources_list if int(s) >= 0]
    param = _parameterize_subhole(int(patch_label), closed_points, patch_surface_fits)
    reference_normal = _estimate_subhole_reference_normal(
        int(patch_label), closed_points, patch_surface_fits
    )
    if str(closure_kind) == CLOSURE_CURVE_ARC_PARTITION and not bool(
        assess_patch_boundary_readiness(
            closed_points,
            np.asarray(param.uv_boundary_points, dtype=np.float64),
        ).get("ready", False)
    ):
        fit = patch_surface_fits.get(int(patch_label))
        if fit is not None:
            fallback_param = _arc_chord_monotone_parameterization(
                fit,
                closed_points,
                sources_list,
                reference_normal,
            )
            if fallback_param is not None:
                param = fallback_param
    n_open = len(boundary_vertex_indices)
    if open_as_closed_loop:
        open_edges = _closed_loop_edges(n_closed)
    elif n_open >= 2:
        open_edges = _open_chain_edges(n_open)
    else:
        open_edges = _closed_loop_edges(n_closed)
    return PreparedSubhole(
        patch_label=int(patch_label),
        boundary_vertex_indices=[int(v) for v in boundary_vertex_indices],
        boundary_points=boundary_points,
        open_boundary_edges=open_edges,
        closed_boundary_points=closed_points,
        closed_boundary_edges=_closed_loop_edges(n_closed),
        boundary_points_2d=param.uv_boundary_points,
        parameterization_kind=param.kind,
        parameterization=param,
        boundary_sources=sources_list,
        closure_kind=str(closure_kind),
        feature_point_vertex_indices=(
            int(feature_point_vertex_indices[0]),
            int(feature_point_vertex_indices[1]),
        ),
        reference_normal=reference_normal,
    )


def _opening_carrier_feature_endpoints(
    loop_vertices: Sequence[int],
    fill_classification: FillPatchClassification,
) -> Tuple[int, int]:
    inactive = {int(v) for v in fill_classification.inactive_feature_points}
    active_fp = [int(v) for v in loop_vertices if int(v) not in inactive]
    if len(active_fp) >= 2:
        return int(active_fp[0]), int(active_fp[-1])
    if active_fp:
        v = int(active_fp[0])
        return v, v
    v0 = int(loop_vertices[0])
    return v0, v0


def _prepare_opening_carrier_subholes(
    vertices: np.ndarray,
    loop: Sequence[int],
    fill_classification: FillPatchClassification,
    patch_surface_fits: Mapping[int, SurfaceFit],
) -> List[PreparedSubhole]:
    """S1b：单开孔承载面用完整孔环补洞，邻接条带弧段保留在边界上。"""
    active_labels = sorted(int(x) for x in fill_classification.active_fill_labels)
    if len(active_labels) != 1:
        return []
    carrier_label = int(active_labels[0])
    loop_vertices = [int(v) for v in loop]
    pts = vertices[np.asarray(loop_vertices, dtype=np.int64)]
    feature_endpoints = _opening_carrier_feature_endpoints(
        loop_vertices,
        fill_classification,
    )
    return [
        _make_prepared_subhole(
            patch_label=carrier_label,
            boundary_vertex_indices=loop_vertices,
            boundary_points=pts,
            closed_points=pts.copy(),
            boundary_sources=loop_vertices,
            patch_surface_fits=patch_surface_fits,
            closure_kind="opening_carrier_boundary",
            feature_point_vertex_indices=feature_endpoints,
            open_as_closed_loop=True,
        )
    ]


def _prepare_single_patch_subholes(
    vertices: np.ndarray,
    arcs: Sequence[BoundaryArc],
    loop: Sequence[int],
    patch_surface_fits: Mapping[int, SurfaceFit],
) -> List[PreparedSubhole]:
    loop_vertices = [int(v) for v in loop]
    pts = vertices[np.array(loop_vertices, dtype=np.int64)]
    patch_label = int(arcs[0].patch_label if arcs else 0)
    return [
        _make_prepared_subhole(
            patch_label=patch_label,
            boundary_vertex_indices=loop_vertices,
            boundary_points=pts,
            closed_points=pts.copy(),
            boundary_sources=loop_vertices,
            patch_surface_fits=patch_surface_fits,
            closure_kind="opening_carrier_boundary",
            feature_point_vertex_indices=(loop_vertices[0], loop_vertices[-1]),
            open_as_closed_loop=True,
        )
    ]


def _point_to_segment_distance_3d(
    point: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> Tuple[float, float]:
    """点到 3D 线段距离及沿线参数 t∈[0,1]。"""
    p = np.asarray(point, dtype=np.float64).reshape(3)
    p0 = np.asarray(a, dtype=np.float64).reshape(3)
    p1 = np.asarray(b, dtype=np.float64).reshape(3)
    edge = p1 - p0
    edge_len2 = float(np.dot(edge, edge))
    if edge_len2 <= 1e-24:
        return float(np.linalg.norm(p - p0)), 0.0
    t = float(np.dot(p - p0, edge) / edge_len2)
    t = min(1.0, max(0.0, t))
    q = p0 + t * edge
    return float(np.linalg.norm(p - q)), t


def _carrier_duplicate_skip_start(
    pts_acc: List[np.ndarray],
    src_acc: List[int],
    new_pts: np.ndarray,
    new_src: Sequence[int],
    *,
    tol: float,
) -> int:
    """
    若 layout 折线首段沿已拼接孔边载体折返，跳过重复前缀（命题：载体不重复）。
    """
    if len(pts_acc) < 2 or new_pts.shape[0] < 2:
        return 0
    p_last = np.asarray(pts_acc[-1], dtype=np.float64)
    if int(new_src[0]) >= 0 and len(new_src) >= 2 and int(new_src[1]) < 0:
        # mesh 角点 → layout 内点：仅去掉与孔边末点重合的 mesh 角，不沿孔边
        # 切向吞掉 layout 采样（endpoint_remap 后 303→302 等汇交角处须保留
        # 完整 layout 链，否则相邻子孔 layout 段不对称并在 L4 留 seam 微环）。
        if float(np.linalg.norm(p_last - new_pts[0])) <= tol:
            return 1
        return 0
    if int(src_acc[-1]) < 0 or int(src_acc[-2]) < 0:
        return 0
    if int(new_src[0]) < 0:
        return 0
    p_prev = np.asarray(pts_acc[-2], dtype=np.float64)
    start = 0
    if float(np.linalg.norm(p_last - new_pts[0])) <= tol:
        start = 1
    if start >= int(new_pts.shape[0]):
        return start
    hole_dir = p_last - p_prev
    hole_len = float(np.linalg.norm(hole_dir))
    if hole_len <= tol:
        return start
    hole_dir = hole_dir / hole_len
    skip = start
    for i in range(start, int(new_pts.shape[0])):
        if int(new_src[i]) >= 0:
            break
        pi = np.asarray(new_pts[i], dtype=np.float64)
        rel = pi - p_last
        along = float(np.dot(rel, hole_dir))
        perp = float(np.linalg.norm(rel - along * hole_dir))
        if along < -tol * 2.0 or perp > tol * 3.0:
            break
        skip = i + 1
    return skip if skip > start else start


def _append_ring_segment(
    pts_acc: List[np.ndarray],
    src_acc: List[int],
    new_pts: np.ndarray,
    new_src: Sequence[int],
    *,
    tol: float = 1e-9,
) -> None:
    new_pts = np.asarray(new_pts, dtype=np.float64)
    new_src_list = [int(s) for s in new_src]
    start = _carrier_duplicate_skip_start(
        pts_acc, src_acc, new_pts, new_src_list, tol=tol
    )
    if pts_acc and new_pts.shape[0] > 0 and start < int(new_pts.shape[0]):
        if float(np.linalg.norm(pts_acc[-1] - new_pts[start])) <= tol:
            start += 1
    for i in range(start, int(new_pts.shape[0])):
        if (
            not pts_acc
            or float(np.linalg.norm(pts_acc[-1] - new_pts[i])) > tol
        ):
            pts_acc.append(np.asarray(new_pts[i], dtype=np.float64))
            src_acc.append(new_src_list[i] if i < len(new_src_list) else -1)


def _curve_polyline_bundle(
    curve: IntersectionCurve,
    curve_idx: int,
) -> Tuple[np.ndarray, List[int], int, int]:
    pts = np.asarray(curve.curve_points, dtype=np.float64)
    raw0 = int(curve.endpoint_vertex_indices[0])
    raw1 = int(curve.endpoint_vertex_indices[1])
    e0, e1 = _arrangement_endpoint_sources_for_curve(curve, curve_idx)
    sources = [int(e0)]
    for i in range(1, int(pts.shape[0]) - 1):
        sources.append(-(910_000 + 100 * int(curve_idx) + i))
    sources.append(int(e1))
    if pts.ndim == 2 and pts.shape[0] > 2:
        # Canonicalize near-endpoint samples once, before any ring assembly.
        # Otherwise one subhole may drop a point as a duplicate at append time
        # while the adjacent subhole keeps the reverse sample, leaving an
        # internal residual seam loop in L4.
        keep = [0]
        p0 = np.asarray(pts[0], dtype=np.float64)
        p1 = np.asarray(pts[-1], dtype=np.float64)
        for i in range(1, int(pts.shape[0]) - 1):
            pi = np.asarray(pts[i], dtype=np.float64)
            if float(np.linalg.norm(pi - p0)) <= 1e-9:
                continue
            if float(np.linalg.norm(pi - p1)) <= 1e-9:
                continue
            keep.append(i)
        keep.append(int(pts.shape[0]) - 1)
        if len(keep) != int(pts.shape[0]):
            pts = pts[np.asarray(keep, dtype=np.int64)]
            sources = [int(sources[i]) for i in keep]
    return pts, sources, raw0, raw1


def _curve_polyline_from_endpoint(
    curve: IntersectionCurve,
    curve_idx: int,
    start_endpoint: int,
) -> Tuple[np.ndarray, List[int]]:
    pts, sources, raw0, raw1 = _curve_polyline_bundle(curve, curve_idx)
    se = int(start_endpoint)
    if se == raw0 or se == int(sources[0]):
        return pts, sources
    if se == raw1 or se == int(sources[-1]):
        return pts[::-1], [int(x) for x in sources[::-1]]
    raise ValueError(
        f"endpoint {start_endpoint} not on curve pair {curve.patch_pair}"
    )


def _layout_source_xyz(
    source: int,
    curves: Sequence[IntersectionCurve],
) -> Optional[np.ndarray]:
    """arrangement source（含 ``-920000`` 等）在 layout 上的 3D 端点。"""
    for idx, curve in enumerate(curves):
        e0, e1 = _arrangement_endpoint_sources_for_curve(curve, idx)
        pts = np.asarray(curve.curve_points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] < 1:
            continue
        if int(e0) == int(source):
            return np.asarray(pts[0], dtype=np.float64).reshape(3)
        if int(e1) == int(source):
            return np.asarray(pts[-1], dtype=np.float64).reshape(3)
    return None


def _virtual_sources_spatially_coincident(
    j_a: int,
    j_b: int,
    curves: Sequence[IntersectionCurve],
    tol: float,
) -> bool:
    if int(j_a) == int(j_b):
        return True
    pa = _layout_source_xyz(int(j_a), curves)
    pb = _layout_source_xyz(int(j_b), curves)
    if pa is None or pb is None:
        return False
    return float(np.linalg.norm(pa - pb)) <= float(tol)


def _virtual_bridge_index_between(
    label: int,
    j_a: int,
    j_b: int,
    curves: Sequence[IntersectionCurve],
) -> Optional[int]:
    targets = {int(j_a), int(j_b)}
    for idx, curve in enumerate(curves):
        if int(label) not in {int(curve.patch_pair[0]), int(curve.patch_pair[1])}:
            continue
        if not _is_virtual_bridge_curve(curve):
            continue
        e0, e1 = _arrangement_endpoint_sources_for_curve(curve, idx)
        if {int(e0), int(e1)} == targets:
            return int(idx)
    return None


def _closest_polyline_index(
    point: np.ndarray,
    polyline: np.ndarray,
) -> int:
    p = np.asarray(point, dtype=np.float64).reshape(3)
    pts = np.asarray(polyline, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 1:
        return 0
    dists = np.linalg.norm(pts - p.reshape(1, 3), axis=1)
    return int(np.argmin(dists))


def _best_carrier_contact_index(
    polyline: np.ndarray,
    carrier_pts: np.ndarray,
    *,
    skip_last: int = 1,
) -> int:
    pts = np.asarray(polyline, dtype=np.float64)
    carrier = np.asarray(carrier_pts, dtype=np.float64)
    if pts.shape[0] < 2 or carrier.shape[0] < 1:
        return 0
    last_allowed = max(1, int(pts.shape[0]) - int(skip_last))
    best_i = 0
    best_d = float("inf")
    for i in range(last_allowed):
        d = float(np.min(np.linalg.norm(carrier - pts[i].reshape(1, 3), axis=1)))
        if d < best_d:
            best_d = d
            best_i = int(i)
    return best_i


def _dual_junction_carrier_ring_continuation(
    corner_a: int,
    corner_b: int,
    leave_a: Tuple[int, IntersectionCurve, int],
    leave_b: Tuple[int, IntersectionCurve, int],
    layout_curves: Sequence[IntersectionCurve],
    label: int,
    j_a: int,
    j_b: int,
) -> Optional[Tuple[np.ndarray, List[int]]]:
    """
    双汇交楔形：mesh→virtual 辐射线 + 完整 virtual bridge + virtual→mesh 辐射线。

    virtual bridge 是相邻子孔共享的 seam；不能按最近接触点裁成局部子段，
    否则两侧子孔会使用不同的采样 source，L4 焊接后留下内部残余环。
    """
    carrier_idx = _virtual_bridge_index_between(
        int(label), int(j_a), int(j_b), layout_curves
    )
    if carrier_idx is None:
        return None

    idx_a, curve_a, ep_a = leave_a
    idx_b, curve_b, ep_b = leave_b
    seg_b_pts, seg_b_src = _curve_polyline_from_endpoint(curve_b, int(idx_b), int(ep_b))
    seg_a_pts, seg_a_src = _curve_polyline_from_endpoint(curve_a, int(idx_a), int(ep_a))

    carrier_pts, carrier_src, _, _ = _curve_polyline_bundle(
        layout_curves[int(carrier_idx)],
        int(carrier_idx),
    )
    if int(carrier_src[0]) != int(j_b):
        carrier_pts = np.asarray(carrier_pts[::-1], dtype=np.float64)
        carrier_src = [int(x) for x in carrier_src[::-1]]

    pts_acc: List[np.ndarray] = []
    src_acc: List[int] = []
    _append_ring_segment(pts_acc, src_acc, seg_b_pts, seg_b_src)
    _append_ring_segment(pts_acc, src_acc, carrier_pts, carrier_src)
    _append_ring_segment(
        pts_acc,
        src_acc,
        np.asarray(seg_a_pts[::-1], dtype=np.float64),
        [int(x) for x in seg_a_src[::-1]],
    )
    if int(src_acc[-1]) != int(corner_a):
        return None
    closed = np.asarray(pts_acc, dtype=np.float64)
    if closed.shape[0] < 3:
        return None
    return closed, src_acc


def _find_virtual_bridge_curve_between(
    j_from: int,
    j_to: int,
    curves: Sequence[IntersectionCurve],
    label: int,
) -> Optional[Tuple[int, int, int]]:
    """查找连接两虚拟汇交点的 carrier bridge（curve_idx, start, end）。"""
    targets = {int(j_from), int(j_to)}
    for idx, curve in enumerate(curves):
        if not _is_virtual_bridge_curve(curve):
            continue
        if int(label) not in {int(curve.patch_pair[0]), int(curve.patch_pair[1])}:
            continue
        e0, e1 = _arrangement_endpoint_sources_for_curve(curve, int(idx))
        if {int(e0), int(e1)} != targets:
            continue
        if int(e0) == int(j_from):
            return int(idx), int(e0), int(e1)
        return int(idx), int(e1), int(e0)
    return None


def _junction_bridge_path(
    label: int,
    j_from: int,
    j_to: int,
    curves: Sequence[IntersectionCurve],
) -> Optional[List[Tuple[int, int, int]]]:
    """沿涉及 ``label`` 的辐射交线，在虚拟汇交端点间 BFS 找桥接路径。

    virtual-virtual bridge（两端皆 ``-920000`` 类 source）不参与拼环，避免子孔边界绕双汇交转一圈。
    """
    touching: List[Tuple[int, IntersectionCurve]] = [
        (int(i), curve)
        for i, curve in enumerate(curves)
        if int(label) in {int(curve.patch_pair[0]), int(curve.patch_pair[1])}
    ]
    adjacency: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
    for idx, curve in touching:
        if _is_virtual_bridge_curve(curve):
            continue
        e0, e1 = _arrangement_endpoint_sources_for_curve(curve, idx)
        adjacency[int(e0)].append((int(idx), int(e0), int(e1)))
        adjacency[int(e1)].append((int(idx), int(e1), int(e0)))
    target = int(j_to)
    queue: deque[Tuple[int, List[Tuple[int, int, int]]]] = deque([(int(j_from), [])])
    visited: Set[int] = {int(j_from)}
    while queue:
        node, path = queue.popleft()
        if int(node) == target:
            return path
        for curve_idx, start_ep, end_ep in adjacency.get(int(node), []):
            if int(end_ep) in visited:
                continue
            visited.add(int(end_ep))
            queue.append((int(end_ep), list(path) + [(int(curve_idx), int(start_ep), int(end_ep))]))
    return None


def _is_degenerate_intersection_curve(curve: IntersectionCurve) -> bool:
    """端点重合或零长度折线，不能参与 L3 剖分。

    虚拟端点占位（如 ``(-1,-1)``）不算退化，除非折线本身零长。
    """
    e0, e1 = (
        int(curve.endpoint_vertex_indices[0]),
        int(curve.endpoint_vertex_indices[1]),
    )
    if e0 >= 0 and e1 >= 0 and e0 == e1:
        return True
    pts = np.asarray(curve.curve_points, dtype=np.float64)
    if pts.ndim == 2 and pts.shape[0] >= 2:
        if float(np.linalg.norm(pts[-1] - pts[0])) <= 1e-12:
            return True
    return False


def _filter_layout_curves(curves: Sequence[IntersectionCurve]) -> List[IntersectionCurve]:
    return [curve for curve in curves if not _is_degenerate_intersection_curve(curve)]


def _ordered_arcs_for_label(arcs: Sequence[BoundaryArc], label: int) -> List[BoundaryArc]:
    return [arc for arc in arcs if int(arc.patch_label) == int(label)]


def _support_strip_bridge_vertices(
    v_from: int,
    v_to: int,
    degenerate_label_paths: Mapping[int, Sequence[Sequence[int]]],
) -> Optional[List[int]]:
    """
    命题 1（L3 构造推论）：两段 active 弧之间的孔边缺口须由 L2 登记的 support 退化路径填补。

    返回 ``v_from``→``v_to`` 方向上的中间顶点（不含两端）；找不到则 ``None``（O3）。
    """
    vf, vt = int(v_from), int(v_to)
    if vf == vt:
        return []
    for paths in degenerate_label_paths.values():
        for raw in paths:
            path = [int(x) for x in raw]
            if len(path) < 2:
                continue
            if path[0] == vf and path[-1] == vt:
                return path[1:-1] if len(path) > 2 else []
            if path[-1] == vf and path[0] == vt:
                inner = path[1:-1]
                return list(reversed(inner)) if inner else []
            if vf in path and vt in path:
                i0, i1 = path.index(vf), path.index(vt)
                if i0 < i1:
                    return path[i0 + 1 : i1]
                if i1 < i0:
                    return list(reversed(path[i1 + 1 : i0]))
    return None


def _loop_vertices_adjacent(loop: Sequence[int], va: int, vb: int) -> bool:
    """两顶点在孔环上是否相邻（含首尾）。"""
    loop_list = [int(v) for v in loop]
    n = len(loop_list)
    if n < 2:
        return False
    ia = loop_list.index(int(va))
    ib = loop_list.index(int(vb))
    return (ia + 1) % n == ib or (ib + 1) % n == ia


def _layout_curve_mesh_bridge(
    layout_curves: Sequence[IntersectionCurve],
    label: int,
    v_from: int,
    v_to: int,
) -> Optional[IntersectionCurve]:
    """同 label 的 layout 是否在 mesh 端点 ``v_from`` 与 ``v_to`` 间闭合缺口。"""
    vf, vt = int(v_from), int(v_to)
    for curve in layout_curves:
        if int(label) not in {int(curve.patch_pair[0]), int(curve.patch_pair[1])}:
            continue
        e0, e1 = (
            int(curve.endpoint_vertex_indices[0]),
            int(curve.endpoint_vertex_indices[1]),
        )
        if e0 >= 0 and e1 >= 0 and {e0, e1} == {vf, vt}:
            return curve
    return None


def _layout_curve_mesh_fan_bridge(
    layout_curves: Sequence[IntersectionCurve],
    label: int,
    v_from: int,
    v_to: int,
) -> Optional[List[Tuple[int, IntersectionCurve, int, int]]]:
    """两条 mesh↔virtual layout 曲线共享虚拟端点时，闭合同 label 的 mesh 缺口。"""
    vf, vt = int(v_from), int(v_to)
    candidates_from: List[Tuple[int, IntersectionCurve, int]] = []
    candidates_to: List[Tuple[int, IntersectionCurve, int]] = []
    for idx, curve in enumerate(layout_curves):
        if int(label) not in {int(curve.patch_pair[0]), int(curve.patch_pair[1])}:
            continue
        e0, e1 = _arrangement_endpoint_sources_for_curve(curve, int(idx))
        if int(e0) == vf and int(e1) < 0:
            candidates_from.append((int(idx), curve, int(e1)))
        elif int(e1) == vf and int(e0) < 0:
            candidates_from.append((int(idx), curve, int(e0)))
        if int(e0) == vt and int(e1) < 0:
            candidates_to.append((int(idx), curve, int(e1)))
        elif int(e1) == vt and int(e0) < 0:
            candidates_to.append((int(idx), curve, int(e0)))
    for idx_a, curve_a, virtual_a in candidates_from:
        for idx_b, curve_b, virtual_b in candidates_to:
            if int(virtual_a) != int(virtual_b):
                continue
            return [
                (int(idx_a), curve_a, vf, int(virtual_a)),
                (int(idx_b), curve_b, int(virtual_b), vt),
            ]
    return None


def _repartition_boundary_arcs_for_corner_vertices(
    loop: Sequence[int],
    arcs: Sequence[BoundaryArc],
    corner_vertex_ids: Sequence[int],
) -> List[BoundaryArc]:
    """按 L3 有效角点（active ± refinement）重切孔边弧，保留边 label。"""
    fp_positions = _loop_positions_for_vertices(loop, corner_vertex_ids)
    if len(fp_positions) < 2:
        return list(arcs)
    edge_labels = _boundary_edge_labels_from_arcs(loop, arcs)
    return _extract_boundary_arcs_from_labeled_feature_points(
        loop,
        fp_positions,
        edge_labels,
    )


def _hole_arc_polyline_with_layout_sutures(
    vertices: np.ndarray,
    loop: Sequence[int],
    arc_vids: Sequence[int],
    layout_curves: Sequence[IntersectionCurve],
    label: int,
) -> Tuple[np.ndarray, List[int]]:
    """拼接 label 孔边折线；非孔环邻接但存在 layout mesh 桥时插入交线段。"""
    vids = [int(v) for v in arc_vids]
    if len(vids) < 2:
        return np.zeros((0, 3), dtype=np.float64), []
    pts_acc: List[np.ndarray] = []
    src_acc: List[int] = []
    for i, vid in enumerate(vids):
        if i > 0:
            prev = int(vids[i - 1])
            cur = int(vid)
            if not _loop_vertices_adjacent(loop, prev, cur):
                bridge = _layout_curve_mesh_bridge(
                    layout_curves, int(label), prev, cur
                )
                if bridge is not None:
                    bridge_idx = next(
                        (
                            int(j)
                            for j, c in enumerate(layout_curves)
                            if int(c.endpoint_vertex_indices[0])
                            == int(bridge.endpoint_vertex_indices[0])
                            and int(c.endpoint_vertex_indices[1])
                            == int(bridge.endpoint_vertex_indices[1])
                            and tuple(sorted(c.patch_pair))
                            == tuple(sorted(bridge.patch_pair))
                        ),
                        0,
                    )
                    b_pts, b_src = _curve_polyline_from_endpoint(
                        bridge, bridge_idx, prev
                    )
                    _append_ring_segment(pts_acc, src_acc, b_pts, b_src)
                    if src_acc and int(src_acc[-1]) == cur:
                        continue
                fan_bridge = _layout_curve_mesh_fan_bridge(
                    layout_curves, int(label), prev, cur
                )
                if fan_bridge is not None:
                    ok = True
                    for bridge_idx, bridge_curve, start_src, end_src in fan_bridge:
                        if not _append_oriented_curve_polyline(
                            pts_acc,
                            src_acc,
                            bridge_curve,
                            bridge_idx,
                            start_src,
                            end_src,
                        ):
                            ok = False
                            break
                    if ok and src_acc and int(src_acc[-1]) == cur:
                        continue
        if not pts_acc or int(src_acc[-1]) != int(vid):
            pts_acc.append(np.asarray(vertices[int(vid)], dtype=np.float64))
            src_acc.append(int(vid))
    if not pts_acc:
        sub_p = vertices[np.asarray(vids, dtype=np.int64)]
        _append_ring_segment(pts_acc, src_acc, sub_p, vids)
    return np.asarray(pts_acc, dtype=np.float64), src_acc


def _layout_mesh_gap_bridge(
    layout_curves: Sequence[IntersectionCurve],
    label: int,
    v_from: int,
    v_to: int,
) -> Optional[List[int]]:
    """layout 在 mesh 端点间闭合缺口时返回空桥（孔边链直接相接，layout 在拼环时插入）。"""
    if _layout_curve_mesh_bridge(layout_curves, int(label), int(v_from), int(v_to)) is not None:
        return []
    if _layout_curve_mesh_fan_bridge(layout_curves, int(label), int(v_from), int(v_to)) is not None:
        return []
    return None


def _concatenate_label_boundary_arcs(
    vertices: np.ndarray,
    arcs_for_label: Sequence[BoundaryArc],
    *,
    active_label: int,
    degenerate_label_paths: Mapping[int, Sequence[Sequence[int]]],
    layout_curves: Optional[Sequence[IntersectionCurve]] = None,
) -> Tuple[List[int], np.ndarray, Optional[PartitionObstacle]]:
    """同 label 多段孔边弧串联；缺口仅允许 L2 support 退化路径桥接。"""
    vids_acc: List[int] = []
    pts_acc: List[np.ndarray] = []
    for arc in arcs_for_label:
        arc_vids = [int(v) for v in arc.vertex_indices]
        if len(arc_vids) < 2:
            continue
        arc_pts = vertices[np.asarray(arc_vids, dtype=np.int64)]
        if vids_acc and arc_vids[0] != vids_acc[-1]:
            bridge = _support_strip_bridge_vertices(
                vids_acc[-1],
                arc_vids[0],
                degenerate_label_paths,
            )
            if bridge is None and layout_curves is not None:
                bridge = _layout_mesh_gap_bridge(
                    layout_curves,
                    int(active_label),
                    vids_acc[-1],
                    arc_vids[0],
                )
            if bridge is None:
                return (
                    [],
                    np.zeros((0, 3), dtype=np.float64),
                    PartitionObstacle(
                        kind=PARTITION_OBSTACLE_O3,
                        label=int(active_label),
                        detail=(
                            f"support_strip_bridge_missing "
                            f"from={int(vids_acc[-1])} to={int(arc_vids[0])}"
                        ),
                    ),
                )
            for bv in bridge:
                if not vids_acc or int(bv) != vids_acc[-1]:
                    vids_acc.append(int(bv))
                    pts_acc.append(np.asarray(vertices[int(bv)], dtype=np.float64))
        start = 1 if vids_acc and arc_vids[0] == vids_acc[-1] else 0
        for i in range(start, len(arc_vids)):
            if not vids_acc or arc_vids[i] != vids_acc[-1]:
                vids_acc.append(arc_vids[i])
                pts_acc.append(np.asarray(arc_pts[i], dtype=np.float64))
    if len(vids_acc) < 2:
        return [], np.zeros((0, 3), dtype=np.float64), None
    return vids_acc, np.asarray(pts_acc, dtype=np.float64), None


def _layout_curves_for_label(
    layout_curves: Sequence[IntersectionCurve],
    label: int,
) -> List[Tuple[int, IntersectionCurve]]:
    return [
        (int(i), curve)
        for i, curve in enumerate(layout_curves)
        if int(label) in {int(curve.patch_pair[0]), int(curve.patch_pair[1])}
    ]


def _curve_leave_mesh_corner(
    touching: Sequence[Tuple[int, IntersectionCurve]],
    corner_vid: int,
) -> Optional[Tuple[int, IntersectionCurve, int]]:
    cv = int(corner_vid)
    for idx, curve in touching:
        r0 = int(curve.endpoint_vertex_indices[0])
        r1 = int(curve.endpoint_vertex_indices[1])
        if r0 == cv:
            return int(idx), curve, r0
        if r1 == cv:
            return int(idx), curve, r1
    return None


def _curve_mesh_endpoints(
    curve: IntersectionCurve,
) -> Tuple[Optional[int], Optional[int]]:
    e0, e1 = (
        int(curve.endpoint_vertex_indices[0]),
        int(curve.endpoint_vertex_indices[1]),
    )
    return (
        e0 if e0 >= 0 else None,
        e1 if e1 >= 0 else None,
    )


def _direct_wedge_curve_for_corners(
    touching: Sequence[Tuple[int, IntersectionCurve]],
    corner_a: int,
    corner_b: int,
) -> Optional[Tuple[int, IntersectionCurve]]:
    """命题 3：存在 mesh 端点恰为弧端点的单条 layout 交线。"""
    for idx, curve in touching:
        if _curve_connects_corners_directly(curve, int(corner_a), int(corner_b)):
            return int(idx), curve
    return None


def _mesh_curve_endpoint_sets(
    touching: Sequence[Tuple[int, IntersectionCurve]],
) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for _idx, curve in touching:
        e0, e1 = _curve_mesh_endpoints(curve)
        if e0 is None and e1 is None:
                continue
        out.append(
            (
                int(e0) if e0 is not None else -1,
                int(e1) if e1 is not None else -1,
            )
        )
    return out


def _loop_neighbor_vertices(loop: Sequence[int], vertex_id: int) -> Set[int]:
    loop_list = [int(v) for v in loop]
    n = len(loop_list)
    out: Set[int] = set()
    for i, v in enumerate(loop_list):
        if int(v) != int(vertex_id):
                continue
        out.add(int(loop_list[(i - 1) % n]))
        out.add(int(loop_list[(i + 1) % n]))
    return out


def _subarc_open_chain(
    arc_vids: Sequence[int],
    arc_pts: np.ndarray,
    start_v: int,
    end_v: int,
) -> Tuple[List[int], np.ndarray]:
    """沿已拼接孔边弧（开链）从 ``start_v`` 走到 ``end_v``。"""
    vids = [int(v) for v in arc_vids]
    pts = np.asarray(arc_pts, dtype=np.float64)
    sv, ev = int(start_v), int(end_v)
    if sv not in vids or ev not in vids:
        raise ValueError(f"subarc endpoints ({sv},{ev}) not on arc {vids}")
    i0, i1 = vids.index(sv), vids.index(ev)
    if i0 <= i1:
        idxs = list(range(i0, i1 + 1))
    else:
        idxs = list(range(i0, i1 - 1, -1))
    out_v = [vids[i] for i in idxs]
    out_p = pts[np.asarray(idxs, dtype=np.int64)]
    return out_v, out_p


def _feature_line_local_tolerance(
    vertices: np.ndarray,
    loop: Sequence[int],
    loop_step: float,
) -> float:
    """layout 特征线在孔边局部的距离容差（判断范围上界）。"""
    loop_pts = vertices[np.asarray(loop, dtype=np.int64)]
    diag = max(float(np.linalg.norm(np.ptp(loop_pts, axis=0))), 1e-12)
    step = float(loop_step) if loop_step > 1e-15 else 0.0
    return max(1e-8 * diag, 0.18 * step, 1e-9)


def _vertex_patch_labels(
    boundary_vertex_labels: Mapping[int, Sequence[int]],
    vertex_id: int,
) -> Set[int]:
    raw = boundary_vertex_labels.get(int(vertex_id))
    if raw is None:
        return set()
    return {int(x) for x in raw}


def _layout_direction_at_anchor(
    anchor: int,
    curve: IntersectionCurve,
    *,
    anchor_at_curve_start: bool,
    vertices: np.ndarray,
) -> np.ndarray:
    """锚点 L 处 layout 对接方向：指向 virtual junction 或曲线首段切向。"""
    pts = np.asarray(curve.curve_points, dtype=np.float64)
    x_anchor = np.asarray(vertices[int(anchor)], dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 1:
        return np.zeros(3, dtype=np.float64)
    e0, e1 = (
        int(curve.endpoint_vertex_indices[0]),
        int(curve.endpoint_vertex_indices[1]),
    )
    if anchor_at_curve_start:
        if e0 < 0:
            target = np.asarray(pts[0], dtype=np.float64)
        elif pts.shape[0] >= 2:
            target = np.asarray(pts[1], dtype=np.float64)
        else:
            target = np.asarray(pts[0], dtype=np.float64)
    else:
        if e1 < 0:
            target = np.asarray(pts[-1], dtype=np.float64)
        elif pts.shape[0] >= 2:
            target = np.asarray(pts[-2], dtype=np.float64)
        else:
            target = np.asarray(pts[-1], dtype=np.float64)
    direction = target - x_anchor
    length = float(np.linalg.norm(direction))
    if length <= 1e-15:
        return np.zeros(3, dtype=np.float64)
    return direction / length


def _point_near_local_layout_curve(
    point: np.ndarray,
    curve_points: np.ndarray,
    *,
    anchor_at_curve_start: bool,
    tol: float,
    max_segments: int = 4,
) -> bool:
    """点是否落在锚点侧 layout 折线的局部特征带内（不超出容差范围）。"""
    pts = np.asarray(curve_points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return False
    if anchor_at_curve_start:
        seg_ids = list(range(min(int(max_segments), int(pts.shape[0]) - 1)))
    else:
        start = max(0, int(pts.shape[0]) - 1 - int(max_segments))
        seg_ids = list(range(start, int(pts.shape[0]) - 1))
    p = np.asarray(point, dtype=np.float64).reshape(3)
    best = float("inf")
    for j in seg_ids:
        dist, _ = _point_to_segment_distance_3d(p, pts[j], pts[j + 1])
        best = min(best, float(dist))
    return best <= float(tol)


def _local_feature_band_window(
    loop: Sequence[int],
    anchor: int,
    direction: np.ndarray,
    curve: IntersectionCurve,
    vertices: np.ndarray,
    boundary_vertex_labels: Mapping[int, Sequence[int]],
    feature_vertex_ids: Set[int],
    *,
    tol: float,
    max_vertices: int = 8,
) -> List[int]:
    """
    锚点 L 处、沿 layout 方向、在特征线容差内的孔环局部顶点链 W(L)。

    不跨越：下一 L1 特征点、carrier label 脱离、特征线容差。
    """
    loop_list = [int(v) for v in loop]
    n = len(loop_list)
    anchor_i = loop_list.index(int(anchor))
    x_anchor = np.asarray(vertices[int(anchor)], dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    labels_anchor = _vertex_patch_labels(boundary_vertex_labels, int(anchor))

    def forward_dot(step: int) -> float:
        ni = (anchor_i + int(step)) % n
        delta = np.asarray(vertices[int(loop_list[ni])], dtype=np.float64) - x_anchor
        return float(np.dot(delta, direction))

    walk_step = 1 if forward_dot(1) >= forward_dot(-1) else -1
    e0, e1 = (
        int(curve.endpoint_vertex_indices[0]),
        int(curve.endpoint_vertex_indices[1]),
    )
    anchor_at_start = int(e0) == int(anchor)

    chain: List[int] = [int(anchor)]
    pos = anchor_i
    for _ in range(max(1, int(max_vertices) - 1)):
        pos = (pos + walk_step) % n
        vid = int(loop_list[pos])
        if vid in feature_vertex_ids and vid != int(anchor):
            break
        labels_v = _vertex_patch_labels(boundary_vertex_labels, vid)
        if labels_anchor and labels_v and not (labels_v & labels_anchor):
            break
        if not _point_near_local_layout_curve(
            vertices[vid],
            np.asarray(curve.curve_points, dtype=np.float64),
            anchor_at_curve_start=anchor_at_start,
            tol=float(tol),
        ):
            break
        chain.append(vid)
    return chain


def _first_local_protrusion_peak(
    chain: Sequence[int],
    anchor: int,
    direction: np.ndarray,
    vertices: np.ndarray,
    *,
    eps: float,
) -> Optional[int]:
    """W(L) 内沿 layout 方向的第一个 protrusion 局部极大（首峰，非全局）。"""
    if len(chain) < 2:
        return None
    x_anchor = np.asarray(vertices[int(anchor)], dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    protrusion = [
        float(np.dot(np.asarray(vertices[int(v)], dtype=np.float64) - x_anchor, direction))
        for v in chain
    ]
    for i in range(1, len(chain) - 1):
        if protrusion[i] <= protrusion[0] + float(eps):
            continue
        if protrusion[i] >= protrusion[i - 1] and protrusion[i] >= protrusion[i + 1]:
            return int(chain[i])
    if len(chain) >= 2:
        last = len(chain) - 1
        if protrusion[last] > protrusion[0] + float(eps) and protrusion[last] >= protrusion[last - 1]:
            return int(chain[last])
    return None


def refine_endpoint(
    anchor: int,
    curve: IntersectionCurve,
    *,
    loop: Sequence[int],
    vertices: np.ndarray,
    boundary_vertex_labels: Mapping[int, Sequence[int]],
    feature_vertex_ids: Set[int],
    loop_step: float,
) -> Tuple[int, Dict[str, object]]:
    """
    对单个 layout mesh 锚点 L 做局部端点 refinement。

    仅在特征线容差带 W(L) 内搜索；无更合理点则返回 L。
    """
    diag: Dict[str, object] = {"anchor": int(anchor), "applied": False}
    e0, e1 = (
        int(curve.endpoint_vertex_indices[0]),
        int(curve.endpoint_vertex_indices[1]),
    )
    if int(anchor) not in {e0, e1} or int(anchor) < 0:
        diag["reason"] = "not_curve_mesh_endpoint"
        return int(anchor), diag

    tol = _feature_line_local_tolerance(vertices, loop, float(loop_step))
    anchor_at_start = int(e0) == int(anchor)
    direction = _layout_direction_at_anchor(
        int(anchor),
        curve,
        anchor_at_curve_start=anchor_at_start,
        vertices=vertices,
    )
    if float(np.linalg.norm(direction)) <= 1e-12:
        diag["reason"] = "degenerate_layout_direction"
        return int(anchor), diag

    band = _local_feature_band_window(
        loop,
        int(anchor),
        direction,
        curve,
        vertices,
        boundary_vertex_labels,
        feature_vertex_ids,
        tol=tol,
    )
    diag["band"] = [int(v) for v in band]
    diag["feature_line_tol"] = float(tol)

    peak = _first_local_protrusion_peak(
        band,
        int(anchor),
        direction,
        vertices,
        eps=max(0.08 * float(tol), 1e-9),
    )
    if peak is None:
        diag["reason"] = "no_local_protrusion_peak"
        return int(anchor), diag

    labels_anchor = _vertex_patch_labels(boundary_vertex_labels, int(anchor))
    labels_peak = _vertex_patch_labels(boundary_vertex_labels, int(peak))
    if len(labels_anchor) >= 2:
        if not labels_peak or not (labels_peak < labels_anchor):
            diag["reason"] = "carrier_gate_failed"
            diag["candidate"] = int(peak)
            return int(anchor), diag

    protrusion_delta = float(
        np.dot(
            np.asarray(vertices[int(peak)], dtype=np.float64)
            - np.asarray(vertices[int(anchor)], dtype=np.float64),
            direction,
        )
    )
    if int(peak) == int(anchor) or protrusion_delta <= max(0.08 * float(tol), 1e-9):
        diag["reason"] = "insufficient_protrusion"
        diag["candidate"] = int(peak)
        return int(anchor), diag

    diag.update(
        {
            "applied": True,
            "refined": int(peak),
            "protrusion_delta": protrusion_delta,
        }
    )
    return int(peak), diag


def _build_layout_endpoint_remap(
    layout_curves: Sequence[IntersectionCurve],
    *,
    loop: Sequence[int],
    vertices: np.ndarray,
    boundary_vertex_labels: Mapping[int, Sequence[int]],
    feature_vertex_ids: Sequence[int],
    loop_step: float,
) -> Tuple[Dict[int, int], List[Dict[str, object]]]:
    """对每个 layout mesh 端点独立调用 ``refine_endpoint``，汇总非恒等映射。"""
    loop_set = {int(v) for v in loop}
    feature_set = {int(v) for v in feature_vertex_ids if int(v) in loop_set}
    endpoint_remap: Dict[int, int] = {}
    diagnostics: List[Dict[str, object]] = []
    seen_anchors: Set[int] = set()

    for curve in layout_curves:
        for raw in curve.endpoint_vertex_indices:
            anchor = int(raw)
            if anchor < 0 or anchor not in loop_set or anchor in seen_anchors:
                continue
            seen_anchors.add(anchor)
            refined, diag = refine_endpoint(
                anchor,
                curve,
                loop=loop,
                vertices=vertices,
                boundary_vertex_labels=boundary_vertex_labels,
                feature_vertex_ids=feature_set,
                loop_step=float(loop_step),
            )
            diagnostics.append(dict(diag))
            if int(refined) != int(anchor):
                endpoint_remap[int(anchor)] = int(refined)
    return endpoint_remap, diagnostics


def _layout_is_complete_mesh_pair_partition(
    layout_curves: Sequence[IntersectionCurve],
    feature_vertex_ids: Sequence[int],
) -> bool:
    feature_set = {int(v) for v in feature_vertex_ids}
    if len(feature_set) < 6 or len(feature_set) % 2 != 0:
        return False
    mesh_endpoints: List[int] = []
    for curve in layout_curves:
        e0, e1 = (
            int(curve.endpoint_vertex_indices[0]),
            int(curve.endpoint_vertex_indices[1]),
        )
        if e0 < 0 or e1 < 0:
            return False
        mesh_endpoints.extend([int(e0), int(e1)])
    return len(mesh_endpoints) == len(feature_set) and set(mesh_endpoints) == feature_set


def _layout_curve_endpoint_position(
    endpoint_vid: int,
    curve: IntersectionCurve,
    vertices: np.ndarray,
    *,
    at_start: bool,
) -> np.ndarray:
    """mesh 端点用顶点坐标；virtual junction（负 id）用曲线端点几何。"""
    if int(endpoint_vid) >= 0:
        return np.asarray(vertices[int(endpoint_vid)], dtype=np.float64).reshape(3)
    eos = np.asarray(curve.endpoints_on_boundary, dtype=np.float64)
    if eos.ndim == 2 and eos.shape[0] >= 2:
        return np.asarray(eos[0 if at_start else 1], dtype=np.float64).reshape(3)
        pts = np.asarray(curve.curve_points, dtype=np.float64)
    return np.asarray(pts[0 if at_start else -1], dtype=np.float64).reshape(3)


def _remap_layout_curves_for_endpoint_remap(
    layout_curves: Sequence[IntersectionCurve],
    endpoint_remap: Mapping[int, int],
    vertices: np.ndarray,
    patch_surface_fits: Mapping[int, SurfaceFit],
    hole_center: np.ndarray,
    loop_step: float,
) -> List[IntersectionCurve]:
    """按 ``endpoint_remap`` 重连 layout 曲线 mesh 端点并重采样。"""
    if not endpoint_remap:
        return list(layout_curves)
    out: List[IntersectionCurve] = []
    for curve in layout_curves:
        e0, e1 = (
            int(curve.endpoint_vertex_indices[0]),
            int(curve.endpoint_vertex_indices[1]),
        )
        n0 = int(endpoint_remap.get(e0, e0))
        n1 = int(endpoint_remap.get(e1, e1))
        if n0 >= 0 and n1 >= 0 and n0 == n1:
            continue
        pts = np.asarray(curve.curve_points, dtype=np.float64).copy()
        if e0 in endpoint_remap and int(n0) >= 0 and pts.shape[0] >= 1:
            pts[0] = np.asarray(vertices[int(n0)], dtype=np.float64)
        if e1 in endpoint_remap and int(n1) >= 0 and pts.shape[0] >= 1:
            pts[-1] = np.asarray(vertices[int(n1)], dtype=np.float64)
        if int(n0) != e0 or int(n1) != e1:
            pair = tuple(sorted((int(curve.patch_pair[0]), int(curve.patch_pair[1]))))
            if pair[0] in patch_surface_fits and pair[1] in patch_surface_fits:
                p0 = _layout_curve_endpoint_position(n0, curve, vertices, at_start=True)
                p1 = _layout_curve_endpoint_position(n1, curve, vertices, at_start=False)
                ref_step = max(0.75 * float(loop_step or 0.0), 1e-9)
                length = float(np.linalg.norm(p1 - p0))
                target_n = max(
                    int(pts.shape[0]),
                    feature_curve_sample_count(length, ref_step),
                )
                pts_curve = recover_curve_between_points(
                    patch_surface_fits[pair[0]],
                    patch_surface_fits[pair[1]],
                    p0,
                    p1,
                    _feature_curve_guide_point(
                        p0, p1, hole_center, endpoint_vertex_indices=(int(n0), int(n1))
                    ),
                    n_samples=target_n,
                    min_samples=0,
                    endpoint_vertex_indices=(int(n0), int(n1)),
                    intersection_sampling_reference_step=ref_step,
                )
                pts = np.asarray(pts_curve.curve_points, dtype=np.float64)
        out.append(
            replace(
                curve,
                curve_points=pts,
                endpoints_on_boundary=np.vstack((pts[0], pts[-1])),
                endpoint_vertex_indices=(int(n0), int(n1)),
            )
        )
    return out if out else list(layout_curves)


def _virtual_layout_corner_at_mesh(
    touching: Sequence[Tuple[int, IntersectionCurve]],
    mesh_corner: int,
) -> Optional[int]:
    """与 ``mesh_corner`` 相接的 layout 曲线上的虚拟汇交端点 id。"""
    mc = int(mesh_corner)
    for _idx, curve in touching:
        e0, e1 = (
            int(curve.endpoint_vertex_indices[0]),
            int(curve.endpoint_vertex_indices[1]),
        )
        if e0 == mc and e1 < 0:
            return int(e1)
        if e1 == mc and e0 < 0:
            return int(e0)
    return None


def _adjust_wedge_corners_for_inactive(
    corner_a: int,
    corner_b: int,
    touching: Sequence[Tuple[int, IntersectionCurve]],
    inactive: Set[int],
) -> Tuple[int, int]:
    """已降级角点用同 label layout 上的虚拟汇交端点替代。"""
    ca, cb = int(corner_a), int(corner_b)
    if ca in inactive and cb not in inactive:
        virt = _virtual_layout_corner_at_mesh(touching, cb)
        if virt is not None:
            ca = int(virt)
    if cb in inactive and ca not in inactive:
        virt = _virtual_layout_corner_at_mesh(touching, ca)
        if virt is not None:
            cb = int(virt)
    return ca, cb


def _normalize_wedge_corners_for_layout(
    corner_a: int,
    corner_b: int,
    touching: Sequence[Tuple[int, IntersectionCurve]],
    inactive: Set[int],
) -> Tuple[int, int]:
    """
    inactive 修剪后孔边弧端点可能落在非 layout 的边界顶点上；
    若对侧角点可接 layout，则改接其虚拟汇交端点。
    """
    ca, cb = _adjust_wedge_corners_for_inactive(
        int(corner_a), int(corner_b), touching, inactive
    )
    if _curve_leave_mesh_corner(touching, ca) is None:
        if _curve_leave_mesh_corner(touching, cb) is not None:
            virt = _virtual_layout_corner_at_mesh(touching, cb)
            if virt is not None:
                ca = int(virt)
    if _curve_leave_mesh_corner(touching, cb) is None:
        if _curve_leave_mesh_corner(touching, ca) is not None:
            virt = _virtual_layout_corner_at_mesh(touching, ca)
            if virt is not None:
                cb = int(virt)
    return ca, cb


def _trim_inactive_arc_endpoints(
    arc_vids: Sequence[int],
    inactive: Set[int],
) -> List[int]:
    """孔边弧链两端去掉已降级特征点，避免虚拟楔角仍拖入 inactive 端点。"""
    vids = [int(v) for v in arc_vids]
    while len(vids) > 1 and int(vids[0]) in inactive:
        vids = vids[1:]
    while len(vids) > 1 and int(vids[-1]) in inactive:
        vids = vids[:-1]
    return vids


def _fan_wedge_ready_alternative_candidates(
    vertices: np.ndarray,
    loop: Sequence[int],
    layout_curves: Sequence[IntersectionCurve],
    patch_surface_fits: Mapping[int, SurfaceFit],
    label: int,
    *,
    closure_kind: str,
) -> List[Dict[str, object]]:
    """同一 virtual fan 上多条 mesh 辐射线时，枚举可三角化的楔形闭环。"""
    touching = _layout_curves_for_label(layout_curves, int(label))
    by_virtual: Dict[int, Dict[int, List[Tuple[int, IntersectionCurve, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for idx, curve in touching:
        e0, e1 = _arrangement_endpoint_sources_for_curve(curve, int(idx))
        if int(e0) >= 0 and int(e1) < 0:
            by_virtual[int(e1)][int(e0)].append((int(idx), curve, int(e0)))
        elif int(e1) >= 0 and int(e0) < 0:
            by_virtual[int(e0)][int(e1)].append((int(idx), curve, int(e1)))

    candidates: List[Dict[str, object]] = []
    for virtual, endpoints in by_virtual.items():
        mesh_endpoints = sorted(int(v) for v in endpoints)
        if len(mesh_endpoints) < 2:
            continue
        for a, b in combinations(mesh_endpoints, 2):
            for forward in (True, False):
                chain = _loop_chain_directed(loop, int(a), int(b), forward=forward)
                if len(chain) < 2:
                    continue
                arc_pts = vertices[np.asarray(chain, dtype=np.int64)]
                for leave_a in endpoints[int(a)]:
                    for leave_b in endpoints[int(b)]:
                        assembled = _assemble_curve_arc_subhole_ring(
                            vertices,
                            arc_pts,
                            chain,
                            int(a),
                            int(b),
                            leave_a,
                            leave_b,
                            layout_curves,
                            int(label),
                        )
                        if assembled is None:
                            continue
                        closed, src_acc = assembled
                        sub = _make_prepared_subhole(
                            patch_label=int(label),
                            boundary_vertex_indices=chain,
                            boundary_points=closed,
                            closed_points=closed.copy(),
                            boundary_sources=src_acc,
                            patch_surface_fits=patch_surface_fits,
                            closure_kind=str(closure_kind),
                            feature_point_vertex_indices=(int(a), int(b)),
                            open_as_closed_loop=True,
                        )
                        readiness = assess_patch_boundary_readiness(
                            np.asarray(sub.closed_boundary_points, dtype=np.float64),
                            np.asarray(sub.boundary_points_2d, dtype=np.float64),
                        )
                        if not bool(readiness.get("ready", False)):
                            continue
                        # Prefer compact fan wedges; tie-break with fewer samples.
                        score = len(chain) * 1000 + int(sub.closed_boundary_points.shape[0])
                        candidates.append(
                            {
                                "score": int(score),
                                "subhole": sub,
                                "label": int(label),
                                "virtual": int(virtual),
                                "mesh_endpoints": (int(a), int(b)),
                                "curve_indices": frozenset(
                                    {int(leave_a[0]), int(leave_b[0])}
                                ),
                            }
                        )
    candidates.sort(
        key=lambda item: (
            int(item["score"]),
            int(item["label"]),
            tuple(sorted(int(x) for x in item["curve_indices"])),
        )
    )
    return candidates


def _fan_wedge_ready_alternative_subhole(
    vertices: np.ndarray,
    loop: Sequence[int],
    layout_curves: Sequence[IntersectionCurve],
    patch_surface_fits: Mapping[int, SurfaceFit],
    label: int,
    *,
    closure_kind: str,
    required_boundary_chain: Optional[Sequence[int]] = None,
) -> Optional[PreparedSubhole]:
    candidates = _fan_wedge_ready_alternative_candidates(
        vertices,
        loop,
        layout_curves,
        patch_surface_fits,
        int(label),
        closure_kind=str(closure_kind),
    )
    if not candidates:
        return None
    required_edges = _open_chain_edge_set(required_boundary_chain or [])
    for candidate in candidates:
        sub = candidate.get("subhole")
        if not isinstance(sub, PreparedSubhole):
            continue
        if required_edges:
            candidate_edges = _open_chain_edge_set(sub.boundary_vertex_indices)
            if candidate_edges != required_edges:
                continue
        return sub
    return None


def _open_chain_edge_set(chain: Sequence[int]) -> Set[Tuple[int, int]]:
    """Undirected edge set for an open hole-boundary chain."""
    vids = [int(v) for v in chain]
    out: Set[Tuple[int, int]] = set()
    for a, b in zip(vids[:-1], vids[1:]):
        ia, ib = int(a), int(b)
        if ia == ib:
            continue
        out.add((ia, ib) if ia < ib else (ib, ia))
    return out


def _feature_curve_indices_from_sources(sources: Sequence[int]) -> Set[int]:
    """从 ``-910xxx`` 采样 source 反推高采样 feature curve 索引。"""
    out: Set[int] = set()
    for src in sources:
        s = int(src)
        if -920000 < s <= -910000:
            out.add(int((-s - 910000) // 100))
    return out


def _augment_unpaired_fan_seams(
    prepared: Sequence[PreparedSubhole],
    vertices: np.ndarray,
    loop: Sequence[int],
    layout_curves: Sequence[IntersectionCurve],
    patch_surface_fits: Mapping[int, SurfaceFit],
    active_labels: Sequence[int],
    *,
    closure_kind: str,
) -> List[PreparedSubhole]:
    """补齐只被单个子孔使用的 fan seam，避免 L4 留内部 residual loop。"""
    out = list(prepared)
    active = {int(label) for label in active_labels}
    usage: Dict[int, Set[int]] = defaultdict(set)
    for subhole in out:
        label = int(subhole.patch_label)
        for curve_idx in _feature_curve_indices_from_sources(subhole.boundary_sources):
            usage[int(curve_idx)].add(label)

    existing: Set[Tuple[int, Tuple[int, ...]]] = {
        (
            int(subhole.patch_label),
            tuple(int(src) for src in subhole.boundary_sources),
        )
        for subhole in out
    }
    candidates_by_label: Dict[int, List[Dict[str, object]]] = {
        int(label): _fan_wedge_ready_alternative_candidates(
            vertices,
            loop,
            layout_curves,
            patch_surface_fits,
            int(label),
            closure_kind=str(closure_kind),
        )
        for label in active
    }
    for curve_idx, labels in sorted(usage.items()):
        if len(labels) != 1:
            continue
        curve = layout_curves[int(curve_idx)] if 0 <= int(curve_idx) < len(layout_curves) else None
        if curve is None or int(np.asarray(curve.curve_points).shape[0]) <= 2:
            continue
        pair_labels = {int(x) for x in curve.patch_pair if int(x) in active}
        missing_labels = sorted(pair_labels - {int(x) for x in labels})
        for missing_label in missing_labels:
            for cand in candidates_by_label.get(int(missing_label), []):
                curve_indices = {int(x) for x in cand["curve_indices"]}
                if int(curve_idx) not in curve_indices:
                    continue
                sub = cand["subhole"]
                if not isinstance(sub, PreparedSubhole):
                    continue
                key = (
                    int(sub.patch_label),
                    tuple(int(src) for src in sub.boundary_sources),
                )
                if key in existing:
                    continue
                out.append(sub)
                existing.add(key)
                for used_idx in _feature_curve_indices_from_sources(sub.boundary_sources):
                    usage[int(used_idx)].add(int(sub.patch_label))
                break
    return out


def _wedge_partition_for_label(
    vertices: np.ndarray,
    loop: Sequence[int],
    arcs: Sequence[BoundaryArc],
    layout_curves: Sequence[IntersectionCurve],
    patch_surface_fits: Mapping[int, SurfaceFit],
    fill_classification: FillPatchClassification,
    label: int,
    *,
    forced_corners: Optional[Tuple[int, int]] = None,
    forced_arc: Optional[Tuple[List[int], np.ndarray]] = None,
    layout_curves_override: Optional[Sequence[IntersectionCurve]] = None,
    closure_kind: str = CLOSURE_CURVE_ARC_PARTITION,
    endpoint_remap: Optional[Mapping[int, int]] = None,
) -> Tuple[Optional[PreparedSubhole], Optional[PartitionObstacle]]:
    """
    楔形剖分闸门（命题 3/4）+ 构造单 label 子孔。

    - 弧端点无 layout 交线 → O4（角点不可达）
    - mesh 端点不与弧端点配对且拼环失败 → O4（非楔形）
    """
    layout_use = (
        list(layout_curves_override)
        if layout_curves_override is not None
        else list(layout_curves)
    )
    if forced_arc is not None:
        arc_vids, arc_pts = forced_arc
        arc_vids = [int(v) for v in arc_vids]
        arc_pts = np.asarray(arc_pts, dtype=np.float64)
        arc_obstacle = None
    else:
        label_arcs = _ordered_arcs_for_label(arcs, int(label))
        if not label_arcs:
            return None, PartitionObstacle(
                kind=PARTITION_OBSTACLE_O4,
                label=int(label),
                detail="missing_boundary_arc",
            )
        arc_vids, arc_pts, arc_obstacle = _concatenate_label_boundary_arcs(
            vertices,
            label_arcs,
            active_label=int(label),
            degenerate_label_paths=fill_classification.degenerate_label_paths,
            layout_curves=layout_use,
        )
    if arc_obstacle is not None:
        return None, arc_obstacle
    inactive = {
        int(v) for v in fill_classification.inactive_feature_points
    }
    if inactive:
        arc_vids = _trim_inactive_arc_endpoints(arc_vids, inactive)
        if len(arc_vids) >= 2:
            arc_pts = vertices[np.asarray(arc_vids, dtype=np.int64)]
        elif len(arc_vids) == 1:
            arc_pts = vertices[np.asarray(arc_vids, dtype=np.int64)]
        else:
            return None, PartitionObstacle(
                kind=PARTITION_OBSTACLE_O4,
                label=int(label),
                detail="degenerate_boundary_arc_after_inactive_trim",
            )

    if len(arc_vids) < 1:
        return None, PartitionObstacle(
            kind=PARTITION_OBSTACLE_O4,
            label=int(label),
            detail="degenerate_boundary_arc",
        )

    if forced_corners is not None:
        corner_a, corner_b = int(forced_corners[0]), int(forced_corners[1])
    elif len(arc_vids) >= 2:
        corner_a = int(arc_vids[0])
        corner_b = int(arc_vids[-1])
    else:
        corner_a = int(arc_vids[0])
        corner_b = int(arc_vids[0])
    touching = _layout_curves_for_label(layout_use, int(label))
    if not touching:
        return None, PartitionObstacle(
            kind=PARTITION_OBSTACLE_O4,
            label=int(label),
            detail=(
                f"no_layout_curve_for_label "
                f"arc_endpoints=({corner_a},{corner_b})"
            ),
        )

    corner_a, corner_b = _normalize_wedge_corners_for_layout(
        corner_a, corner_b, touching, inactive
    )

    mesh_virtual = _wedge_corners_mesh_virtual(corner_a, corner_b)
    if mesh_virtual is not None:
        mesh_corner, _virtual_corner = mesh_virtual
        arc_vids, arc_pts = _shorten_arc_for_virtual_wedge(
            vertices,
            loop,
            arc_vids,
            int(mesh_corner),
        )
        if len(arc_vids) < 1:
            return None, PartitionObstacle(
                kind=PARTITION_OBSTACLE_O4,
                label=int(label),
                detail="degenerate_boundary_arc_after_virtual_shorten",
            )
    elif endpoint_remap:
        arc_vids, arc_pts = _maybe_shorten_arc_for_refined_corner(
            vertices,
            loop,
            arc_vids,
            int(corner_a),
            int(corner_b),
            endpoint_remap,
            patch_surface_fits,
            int(label),
        )
        if len(arc_vids) < 1:
            return None, PartitionObstacle(
                kind=PARTITION_OBSTACLE_O4,
                label=int(label),
                detail="degenerate_boundary_arc_after_refined_shorten",
            )

    leave_a = _curve_leave_mesh_corner(touching, corner_a)
    leave_b = _curve_leave_mesh_corner(touching, corner_b)
    if leave_a is None:
        return None, PartitionObstacle(
            kind=PARTITION_OBSTACLE_O4,
            label=int(label),
            detail=(
                f"corner_without_layout_curve vertex={corner_a} "
                f"arc_endpoints=({corner_a},{corner_b}) "
                f"curve_mesh_endpoints={_mesh_curve_endpoint_sets(touching)}"
            ),
        )
    if leave_b is None:
        return None, PartitionObstacle(
            kind=PARTITION_OBSTACLE_O4,
            label=int(label),
            detail=(
                f"corner_without_layout_curve vertex={corner_b} "
                f"arc_endpoints=({corner_a},{corner_b}) "
                f"curve_mesh_endpoints={_mesh_curve_endpoint_sets(touching)}"
            ),
        )

    direct = _direct_wedge_curve_for_corners(touching, corner_a, corner_b)
    mesh_mismatch = direct is None
    idx_a, curve_a, ep_a = leave_a
    idx_b, curve_b, ep_b = leave_b

    hole_pts, hole_src = _hole_arc_polyline_with_layout_sutures(
        vertices,
        loop,
        arc_vids,
        layout_use,
        int(label),
    )

    junction_coincidence_tol = _junction_cluster_tolerance(
        vertices,
        loop,
        _mean_hole_loop_edge_len(vertices, loop),
    )
    assembled = _assemble_curve_arc_subhole_ring(
        vertices,
        hole_pts,
        hole_src,
        corner_a,
        corner_b,
        (int(idx_a), curve_a, int(ep_a)),
        (int(idx_b), curve_b, int(ep_b)),
        layout_use,
        int(label),
        junction_coincidence_tol=float(junction_coincidence_tol),
    )
    if assembled is None:
        return None, PartitionObstacle(
            kind=PARTITION_OBSTACLE_O4,
            label=int(label),
            detail=(
                f"non_wedge_closure_failed "
                f"arc_endpoints=({corner_a},{corner_b}) "
                f"curve_mesh_endpoints={_mesh_curve_endpoint_sets(touching)} "
                f"mesh_endpoint_mismatch={mesh_mismatch}"
            ),
        )

    if mesh_mismatch:
        # 允许虚拟汇交拼环（hole_test3），但记录 mesh 端点未直接配对。
        pass

    closed, src_acc = assembled
    sub = _make_prepared_subhole(
        patch_label=int(label),
        boundary_vertex_indices=arc_vids,
        boundary_points=closed,
        closed_points=closed.copy(),
        boundary_sources=src_acc,
        patch_surface_fits=patch_surface_fits,
        closure_kind=str(closure_kind),
        feature_point_vertex_indices=(corner_a, corner_b),
        open_as_closed_loop=True,
    )
    readiness = assess_patch_boundary_readiness(
        np.asarray(sub.closed_boundary_points, dtype=np.float64),
        np.asarray(sub.boundary_points_2d, dtype=np.float64),
    )
    if not bool(readiness.get("ready", False)):
        fan_sub = _fan_wedge_ready_alternative_subhole(
            vertices,
            loop,
            layout_use,
            patch_surface_fits,
            int(label),
            closure_kind=str(closure_kind),
            required_boundary_chain=arc_vids,
        )
        if fan_sub is not None:
            return fan_sub, None
    return sub, None


def _curve_connects_corners_directly(
    curve: IntersectionCurve,
    corner_a: int,
    corner_b: int,
) -> bool:
    """交线两端均为孔环 mesh 角点且正好是两角点。"""
    e0, e1 = (
        int(curve.endpoint_vertex_indices[0]),
        int(curve.endpoint_vertex_indices[1]),
    )
    if e0 < 0 or e1 < 0:
        return False
    return {e0, e1} == {int(corner_a), int(corner_b)}


def _loop_chain_directed(
    loop: Sequence[int],
    start_v: int,
    end_v: int,
    *,
    forward: bool = True,
) -> List[int]:
    loop_list = [int(v) for v in loop]
    n = len(loop_list)
    try:
        ia = loop_list.index(int(start_v))
        ib = loop_list.index(int(end_v))
    except ValueError:
        return [int(start_v), int(end_v)]
    if ia == ib:
        return [int(start_v)]
    chain: List[int] = []
    i = ia
    while True:
        chain.append(int(loop_list[i]))
        if int(loop_list[i]) == int(end_v):
            break
        i = (i + 1) % n if forward else (i - 1) % n
        if len(chain) > n:
            break
    return chain


def _short_loop_chain(
    loop: Sequence[int],
    start_v: int,
    end_v: int,
) -> List[int]:
    fwd = _loop_chain_directed(loop, start_v, end_v, forward=True)
    bwd = _loop_chain_directed(loop, start_v, end_v, forward=False)
    return fwd if len(fwd) <= len(bwd) else bwd


def _maybe_shorten_arc_for_refined_corner(
    vertices: np.ndarray,
    loop: Sequence[int],
    arc_vids: Sequence[int],
    corner_a: int,
    corner_b: int,
    endpoint_remap: Mapping[int, int],
    patch_surface_fits: Mapping[int, SurfaceFit],
    label: int,
) -> Tuple[List[int], np.ndarray]:
    """
    L3 端点替换后的 refined 角点仍必须保留重切得到的孔边弧。

    参数域自交属于后续参数化/重参数化问题；不能在这里裁短孔边弧，
    否则多个 PreparedSubhole 的边界并集可能不再覆盖原始孔环。
    """
    vids = [int(v) for v in arc_vids]
    return vids, vertices[np.asarray(vids, dtype=np.int64)]


def _merge_coincident_virtual_ring_vertices(
        points: np.ndarray,
        sources: Sequence[int],
    *,
    tol: float,
) -> Tuple[np.ndarray, List[int]]:
    """合并拼环时连续重合的虚拟汇交采样点（如重复 ``-920000``）。"""
    def _is_virtual_junction_source(source: int) -> bool:
        # ``-920000`` 系列是 arrangement 虚拟汇交点；
        # ``-910xxx`` 是曲线内部采样点，不能用 junction tolerance 吞掉。
        return int(source) <= -920000

    pts = np.asarray(points, dtype=np.float64)
    src = [int(s) for s in sources]
    if pts.shape[0] < 2 or len(src) != int(pts.shape[0]):
        return pts, src
    out_p: List[np.ndarray] = [pts[0]]
    out_s: List[int] = [src[0]]
    out_n: List[int] = [1]
    for i in range(1, int(pts.shape[0])):
        if int(src[i]) == int(out_s[-1]):
            n = int(out_n[-1])
            out_p[-1] = (out_p[-1] * float(n) + pts[i]) / float(n + 1)
            out_n[-1] = n + 1
            continue
        dup = float(np.linalg.norm(pts[i] - out_p[-1])) <= float(tol)
        if (
            dup
            and _is_virtual_junction_source(int(src[i]))
            and _is_virtual_junction_source(int(out_s[-1]))
        ):
            continue
        out_p.append(pts[i])
        out_s.append(src[i])
        out_n.append(1)
    return np.asarray(out_p, dtype=np.float64), out_s


def _shorten_arc_for_virtual_wedge(
    vertices: np.ndarray,
    loop: Sequence[int],
    arc_vids: Sequence[int],
    mesh_corner: int,
) -> Tuple[List[int], np.ndarray]:
    """
    opening-carrier：孔边弧取 layout mesh 角点到对侧弧端在孔环上的短链，
    避免长弧 + layout 弦在平面投影自交。
    """
    vids = [int(v) for v in arc_vids]
    mc = int(mesh_corner)
    loop_set = {int(v) for v in loop}
    if mc not in loop_set:
        return vids, vertices[np.asarray(vids, dtype=np.int64)]
    if len(vids) < 2:
        return vids, vertices[np.asarray(vids, dtype=np.int64)]
    other_candidates = sorted({int(vids[0]), int(vids[-1])} - {mc})
    if not other_candidates:
        other_candidates = [int(v) for v in vids if int(v) != mc]
    if not other_candidates:
        return vids, vertices[np.asarray(vids, dtype=np.int64)]
    other = int(other_candidates[0])
    short = _short_loop_chain(loop, mc, other)
    if len(short) >= 2 and len(short) < len(vids):
        return short, vertices[np.asarray(short, dtype=np.int64)]
    return vids, vertices[np.asarray(vids, dtype=np.int64)]


def _wedge_corners_mesh_virtual(
    corner_a: int,
    corner_b: int,
) -> Optional[Tuple[int, int]]:
    """楔形剖分角点为 (mesh, virtual) 时返回 (mesh, virtual)。"""
    ca, cb = int(corner_a), int(corner_b)
    if ca >= 0 and cb < 0:
        return ca, cb
    if cb >= 0 and ca < 0:
        return cb, ca
    return None


def _layout_endpoint_xyz(
    source: int,
    vertices: np.ndarray,
    layout_curves: Sequence[IntersectionCurve],
) -> Optional[np.ndarray]:
    """layout source / mesh 顶点 → 3D 坐标。"""
    sid = int(source)
    if sid >= 0:
        return np.asarray(vertices[sid], dtype=np.float64).reshape(3)
    xyz = _layout_source_xyz(sid, layout_curves)
    if xyz is None:
        return None
    return np.asarray(xyz, dtype=np.float64).reshape(3)


def _append_oriented_curve_polyline(
    pts_acc: List[np.ndarray],
    src_acc: List[int],
    curve: IntersectionCurve,
    curve_idx: int,
    start_source: int,
    end_source: int,
) -> bool:
    """将 L2 恢复的特征折线（含内点采样）按 source 方向编入子孔边界。"""
    pts, sources, raw0, raw1 = _curve_polyline_bundle(curve, int(curve_idx))
    se = int(start_source)
    ee = int(end_source)
    if se == ee:
        return True
    e0, e1 = _arrangement_endpoint_sources_for_curve(curve, int(curve_idx))
    if se == int(e0) and ee == int(e1):
        seg_pts, seg_src = pts, sources
    elif se == int(e1) and ee == int(e0):
        seg_pts, seg_src = pts[::-1], [int(x) for x in sources[::-1]]
    elif se == int(raw0) or se == int(sources[0]):
        if ee == int(raw1) or ee == int(sources[-1]) or ee == int(e1):
            seg_pts, seg_src = pts, sources
        else:
            return False
    elif se == int(raw1) or se == int(sources[-1]):
        if ee == int(raw0) or ee == int(sources[0]) or ee == int(e0):
            seg_pts, seg_src = pts[::-1], [int(x) for x in sources[::-1]]
        else:
            return False
    else:
        return False
    _append_ring_segment(pts_acc, src_acc, seg_pts, seg_src)
    return True


def _append_layout_chord(
    pts_acc: List[np.ndarray],
    src_acc: List[int],
    vertices: np.ndarray,
    layout_curves: Sequence[IntersectionCurve],
    source_from: int,
    source_to: int,
) -> bool:
    """子孔内边界拓扑边：角点/汇交点间稀疏弦（非密集辐射折线）。"""
    p0 = _layout_endpoint_xyz(int(source_from), vertices, layout_curves)
    p1 = _layout_endpoint_xyz(int(source_to), vertices, layout_curves)
    if p0 is None or p1 is None:
        return False
    if float(np.linalg.norm(p0 - p1)) <= 1e-12:
        return True
    _append_ring_segment(
        pts_acc,
        src_acc,
        np.vstack([p0, p1]),
        [int(source_from), int(source_to)],
    )
    return True


def _assemble_curve_arc_subhole_ring(
    vertices: np.ndarray,
    arc_pts: np.ndarray,
    arc_vids: Sequence[int],
    corner_a: int,
    corner_b: int,
    leave_a: Tuple[int, IntersectionCurve, int],
    leave_b: Tuple[int, IntersectionCurve, int],
    layout_curves: Sequence[IntersectionCurve],
    label: int,
    *,
    junction_coincidence_tol: float = 1e-9,
) -> Optional[Tuple[np.ndarray, List[int]]]:
    """
    孔边弧 + 定稿交线拼闭合子环。

    子孔边界 = 孔边弧（mesh） + L2 恢复的特征折线（解析采样内点） + 汇交桥接。
    mesh↔virtual 段使用 ``_curve_polyline_from_endpoint`` 保留特征线采样，
    不再退化为角点↔汇交点直线弦。
    """
    idx_a, curve_a, ep_a = leave_a
    idx_b, curve_b, ep_b = leave_b
    pts_acc: List[np.ndarray] = []
    src_acc: List[int] = []
    _append_ring_segment(pts_acc, src_acc, arc_pts, arc_vids)

    if int(idx_a) == int(idx_b):
        if _curve_connects_corners_directly(curve_a, corner_a, corner_b):
            # Direct mesh-mesh carriers still need their recovered samples.
            # Source ids from _curve_polyline_bundle are deterministic per curve,
            # so adjacent subholes can share the same seam vertices in L4.
            seg_pts, seg_src = _curve_polyline_from_endpoint(
                curve_a, int(idx_a), int(corner_b)
            )
            _append_ring_segment(pts_acc, src_acc, seg_pts, seg_src)
            closed = np.asarray(pts_acc, dtype=np.float64)
            if closed.shape[0] < 3:
                return None
            return closed, src_acc
        mesh_virtual = _wedge_corners_mesh_virtual(corner_a, corner_b)
        if mesh_virtual is not None:
            _mesh_corner, virtual_corner = mesh_virtual
            if not _append_oriented_curve_polyline(
                pts_acc,
                src_acc,
                curve_a,
                int(idx_a),
                int(_mesh_corner),
                int(virtual_corner),
            ):
                return None
            closed = np.asarray(pts_acc, dtype=np.float64)
            if closed.shape[0] < 3:
                return None
            return closed, src_acc

    seg_b_pts, seg_b_src = _curve_polyline_from_endpoint(curve_b, idx_b, ep_b)
    j_b = int(seg_b_src[-1])
    seg_a_pts, seg_a_src = _curve_polyline_from_endpoint(curve_a, idx_a, ep_a)
    j_a = int(seg_a_src[-1])

    def _append_feature_interior(
        j_from: int,
        j_to: int,
        *,
        include_bridge: Optional[List[Tuple[int, int, int]]] = None,
    ) -> bool:
        if not _append_oriented_curve_polyline(
            pts_acc,
            src_acc,
            curve_b,
            int(idx_b),
            int(corner_b),
            int(j_from),
        ):
            return False
        if include_bridge is not None:
            for bridge_idx, bridge_start, _bridge_end in include_bridge:
                b_pts, b_src = _curve_polyline_from_endpoint(
                    layout_curves[int(bridge_idx)],
                    int(bridge_idx),
                    int(bridge_start),
                )
                _append_ring_segment(pts_acc, src_acc, b_pts, b_src)
        elif int(j_from) != int(j_to):
            direct_bridge = _find_virtual_bridge_curve_between(
                int(j_from),
                int(j_to),
                layout_curves,
                int(label),
            )
            if direct_bridge is not None:
                bridge_idx, bridge_start, bridge_end = direct_bridge
                if not _append_oriented_curve_polyline(
                    pts_acc,
                    src_acc,
                    layout_curves[int(bridge_idx)],
                    int(bridge_idx),
                    int(bridge_start),
                    int(bridge_end),
                ):
                    return False
            elif not _append_layout_chord(
                pts_acc,
                src_acc,
                vertices,
                layout_curves,
                int(j_from),
                int(j_to),
            ):
                return False
        if int(j_to) != int(corner_a):
            if not _append_oriented_curve_polyline(
                pts_acc,
                src_acc,
                curve_a,
                int(idx_a),
                int(j_to),
                int(corner_a),
            ):
                return False
        return True

    if int(j_a) != int(j_b):
        bridge = _junction_bridge_path(int(label), j_b, j_a, layout_curves)
        if bridge is not None:
            if not _append_feature_interior(j_b, j_a, include_bridge=bridge):
                return None
        elif _virtual_sources_spatially_coincident(
            int(j_a),
            int(j_b),
            layout_curves,
            float(junction_coincidence_tol),
        ):
            if not _append_feature_interior(j_b, j_a):
                return None
        else:
            if not _append_feature_interior(int(j_b), int(j_a)):
                return None
    else:
        if not _append_feature_interior(int(j_b), int(j_a)):
            return None

    closed = np.asarray(pts_acc, dtype=np.float64)
    if closed.shape[0] < 3:
        return None
    closed, src_acc = _merge_coincident_virtual_ring_vertices(
        closed,
        src_acc,
        tol=max(float(junction_coincidence_tol), 1e-6),
    )
    if closed.shape[0] < 3:
        return None
    return closed, src_acc


def _prepare_curve_arc_partition_subholes(
    vertices: np.ndarray,
    loop: Sequence[int],
    arcs: Sequence[BoundaryArc],
    curves: Sequence[IntersectionCurve],
    patch_surface_fits: Mapping[int, SurfaceFit],
    fill_classification: FillPatchClassification,
    boundary_vertex_labels: Mapping[int, Sequence[int]],
    feature_vertex_ids: Sequence[int],
) -> Tuple[List[PreparedSubhole], List[PartitionObstacle], Dict[str, object]]:
    """
    L3 楔形剖分：每个 active label 须通过命题 3/4 闸门后再构造 ``PreparedSubhole``。

    layout mesh 端点经 ``refine_endpoint`` 得 ``endpoint_remap`` 后，
    以有效特征点（P + 未改动的 L1 特征点）重切孔边弧，再作楔形剖分。
    """
    active_labels = sorted(int(x) for x in fill_classification.active_fill_labels)
    layout_curves = _filter_layout_curves(curves)
    partition_diag: Dict[str, object] = {"mode": "curve_arc_partition"}
    out: List[PreparedSubhole] = []
    obstacles: List[PartitionObstacle] = []

    hole_center = _loop_centroid(vertices, loop)
    loop_step = _mean_hole_loop_edge_len(vertices, loop)
    endpoint_remap, refinement_diags = _build_layout_endpoint_remap(
        layout_curves,
        loop=loop,
        vertices=vertices,
        boundary_vertex_labels=boundary_vertex_labels,
        feature_vertex_ids=feature_vertex_ids,
        loop_step=float(loop_step),
    )
    layout_for_wedge = layout_curves
    arcs_for_wedge = list(arcs)
    effective_vertices = _effective_feature_vertices_after_endpoint_remap(
        feature_vertex_ids,
        {},
    )
    if endpoint_remap:
        layout_for_wedge = _remap_layout_curves_for_endpoint_remap(
            layout_curves,
            endpoint_remap,
            vertices,
            patch_surface_fits,
            hole_center,
            loop_step,
        )
        effective_vertices = _effective_feature_vertices_after_endpoint_remap(
            feature_vertex_ids,
            endpoint_remap,
        )
    fp_positions = _loop_positions_for_vertices(loop, effective_vertices)
    demoted = {
        int(v) for v in fill_classification.inactive_feature_points
    }
    active_corner_vertices = _effective_feature_vertices_after_endpoint_remap(
        sorted(int(v) for v in fill_classification.active_feature_points),
        endpoint_remap,
    )
    active_corner_vertices = [
        int(v) for v in active_corner_vertices if int(v) not in demoted
    ]
    resplit_dropped_active_labels: List[int] = []
    if len(active_corner_vertices) >= 2:
        candidate_arcs = _repartition_boundary_arcs_for_corner_vertices(
            loop,
            arcs,
            active_corner_vertices,
        )
        candidate_labels = {int(arc.patch_label) for arc in candidate_arcs}
        active_label_set = {int(label) for label in active_labels}
        resplit_dropped_active_labels = sorted(active_label_set - candidate_labels)
        if not resplit_dropped_active_labels:
            arcs_for_wedge = candidate_arcs
    if endpoint_remap or refinement_diags or len(fp_positions) >= 2:
        partition_diag = {
            "mode": "curve_arc_partition",
            "endpoint_remap": {int(k): int(v) for k, v in endpoint_remap.items()},
            "endpoint_refinement": refinement_diags,
            "effective_feature_vertices": effective_vertices,
            "active_corner_vertices": active_corner_vertices,
        }
        if resplit_dropped_active_labels:
            partition_diag["resplit_rejected_missing_active_labels"] = (
                resplit_dropped_active_labels
            )
        if len(fp_positions) >= 2:
            partition_diag["resplit_boundary_arcs"] = [
                {
                    "label": int(arc.patch_label),
                    "v0": int(arc.vertex_indices[0]),
                    "v1": int(arc.vertex_indices[-1]),
                }
                for arc in arcs_for_wedge
            ]
    elif refinement_diags:
        partition_diag = {
            "mode": "curve_arc_partition",
            "endpoint_remap": {},
            "endpoint_refinement": refinement_diags,
        }

    layout_override = layout_for_wedge if endpoint_remap else None
    for label in active_labels:
        sub, obstacle = _wedge_partition_for_label(
            vertices,
            loop,
            arcs_for_wedge,
            layout_curves,
            patch_surface_fits,
            fill_classification,
            int(label),
            layout_curves_override=layout_override,
            endpoint_remap=endpoint_remap,
        )
        if obstacle is not None:
            obstacles.append(obstacle)
            continue
        if sub is not None:
            out.append(sub)

    if obstacles:
        return [], obstacles, partition_diag
    support_subholes: List[PreparedSubhole] = []
    support_bridge_diag: List[Dict[str, object]] = []
    covered_support_paths: Set[Tuple[int, ...]] = set()
    for _support_label, paths in sorted(fill_classification.degenerate_label_paths.items()):
        for raw_path in paths:
            path = tuple(int(v) for v in raw_path)
            if len(path) < 2 or path in covered_support_paths:
                continue
            a, b = int(path[0]), int(path[-1])
            for label in active_labels:
                if (
                    _layout_curve_mesh_bridge(layout_for_wedge, int(label), a, b) is None
                    and _layout_curve_mesh_fan_bridge(layout_for_wedge, int(label), a, b) is None
                ):
                    continue
                sub, obstacle = _wedge_partition_for_label(
                    vertices,
                    loop,
                    arcs_for_wedge,
                    layout_curves,
                    patch_surface_fits,
                    fill_classification,
                    int(label),
                    forced_corners=(a, b),
                    forced_arc=(
                        list(path),
                        vertices[np.asarray(path, dtype=np.int64)],
                    ),
                    layout_curves_override=layout_override,
                    closure_kind=CLOSURE_CURVE_ARC_PARTITION,
                    endpoint_remap=endpoint_remap,
                )
                if sub is None or obstacle is not None:
                    continue
                support_subholes.append(sub)
                covered_support_paths.add(path)
                support_bridge_diag.append(
                    {
                        "label": int(label),
                        "path": [int(v) for v in path],
                        "n_boundary_points": int(sub.closed_boundary_points.shape[0]),
                    }
                )
                break
    if support_subholes:
        out.extend(support_subholes)
        partition_diag["support_bridge_subholes"] = support_bridge_diag
    augmented = _augment_unpaired_fan_seams(
        out,
        vertices,
        loop,
        layout_for_wedge,
        patch_surface_fits,
        active_labels,
        closure_kind=CLOSURE_CURVE_ARC_PARTITION,
    )
    if len(augmented) > len(out):
        partition_diag["fan_seam_augmented_subholes"] = int(len(augmented) - len(out))
        out = augmented
    return out, [], partition_diag


def _arrangement_endpoint_sources_for_curve(
    curve: IntersectionCurve,
    curve_idx: int,
) -> Tuple[int, int]:
    e0, e1 = (int(curve.endpoint_vertex_indices[0]), int(curve.endpoint_vertex_indices[1]))
    # 保留显式 arrangement 汇交 source（如 -920000）；仅将 -1 等占位虚拟端点映射到 per-curve id。
    if e0 < 0 and int(e0) >= -900_000:
        e0 = -900_000 - 2 * int(curve_idx)
    if e1 < 0 and int(e1) >= -900_000:
        e1 = -900_000 - 2 * int(curve_idx) - 1
    return e0, e1


def _point_on_protected_boundary_edge(
    point: np.ndarray,
    vertices: np.ndarray,
    loop: Sequence[int],
    tol: float,
) -> bool:
    """Whether ``point`` lies on the interior of an original hole boundary edge."""
    p = np.asarray(point, dtype=np.float64).reshape(3)
    loop_list = [int(v) for v in loop]
    n = len(loop_list)
    if n < 2:
        return False
    for i in range(n):
        a = int(loop_list[i])
        b = int(loop_list[(i + 1) % n])
        p0 = np.asarray(vertices[a], dtype=np.float64)
        p1 = np.asarray(vertices[b], dtype=np.float64)
        edge = p1 - p0
        edge_len2 = float(np.dot(edge, edge))
        if edge_len2 <= 1e-24:
            continue
        t = float(np.dot(p - p0, edge) / edge_len2)
        edge_len = float(np.sqrt(edge_len2))
        margin = max(1e-6, float(tol) / max(edge_len, 1e-12))
        if t <= margin or t >= 1.0 - margin:
            continue
        q = p0 + t * edge
        if float(np.linalg.norm(p - q)) <= float(tol):
            return True
    return False


def _filter_curve_samples_on_protected_boundary(
    vertices: np.ndarray,
    loop: Sequence[int],
    curve: IntersectionCurve,
    loop_step: float,
) -> IntersectionCurve:
    """Remove interior curve samples that encroach on protected hole-boundary edges."""
    pts = np.asarray(curve.curve_points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] <= 2:
        return curve
    loop_pts = vertices[np.asarray(loop, dtype=np.int64)]
    diag = max(float(np.linalg.norm(np.ptp(loop_pts, axis=0))), 1e-12)
    tol = max(1e-8 * diag, 0.08 * float(loop_step or 0.0), 1e-9)
    keep = [0]
    for i in range(1, pts.shape[0] - 1):
        if _point_on_protected_boundary_edge(pts[i], vertices, loop, tol):
            continue
        keep.append(i)
    keep.append(pts.shape[0] - 1)
    if len(keep) == pts.shape[0]:
        return curve
    filtered = pts[np.asarray(keep, dtype=np.int64)]
    return replace(
        curve,
        curve_points=filtered,
        endpoints_on_boundary=np.vstack((filtered[0], filtered[-1])),
    )


def _resample_sparse_arrangement_curves(
    vertices: np.ndarray,
    loop: Sequence[int],
    curves: Sequence[IntersectionCurve],
    patch_surface_fits: Mapping[int, SurfaceFit],
) -> List[IntersectionCurve]:
    """
    Feature-terminated ``*_span`` curves may only store their two endpoints.
    Before L3 partition, resample them on the analytic surface pair so the
    subhole boundary contains the actual feature polyline instead of a chord.
    Interior samples are filtered against the original hole boundary edges:
    samples may approximate an internal feature curve, but they must not become
    new topological vertices on protected boundary edges.
    """
    loop_step = _mean_hole_loop_edge_len(vertices, loop)
    ref_step = 0.75 * float(loop_step or 0.0)
    hole_center = _loop_centroid(vertices, loop)
    out: List[IntersectionCurve] = []
    for curve in curves:
        pts = np.asarray(curve.curve_points, dtype=np.float64)
        is_vv_bridge = _is_virtual_bridge_curve(curve)
        e0_raw, e1_raw = (
            int(curve.endpoint_vertex_indices[0]),
            int(curve.endpoint_vertex_indices[1]),
        )
        is_sparse_virtual_curve = pts.ndim == 2 and pts.shape[0] == 2 and (
            e0_raw < 0 or e1_raw < 0
        )
        if pts.ndim != 2 or pts.shape[0] < 2:
            out.append(curve)
            continue
        pair = tuple(sorted((int(curve.patch_pair[0]), int(curve.patch_pair[1]))))
        if pair[0] not in patch_surface_fits or pair[1] not in patch_surface_fits:
            out.append(
                _filter_curve_samples_on_protected_boundary(
                    vertices,
                    loop,
                    curve,
                    loop_step,
                )
            )
            continue
        length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        if is_vv_bridge and pts.shape[0] >= 2 and length <= 1e-12:
            length = float(np.linalg.norm(pts[-1] - pts[0]))
        if ref_step <= 1e-15 or length <= 1.35 * ref_step:
            out.append(
                _filter_curve_samples_on_protected_boundary(
                    vertices,
                    loop,
                    curve,
                    loop_step,
                )
            )
            continue
        target_n = feature_curve_sample_count(length, ref_step)
        if not is_vv_bridge and pts.shape[0] != 2:
            out.append(
                _filter_curve_samples_on_protected_boundary(
                    vertices,
                    loop,
                    curve,
                    loop_step,
                )
            )
            continue
        if is_vv_bridge and pts.shape[0] >= target_n:
            out.append(
                _filter_curve_samples_on_protected_boundary(
                    vertices,
                    loop,
                    curve,
                    loop_step,
                )
            )
            continue
        resampled = recover_curve_between_points(
            patch_surface_fits[pair[0]],
            patch_surface_fits[pair[1]],
            pts[0],
            pts[-1],
            _feature_curve_guide_point(
                pts[0],
                pts[-1],
                hole_center,
                endpoint_vertex_indices=tuple(
                    int(x) for x in curve.endpoint_vertex_indices
                ),
            ),
            n_samples=target_n,
            min_samples=0,
            endpoint_vertex_indices=tuple(
                int(x) for x in curve.endpoint_vertex_indices
            ),
            intersection_sampling_reference_step=ref_step,
        )
        out.append(
            _filter_curve_samples_on_protected_boundary(
                vertices,
                loop,
                resampled,
                loop_step,
            )
        )
    return out



def _prepare_subholes(
    vertices: np.ndarray,
    faces_arr: np.ndarray,
    loop: Sequence[int],
    arcs: Sequence[BoundaryArc],
    unique_patch_count: int,
    boundary_vertex_labels: Mapping[int, Sequence[int]],
    curves: Sequence[IntersectionCurve],
    patch_surface_fits: Mapping[int, SurfaceFit],
    fill_classification: Optional[FillPatchClassification] = None,
) -> Tuple[List[PreparedSubhole], Optional[FeatureArrangement]]:
    """
    子孔装配入口（L3）。

    路由由 ``infer_fill_strategy(K, M)`` 单一裁决：
    - whole_loop → 单 label 孔环
    - opening_carrier → K>1 且 |M|=1，整环承载
    - curve_arc_partition → |M|>1，弧+交线剖分
    """
    if fill_classification is None:
        arrangement = FeatureArrangement()
        arrangement.diagnostics["subhole_partition"] = "no_fill_classification"
        arrangement.diagnostics["subhole_rejection"] = "no_fill_classification"
        return [], arrangement

    expected_labels = {
        int(x) for x in fill_classification.active_fill_labels
    }
    strategy = infer_fill_strategy(int(unique_patch_count), expected_labels)

    if strategy == FILL_STRATEGY_WHOLE_LOOP:
        return _prepare_single_patch_subholes(
            vertices, arcs, loop, patch_surface_fits
        ), None

    arrangement = FeatureArrangement()
    _attach_ownership_to_arrangement_diagnostics(arrangement, fill_classification)
    arrangement.diagnostics["expected_labels"] = sorted(expected_labels)
    arrangement.diagnostics["fill_strategy"] = strategy

    if strategy == FILL_STRATEGY_OPENING_CARRIER:
        prepared = _prepare_opening_carrier_subholes(
            vertices,
            loop,
            fill_classification,
            patch_surface_fits,
        )
        prepared = _filter_active_prepared_subholes(
            prepared,
            expected_labels=expected_labels,
        )
        ok, reason, fill_ready = _assess_prepared_subholes_fill_ready(
            prepared,
            expected_labels,
        )
        arrangement.diagnostics["fill_pipeline_stage"] = FILL_STAGE_OPENING_CARRIER
        arrangement.diagnostics["fill_ready"] = fill_ready
        arrangement.diagnostics["subhole_partition"] = (
            "opening_carrier_boundary" if ok else f"opening_carrier_rejected:{reason}"
        )
        arrangement.diagnostics["got_labels"] = sorted(
            {int(p.patch_label) for p in prepared}
        )
        if ok:
            return prepared, arrangement
        arrangement.diagnostics["subhole_rejection"] = str(reason)
        return [], arrangement

    prepared, l3_obstacles, partition_diag = _prepare_curve_arc_partition_subholes(
        vertices,
        loop,
        arcs,
        curves,
        patch_surface_fits,
        fill_classification,
        boundary_vertex_labels,
        sorted(
            int(x)
            for x in fill_classification.active_feature_points
        ),
    )
    arrangement.diagnostics["curve_arc_partition"] = dict(partition_diag)
    if l3_obstacles:
        arrangement.diagnostics["partition_obstacles_l3"] = [
            {"kind": o.kind, "label": o.label, "detail": o.detail}
            for o in l3_obstacles
        ]
        l3_detail = "; ".join(
            f"{o.kind} label={o.label}:{o.detail}" for o in l3_obstacles
        )
        arrangement.diagnostics["subhole_rejection"] = l3_detail
    prepared = _filter_active_prepared_subholes(
        prepared,
        expected_labels=expected_labels,
    )
    ok, reason, fill_ready = _assess_prepared_subholes_fill_ready(
        prepared,
        expected_labels,
    )
    arrangement.diagnostics["fill_pipeline_stage"] = FILL_STAGE_CURVE_ARC_PARTITION
    arrangement.diagnostics["fill_ready"] = fill_ready
    arrangement.diagnostics["subhole_partition"] = (
        "curve_arc_partition" if ok else f"curve_arc_partition_rejected:{reason}"
    )
    arrangement.diagnostics["expected_labels"] = sorted(expected_labels)
    arrangement.diagnostics["got_labels"] = sorted(
        {int(p.patch_label) for p in prepared}
    )
    if ok:
        return prepared, arrangement
    if not l3_obstacles:
        arrangement.diagnostics["subhole_rejection"] = str(reason)
    return [], arrangement


def _cells_from_prepared_subholes(
    prepared: Sequence[PreparedSubhole],
) -> List[PatchCell]:
    areas = [
        polygon_area_3d(np.asarray(item.closed_boundary_points, dtype=np.float64))
        for item in prepared
    ]
    max_area = max(areas) if areas else 0.0
    cells: List[PatchCell] = []
    for item, area in zip(prepared, areas):
        n_closed = int(np.asarray(item.closed_boundary_points, dtype=np.float64).shape[0])
        cell = PatchCell(
            patch_label=int(item.patch_label),
            boundary_arc_indices=[],
            segment_indices=[],
            closed_loop_vertex_ids=list(range(n_closed)),
            area=float(area),
        )
        validate_patch_cell(cell, max_area=max_area)
        cells.append(cell)
    return cells


def _filter_active_prepared_subholes(
    prepared: Sequence[PreparedSubhole],
    *,
    expected_labels: Optional[Set[int]] = None,
) -> List[PreparedSubhole]:
    keep_labels = (
        {int(x) for x in expected_labels} if expected_labels is not None else None
    )
    cells = _cells_from_prepared_subholes(prepared)
    areas = [
        polygon_area_3d(np.asarray(item.closed_boundary_points, dtype=np.float64))
        for item in prepared
    ]
    max_area = max(areas, default=0.0)
    active: List[PreparedSubhole] = []
    for item, cell, area in zip(prepared, cells, areas):
        if keep_labels is not None and int(item.patch_label) in keep_labels:
            active.append(item)
            continue
        if not cell.is_active:
            continue
        sources = [int(x) for x in item.boundary_sources]
        boundary_count = sum(1 for src in sources if src >= 0)
        virtual_count = sum(1 for src in sources if src < 0)
        if (
            len(prepared) > 1
            and max_area > 0.0
            and float(area) < 0.10 * float(max_area)
            and boundary_count <= 2
            and virtual_count >= 2
        ):
            continue
        active.append(item)
    return list(active) if active else list(prepared)


def _mean_hole_loop_edge_len(vertices: np.ndarray, loop: Sequence[int]) -> float:
    """孔洞环路上相邻顶点边的平均长度。"""
    n = len(loop)
    if n < 2:
        return 0.0
    s = 0.0
    for i in range(n):
        a, b = int(loop[i]), int(loop[(i + 1) % n])
        s += float(np.linalg.norm(vertices[a] - vertices[b]))
    return s / float(n)


def _ha_edge_key(u: int, v: int) -> Tuple[int, int]:
    u, v = int(u), int(v)
    return (u, v) if u < v else (v, u)


def _build_edge_to_face_indices(faces: np.ndarray) -> Dict[Tuple[int, int], List[int]]:
    out: Dict[Tuple[int, int], List[int]] = {}
    for fi in range(int(faces.shape[0])):
        a, b, c = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
        for u, v in ((a, b), (b, c), (c, a)):
            out.setdefault(_ha_edge_key(u, v), []).append(int(fi))
    return out


def _is_edge_in_triangle(a: int, b: int, tri: np.ndarray) -> bool:
    t = [int(tri[0]), int(tri[1]), int(tri[2])]
    for i in range(3):
        u, v = t[i], t[(i + 1) % 3]
        if u == a and v == b:
            return True
        if u == b and v == a:
            return True
    return False


def _common_edge_length_between_triangles(
    vertices: np.ndarray,
    tri0: np.ndarray,
    tri1: np.ndarray,
) -> Optional[float]:
    """两三角形若共享一条边，返回该边长。"""
    t0 = [int(tri0[0]), int(tri0[1]), int(tri0[2])]
    t1 = [int(tri1[0]), int(tri1[1]), int(tri1[2])]
    shared = set(t0) & set(t1)
    if len(shared) < 2:
        return None
    for ia in range(3):
        u, v = int(tri0[ia]), int(tri0[(ia + 1) % 3])
        if u in shared and v in shared and _is_edge_in_triangle(u, v, tri1):
            return float(np.linalg.norm(vertices[u] - vertices[v]))
    return None


def _mean_triangle_edge_length(vertices: np.ndarray, tri: np.ndarray) -> float:
    a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
    e0 = float(np.linalg.norm(vertices[a] - vertices[b]))
    e1 = float(np.linalg.norm(vertices[b] - vertices[c]))
    e2 = float(np.linalg.norm(vertices[c] - vertices[a]))
    return (e0 + e1 + e2) / 3.0


def _corner_intersection_sampling_reference(
    vertices: np.ndarray,
    faces: np.ndarray,
    edge2f: Dict[Tuple[int, int], List[int]],
    loop: Sequence[int],
    corner_v: int,
) -> Optional[float]:
    """
    角点处沿孔洞环的前后边界边各取一邻接三角形；若二者有公共边，
    则以该公共边长作为交线采样步长的参考（更贴合局部网格尺度）。
    否则退化为两三角形各自平均边长的算术平均。
    """
    n = len(loop)
    if n < 3:
        return None
    idx: Optional[int] = None
    for i in range(n):
        if int(loop[i]) == int(corner_v):
            idx = i
            break
    if idx is None:
        return None
    v_prev = int(loop[(idx - 1) % n])
    v_c = int(corner_v)
    v_next = int(loop[(idx + 1) % n])
    fis0 = edge2f.get(_ha_edge_key(v_prev, v_c))
    fis1 = edge2f.get(_ha_edge_key(v_c, v_next))
    if not fis0 or not fis1:
        return None
    f0, f1 = int(fis0[0]), int(fis1[0])
    tri0 = faces[f0]
    tri1 = faces[f1]
    if f0 == f1:
        return _mean_triangle_edge_length(vertices, tri0)
    common_len = _common_edge_length_between_triangles(vertices, tri0, tri1)
    if common_len is not None and common_len > 1e-15:
        return common_len
    m0 = _mean_triangle_edge_length(vertices, tri0)
    m1 = _mean_triangle_edge_length(vertices, tri1)
    return float(0.5 * (m0 + m1))


def _all_pair_matchings(items: Sequence[int]) -> List[List[Tuple[int, int]]]:
    vals = [int(x) for x in items]
    if not vals:
        return [[]]
    first = vals[0]
    out: List[List[Tuple[int, int]]] = []
    for i in range(1, len(vals)):
        second = vals[i]
        rest = vals[1:i] + vals[i + 1 :]
        for sub in _all_pair_matchings(rest):
            out.append([(first, second)] + sub)
    return out


def _segments_cross_2d(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> bool:
    def orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    eps = 1e-12
    return (o1 * o2 < -eps) and (o3 * o4 < -eps)


def _principal_project_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    centered = pts - np.mean(pts, axis=0, keepdims=True)
    if pts.shape[0] < 3:
        return centered[:, :2]
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh[:2].T
    return centered @ basis


def _adjacent_patch_pairs_from_boundary(
    effective_vertex_labels: Mapping[int, Sequence[int]],
    arcs: Sequence[BoundaryArc],
    patch_surface_fits: Mapping[int, SurfaceFit],
) -> List[Tuple[int, int]]:
    _ = effective_vertex_labels
    pairs: Set[Tuple[int, int]] = set()
    n_arcs = len(arcs)
    for i in range(n_arcs):
        a = int(arcs[i].patch_label)
        b = int(arcs[(i + 1) % n_arcs].patch_label)
        if a == b:
            continue
        if a not in patch_surface_fits or b not in patch_surface_fits:
            continue
        fa = patch_surface_fits[a]
        fb = patch_surface_fits[b]
        if not is_analytic_surface_type(fa.surface_type) or not is_analytic_surface_type(
            fb.surface_type
        ):
            continue
        pairs.add(tuple(sorted((a, b))))
    return sorted(pairs)


def _arc_endpoint_corners_for_pair(
    arcs: Sequence[BoundaryArc],
    pair: Tuple[int, int],
    vertex_labels: Mapping[int, Sequence[int]],
) -> List[int]:
    """弧端点上、同时属于 pair 两 patch 的 transition corner（优先于全环最近点）。"""
    a, b = int(pair[0]), int(pair[1])
    out: Set[int] = set()
    for arc in arcs:
        if int(arc.patch_label) not in (a, b):
            continue
        for vid in (int(arc.vertex_indices[0]), int(arc.vertex_indices[-1])):
            labels = {int(x) for x in vertex_labels.get(vid, [])}
            if a in labels and b in labels:
                out.add(int(vid))
    return sorted(out)


def _build_arc_corner_hints(
    arcs: Sequence[BoundaryArc],
    pairs: Sequence[Tuple[int, int]],
    vertex_labels: Mapping[int, Sequence[int]],
) -> Dict[Tuple[int, int], Tuple[int, ...]]:
    hints: Dict[Tuple[int, int], Tuple[int, ...]] = {}
    for pair in pairs:
        key = tuple(sorted((int(pair[0]), int(pair[1]))))
        corners = _arc_endpoint_corners_for_pair(arcs, key, vertex_labels)
        if corners:
            hints[key] = tuple(corners)
    return hints


def _polyline_arclengths(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 2:
        return np.zeros(pts.shape[0], dtype=np.float64)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(seg)))


def _project_point_to_polyline(
    point: np.ndarray,
    polyline: np.ndarray,
) -> Tuple[float, float, np.ndarray]:
    pts = np.asarray(polyline, dtype=np.float64).reshape(-1, 3)
    p = np.asarray(point, dtype=np.float64).reshape(3)
    if pts.shape[0] == 0:
        return float("inf"), 0.0, p.copy()
    if pts.shape[0] == 1:
        return float(np.linalg.norm(p - pts[0])), 0.0, pts[0].copy()
    seg = pts[1:] - pts[:-1]
    lengths = np.linalg.norm(seg, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    best_dist = float("inf")
    best_s = 0.0
    best_point = pts[0].copy()
    for i, length in enumerate(lengths):
        length = float(length)
        if length <= 1e-12:
            continue
        t = float(np.dot(p - pts[i], seg[i]) / (length * length))
        t = min(1.0, max(0.0, t))
        q = pts[i] + t * seg[i]
        dist = float(np.linalg.norm(p - q))
        if dist < best_dist:
            best_dist = dist
            best_s = float(cumulative[i] + t * length)
            best_point = q
    return best_dist, best_s, best_point


def _hole_curve_distance_tol(vertices: np.ndarray, reference_step: float) -> float:
    diag = max(float(_bbox_diag(vertices)), 1.0)
    step = float(reference_step) if reference_step > 1e-15 else 0.0
    return max(1e-8 * diag, 0.12 * step, 1e-9)


def _merge_param_intervals(
    intervals: Sequence[Tuple[float, float]],
    gap: float,
) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    iv = sorted((min(a, b), max(a, b)) for a, b in intervals)
    out: List[Tuple[float, float]] = [iv[0]]
    for a, b in iv[1:]:
        la, lb = out[-1]
        if a <= lb + gap:
            out[-1] = (la, max(lb, b))
        else:
            out.append((a, b))
    return out


def _subtract_param_intervals(
    full: Tuple[float, float],
    covered: Sequence[Tuple[float, float]],
    gap: float,
) -> List[Tuple[float, float]]:
    t0, t1 = full
    missing: List[Tuple[float, float]] = [(t0, t1)]
    for ca, cb in covered:
        next_missing: List[Tuple[float, float]] = []
        for ma, mb in missing:
            if cb <= ma + gap or ca >= mb - gap:
                next_missing.append((ma, mb))
                continue
            if ma < ca - gap:
                next_missing.append((ma, min(mb, ca)))
            if cb + gap < mb:
                next_missing.append((max(ma, cb), mb))
        missing = next_missing
    return [(a, b) for a, b in missing if b - a > gap]


def _base_curve_kind(curve_kind: str) -> str:
    kind = str(curve_kind)
    for suffix in ("_recover_recover", "_span", "_recover"):
        if kind.endswith(suffix):
            return kind[: -len(suffix)]
    return kind


def _feature_vertices_from_arcs(arcs: Sequence[BoundaryArc]) -> Set[int]:
    out: Set[int] = set()
    for arc in arcs:
        verts = arc.vertex_indices
        if not verts:
            continue
        out.add(int(verts[0]))
        out.add(int(verts[-1]))
    return out


def _edge_chain_covered_intervals_on_curve(
    vertices: np.ndarray,
    vertex_chain: Sequence[int],
    curve_pts: np.ndarray,
    t_bounds: Tuple[float, float],
    tol: float,
) -> List[Tuple[float, float]]:
    t_min, t_max = t_bounds
    covered: List[Tuple[float, float]] = []
    chain = [int(v) for v in vertex_chain]
    for i in range(len(chain) - 1):
        u, v = chain[i], chain[i + 1]
        p0 = np.asarray(vertices[u], dtype=np.float64)
        p1 = np.asarray(vertices[v], dtype=np.float64)
        on_edge = True
        for t in np.linspace(0.0, 1.0, 5):
            q = (1.0 - t) * p0 + t * p1
            if _project_point_to_polyline(q, curve_pts)[0] > tol:
                on_edge = False
                break
        if not on_edge:
            continue
        _du, su, _ = _project_point_to_polyline(p0, curve_pts)
        _dv, sv, _ = _project_point_to_polyline(p1, curve_pts)
        a, b = sorted((float(su), float(sv)))
        a = max(t_min, a)
        b = min(t_max, b)
        if b - a > tol:
            covered.append((a, b))
    return covered


def _covered_intervals_on_curve(
    vertices: np.ndarray,
    loop: Sequence[int],
    arcs: Sequence[BoundaryArc],
    patch_pair: Tuple[int, int],
    curve_pts: np.ndarray,
    t_bounds: Tuple[float, float],
    tol: float,
) -> List[Tuple[float, float]]:
    covered = _loop_covered_intervals_on_curve(
        vertices, loop, curve_pts, t_bounds, tol
    )
    pair_set = {int(patch_pair[0]), int(patch_pair[1])}
    extra: List[Tuple[float, float]] = []
    for arc in arcs:
        if int(arc.patch_label) not in pair_set:
            continue
        extra.extend(
            _edge_chain_covered_intervals_on_curve(
                vertices,
                arc.vertex_indices,
                curve_pts,
                t_bounds,
                tol,
            )
        )
    if extra:
        covered = _merge_param_intervals([*covered, *extra], tol)
    return covered


def _snap_missing_endpoint_vertex(
    vertices: np.ndarray,
    loop: Sequence[int],
    point: np.ndarray,
    tol: float,
    feature_vertices: Set[int],
    prefer_e0: int,
    prefer_e1: int,
) -> int:
    p = np.asarray(point, dtype=np.float64).reshape(3)
    candidates: List[Tuple[int, float, int]] = []
    for fv in feature_vertices:
        dist = float(np.linalg.norm(p - vertices[int(fv)]))
        if dist <= tol:
            candidates.append((2, dist, int(fv)))
    for vid in (int(prefer_e0), int(prefer_e1)):
        if vid < 0:
            continue
        dist = float(np.linalg.norm(p - vertices[vid]))
        if dist <= tol:
            candidates.append((1, dist, vid))
    near = _nearest_loop_vertex_to_point(vertices, loop, p, tol)
    if near >= 0:
        dist = float(np.linalg.norm(p - vertices[int(near)]))
        candidates.append((0, dist, int(near)))
    if candidates:
        _prio, _dist, vid = min(candidates, key=lambda item: (-int(item[0]), float(item[1])))
        return int(vid)
    return int(prefer_e0) if float(np.linalg.norm(p - vertices[prefer_e0])) <= float(
        np.linalg.norm(p - vertices[prefer_e1])
    ) else int(prefer_e1)


def _loop_covered_intervals_on_curve(
    vertices: np.ndarray,
    loop: Sequence[int],
    curve_pts: np.ndarray,
    t_bounds: Tuple[float, float],
    tol: float,
) -> List[Tuple[float, float]]:
    t_min, t_max = t_bounds
    covered: List[Tuple[float, float]] = []
    n = len(loop)
    for i in range(n):
        u, v = int(loop[i]), int(loop[(i + 1) % n])
        p0 = np.asarray(vertices[u], dtype=np.float64)
        p1 = np.asarray(vertices[v], dtype=np.float64)
        on_edge = True
        for t in np.linspace(0.0, 1.0, 5):
            q = (1.0 - t) * p0 + t * p1
            if _project_point_to_polyline(q, curve_pts)[0] > tol:
                on_edge = False
                break
        if not on_edge:
            continue
        _du, su, _ = _project_point_to_polyline(p0, curve_pts)
        _dv, sv, _ = _project_point_to_polyline(p1, curve_pts)
        a, b = sorted((float(su), float(sv)))
        a = max(t_min, a)
        b = min(t_max, b)
        if b - a > tol:
            covered.append((a, b))
    return _merge_param_intervals(covered, tol)


def _nearest_loop_vertex_to_point(
    vertices: np.ndarray,
    loop: Sequence[int],
    point: np.ndarray,
    tol: float,
) -> int:
    p = np.asarray(point, dtype=np.float64).reshape(3)
    best_vid = -1
    best_dist = float("inf")
    for vid in loop:
        dist = float(np.linalg.norm(p - vertices[int(vid)]))
        if dist < best_dist:
            best_dist = dist
            best_vid = int(vid)
    if best_dist <= tol:
        return best_vid
    return -1


def _point_on_curve_at_param(
    curve_pts: np.ndarray,
    cumulative: np.ndarray,
    t: float,
) -> np.ndarray:
    total = float(cumulative[-1])
    t = min(max(float(t), 0.0), total)
    for j in range(1, len(cumulative)):
        if float(cumulative[j]) >= t - 1e-12:
            s0, s1 = float(cumulative[j - 1]), float(cumulative[j])
            if s1 - s0 <= 1e-12:
                return np.asarray(curve_pts[j], dtype=np.float64).reshape(3).copy()
            u = (t - s0) / (s1 - s0)
            return np.asarray(
                curve_pts[j - 1] + u * (curve_pts[j] - curve_pts[j - 1]),
                dtype=np.float64,
            ).reshape(3)
    return np.asarray(curve_pts[-1], dtype=np.float64).reshape(3).copy()


def _prefer_pair_corner_span_curve(
    e0: int,
    e1: int,
    pair: Tuple[int, int],
    feature_vertices: Set[int],
    boundary_vertex_labels: Mapping[int, Sequence[int]],
) -> bool:
    """两端均为 patch 对双标签角点时，避免被单标签边界点错误截断。"""
    if int(e0) not in feature_vertices or int(e1) not in feature_vertices:
        return False
    labels0 = _vertex_label_set(boundary_vertex_labels, int(e0))
    labels1 = _vertex_label_set(boundary_vertex_labels, int(e1))
    return _vertex_has_patch_pair(labels0, pair) and _vertex_has_patch_pair(labels1, pair)


def _refine_one_recovered_curve(
    curve: IntersectionCurve,
    vertices: np.ndarray,
    loop: Sequence[int],
    arcs: Sequence[BoundaryArc],
    patch_surface_fits: Mapping[int, SurfaceFit],
    hole_center: np.ndarray,
    ref_step: Optional[float],
    tol: float,
    feature_vertices: Set[int],
    boundary_vertex_labels: Mapping[int, Sequence[int]],
) -> List[IntersectionCurve]:
    """
    恢复阶段：去掉落在孔边界上的内部采样点，并按「孔边未覆盖」区间重采样。

    输出：
    - 特征点间端点折线（仅两端，供大面子孔引用整段 γ）
    - 若干条仅覆盖 missing 段的有采样交线（端点为环上 mesh 顶点）
    """
    e0, e1 = (
        int(curve.endpoint_vertex_indices[0]),
        int(curve.endpoint_vertex_indices[1]),
    )
    if e0 < 0 or e1 < 0:
        return [curve]

    pair = tuple(sorted((int(curve.patch_pair[0]), int(curve.patch_pair[1]))))
    if pair[0] not in patch_surface_fits or pair[1] not in patch_surface_fits:
        return [curve]

    curve_pts = np.asarray(curve.curve_points, dtype=np.float64)
    if curve_pts.shape[0] < 2:
        return [curve]

    cumulative = _polyline_arclengths(curve_pts)
    total = float(cumulative[-1])
    s0 = _project_point_to_polyline(vertices[e0], curve_pts)[1]
    s1 = _project_point_to_polyline(vertices[e1], curve_pts)[1]
    t_min, t_max = (min(float(s0), float(s1)), max(float(s0), float(s1)))
    t_min = max(0.0, t_min)
    t_max = min(total, t_max)
    full = (t_min, t_max)
    if t_max - t_min <= tol:
        return [curve]

    curve_tol = max(
        tol,
        _hole_curve_distance_tol(vertices, _polyline_target_spacing(curve_pts)),
    )
    covered = _covered_intervals_on_curve(
        vertices, loop, arcs, pair, curve_pts, full, curve_tol
    )
    missing = _subtract_param_intervals(full, covered, curve_tol)

    fa = patch_surface_fits[int(pair[0])]
    fb = patch_surface_fits[int(pair[1])]
    out: List[IntersectionCurve] = []
    base_kind = _base_curve_kind(curve.curve_kind)

    if (
        missing
        and _prefer_pair_corner_span_curve(
            e0,
            e1,
            pair,
            feature_vertices,
            boundary_vertex_labels,
        )
    ):
        missing_len = sum(float(ib) - float(ia) for ia, ib in missing)
        if missing_len + curve_tol < t_max - t_min:
            start = np.asarray(vertices[e0], dtype=np.float64)
            end = np.asarray(vertices[e1], dtype=np.float64)
            guide = _feature_curve_guide_point(
                start, end, hole_center, endpoint_vertex_indices=(e0, e1)
            )
            span_curve = recover_curve_between_points(
                fa,
                fb,
                start,
                end,
                guide,
                endpoint_vertex_indices=(e0, e1),
                intersection_sampling_reference_step=ref_step,
                min_samples=0,
            )
            return [
                replace(
                    span_curve,
                    curve_kind=f"{base_kind}_span",
                )
            ]

    span_pts = np.vstack(
        (
            np.asarray(vertices[e0], dtype=np.float64).reshape(1, 3),
            np.asarray(vertices[e1], dtype=np.float64).reshape(1, 3),
        )
    )
    out.append(
        replace(
            curve,
            curve_points=span_pts,
            endpoints_on_boundary=span_pts.copy(),
            curve_kind=f"{base_kind}_span",
            endpoint_vertex_indices=(e0, e1),
        )
    )

    for ia, ib in missing:
        p_a = _point_on_curve_at_param(curve_pts, cumulative, ia)
        p_b = _point_on_curve_at_param(curve_pts, cumulative, ib)
        vid_a = _snap_missing_endpoint_vertex(
            vertices, loop, p_a, curve_tol, feature_vertices, e0, e1
        )
        vid_b = _snap_missing_endpoint_vertex(
            vertices, loop, p_b, curve_tol, feature_vertices, e0, e1
        )
        if int(vid_a) == int(vid_b):
            continue
        guide = _feature_curve_guide_point(
            vertices[vid_a],
            vertices[vid_b],
            hole_center,
            endpoint_vertex_indices=(int(vid_a), int(vid_b)),
        )
        out.append(
            recover_curve_between_points(
                fa,
                fb,
                np.asarray(vertices[vid_a], dtype=np.float64),
                np.asarray(vertices[vid_b], dtype=np.float64),
                guide,
                endpoint_vertex_indices=(int(vid_a), int(vid_b)),
                intersection_sampling_reference_step=ref_step,
            )
        )
        out[-1] = replace(
            out[-1],
            curve_kind=f"{base_kind}_recover",
        )

    if len(out) == 1 and missing:
        return out
    if not missing:
        return [out[0]]
    return out


def _refine_recovered_curves_boundary_aware(
    vertices: np.ndarray,
    loop: Sequence[int],
    arcs: Sequence[BoundaryArc],
    curves: Sequence[IntersectionCurve],
    patch_surface_fits: Mapping[int, SurfaceFit],
    hole_center: np.ndarray,
    loop_sampling_fallback: Optional[float],
    boundary_vertex_labels: Mapping[int, Sequence[int]],
) -> List[IntersectionCurve]:
    """恢复后处理：不在孔边界上重复采样，按 missing 段输出交线。"""
    ref_step = (
        float(loop_sampling_fallback)
        if loop_sampling_fallback is not None and loop_sampling_fallback > 1e-15
        else _polyline_target_spacing(
            np.asarray(
                curves[0].curve_points if curves else vertices[:1],
                dtype=np.float64,
            )
        )
    )
    if ref_step <= 1e-15:
        ref_step = _mean_hole_loop_edge_len(vertices, loop)
    tol = _hole_curve_distance_tol(vertices, ref_step)
    feature_vertices = _feature_vertices_from_arcs(arcs)
    out: List[IntersectionCurve] = []
    for curve in curves:
        if str(curve.curve_kind).endswith("_span") or str(curve.curve_kind).endswith("_recover"):
            out.append(curve)
            continue
        out.extend(
            _refine_one_recovered_curve(
                curve,
                vertices,
                loop,
                arcs,
                patch_surface_fits,
                hole_center,
                ref_step,
                tol,
                feature_vertices,
                boundary_vertex_labels,
            )
        )
    return out


def _dedupe_curves_by_pair(
    curves: Sequence[IntersectionCurve],
) -> List[IntersectionCurve]:
    """同 patch pair 可有多条交线，按端点对去重而非按 pair 合并。"""

    def score(c: IntersectionCurve) -> Tuple[int, int, int, int]:
        e0, e1 = (int(c.endpoint_vertex_indices[0]), int(c.endpoint_vertex_indices[1]))
        both = e0 >= 0 and e1 >= 0
        kind = str(c.curve_kind)
        is_span = kind.endswith("_span")
        conf = {"high": 3, "medium": 2, "low": 1}.get(str(c.curve_confidence), 0)
        n_interior = max(0, len(c.curve_points) - 2)
        return (1 if is_span else 0, 1 if both else 0, conf, -n_interior)

    def dedupe_key(c: IntersectionCurve) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        pair = tuple(sorted((int(c.patch_pair[0]), int(c.patch_pair[1]))))
        e0, e1 = (int(c.endpoint_vertex_indices[0]), int(c.endpoint_vertex_indices[1]))
        if e0 >= 0 and e1 >= 0:
            return pair, tuple(sorted((e0, e1)))
        pts = np.asarray(c.curve_points, dtype=np.float64)
        if pts.ndim == 2 and pts.shape[0] >= 2:
            scale = max(float(np.linalg.norm(np.ptp(pts, axis=0))), 1.0)
            q0 = tuple(int(round(float(x) / (1e-7 * scale))) for x in pts[0])
            q1 = tuple(int(round(float(x) / (1e-7 * scale))) for x in pts[-1])
            return pair, (hash((e0, q0)), hash((e1, q1)))
        return pair, (e0, e1)

    best: Dict[Tuple[Tuple[int, int], Tuple[int, int]], IntersectionCurve] = {}
    for curve in curves:
        key = dedupe_key(curve)
        prev = best.get(key)
        if prev is None or score(curve) > score(prev):
            best[key] = curve
    return [best[k] for k in sorted(best)]


def _resample_curves_min_points(
    vertices: np.ndarray,
    patch_surface_fits: Mapping[int, SurfaceFit],
    curves: Sequence[IntersectionCurve],
    hole_center: np.ndarray,
    loop_sampling_fallback: Optional[float],
    *,
    min_pts: int = 3,
) -> List[IntersectionCurve]:
    out: List[IntersectionCurve] = []
    for curve in curves:
        n_pts = len(curve.curve_points)
        e0, e1 = (
            int(curve.endpoint_vertex_indices[0]),
            int(curve.endpoint_vertex_indices[1]),
        )
        if n_pts >= min_pts or e0 < 0 or e1 < 0:
            out.append(curve)
            continue
        pair = tuple(sorted((int(curve.patch_pair[0]), int(curve.patch_pair[1]))))
        if pair[0] not in patch_surface_fits or pair[1] not in patch_surface_fits:
            out.append(curve)
            continue
        start = np.asarray(vertices[e0], dtype=np.float64)
        end = np.asarray(vertices[e1], dtype=np.float64)
        guide = _feature_curve_guide_point(
            start, end, hole_center, endpoint_vertex_indices=(e0, e1)
        )
        out.append(
            recover_curve_between_points(
                patch_surface_fits[pair[0]],
                patch_surface_fits[pair[1]],
                start,
                end,
                guide,
                endpoint_vertex_indices=(e0, e1),
                intersection_sampling_reference_step=loop_sampling_fallback,
            )
        )
    return out


def _supplement_boundary_arc_endpoint_span_curves(
    vertices: np.ndarray,
    arcs: Sequence[BoundaryArc],
    curves: Sequence[IntersectionCurve],
    patch_surface_fits: Mapping[int, SurfaceFit],
    boundary_vertex_labels: Mapping[int, Sequence[int]],
    hole_center: np.ndarray,
    loop_sampling_fallback: Optional[float],
    active_labels: Set[int],
) -> List[IntersectionCurve]:
    """由边界弧两端的双标签角点补齐 mesh-mesh 交线。"""
    curves = _collapse_single_pair_virtual_fans_to_span_curves(
        vertices,
        curves,
        patch_surface_fits,
        hole_center,
        loop_sampling_fallback,
        active_labels,
    )
    # 若已有 virtual arrangement，说明 L2 已通过后求交/汇交证书表达拓扑；
    # 不再额外补 mesh-mesh span，避免破坏 fan 结构。
    for curve in curves:
        e0, e1 = (
            int(curve.endpoint_vertex_indices[0]),
            int(curve.endpoint_vertex_indices[1]),
        )
        if e0 < 0 or e1 < 0:
            return list(curves)
    out = list(curves)
    existing: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()
    for curve in out:
        e0, e1 = (
            int(curve.endpoint_vertex_indices[0]),
            int(curve.endpoint_vertex_indices[1]),
        )
        if e0 < 0 or e1 < 0:
            continue
        pair = tuple(sorted((int(curve.patch_pair[0]), int(curve.patch_pair[1]))))
        existing.add((pair, tuple(sorted((e0, e1)))))

    for arc in arcs:
        vids = [int(v) for v in arc.vertex_indices]
        if len(vids) < 2:
            continue
        label = int(arc.patch_label)
        if label not in active_labels:
            continue
        a, b = int(vids[0]), int(vids[-1])
        labels_a = _vertex_label_set(boundary_vertex_labels, a)
        labels_b = _vertex_label_set(boundary_vertex_labels, b)
        for other in sorted((labels_a & labels_b & active_labels) - {label}):
            pair = tuple(sorted((int(label), int(other))))
            if pair[0] not in patch_surface_fits or pair[1] not in patch_surface_fits:
                continue
            key = (pair, tuple(sorted((a, b))))
            if key in existing:
                continue
            start = np.asarray(vertices[a], dtype=np.float64)
            end = np.asarray(vertices[b], dtype=np.float64)
            curve = recover_curve_between_points(
                patch_surface_fits[pair[0]],
                patch_surface_fits[pair[1]],
                start,
                end,
                _feature_curve_guide_point(
                    start, end, hole_center, endpoint_vertex_indices=(a, b)
                ),
                endpoint_vertex_indices=(a, b),
                intersection_sampling_reference_step=loop_sampling_fallback,
                min_samples=0,
            )
            out.append(replace(curve, curve_kind=f"{_base_curve_kind(curve.curve_kind)}_span"))
            existing.add(key)
    return out


def _collapse_single_pair_virtual_fans_to_span_curves(
    vertices: np.ndarray,
    curves: Sequence[IntersectionCurve],
    patch_surface_fits: Mapping[int, SurfaceFit],
    hole_center: np.ndarray,
    loop_sampling_fallback: Optional[float],
    active_labels: Set[int],
) -> List[IntersectionCurve]:
    """
    A two-active-patch hole has only one real active-active feature line.
    If arrangement split that line into two mesh->virtual leaves sharing a
    virtual point, the virtual point is not a topological junction; recover the
    direct mesh-mesh span between the two active feature points.
    """
    active = {int(x) for x in active_labels}
    if len(active) != 2:
        return list(curves)
    active_pair = tuple(sorted(active))
    if active_pair[0] not in patch_surface_fits or active_pair[1] not in patch_surface_fits:
        return list(curves)

    by_virtual: Dict[int, List[Tuple[int, IntersectionCurve, int]]] = defaultdict(list)
    blocked_indices: Set[int] = set()
    for idx, curve in enumerate(curves):
        pair = tuple(sorted((int(curve.patch_pair[0]), int(curve.patch_pair[1]))))
        if pair != active_pair:
            return list(curves)
        e0, e1 = _arrangement_endpoint_sources_for_curve(curve, int(idx))
        if int(e0) >= 0 and int(e1) < 0:
            by_virtual[int(e1)].append((int(idx), curve, int(e0)))
            blocked_indices.add(int(idx))
        elif int(e1) >= 0 and int(e0) < 0:
            by_virtual[int(e0)].append((int(idx), curve, int(e1)))
            blocked_indices.add(int(idx))
        elif int(e0) < 0 or int(e1) < 0:
            return list(curves)

    replacements: List[IntersectionCurve] = []
    consumed: Set[int] = set()
    for virtual, leaves in sorted(by_virtual.items()):
        if len(leaves) != 2:
            return list(curves)
        mesh_endpoints = [int(leaves[0][2]), int(leaves[1][2])]
        if mesh_endpoints[0] == mesh_endpoints[1]:
            return list(curves)
        start = np.asarray(vertices[mesh_endpoints[0]], dtype=np.float64)
        end = np.asarray(vertices[mesh_endpoints[1]], dtype=np.float64)
        fit_a = patch_surface_fits[active_pair[0]]
        fit_b = patch_surface_fits[active_pair[1]]
        if _span_surface_fit_unreliable(fit_a) or _span_surface_fit_unreliable(fit_b):
            span = _linear_endpoint_span_curve(
                active_pair,
                start,
                end,
                (mesh_endpoints[0], mesh_endpoints[1]),
                fit_a,
                fit_b,
                loop_sampling_fallback,
            )
        else:
            span = recover_curve_between_points(
                fit_a,
                fit_b,
                start,
                end,
                _feature_curve_guide_point(
                    start,
                    end,
                    hole_center,
                    endpoint_vertex_indices=(mesh_endpoints[0], mesh_endpoints[1]),
                ),
                endpoint_vertex_indices=(mesh_endpoints[0], mesh_endpoints[1]),
                intersection_sampling_reference_step=loop_sampling_fallback,
                min_samples=0,
            )
        replacements.append(
            replace(span, curve_kind=f"{_base_curve_kind(span.curve_kind)}_span")
        )
        consumed.update(int(idx) for idx, _curve, _mesh in leaves)

    if not replacements or consumed != blocked_indices:
        return list(curves)
    out = [curve for idx, curve in enumerate(curves) if int(idx) not in consumed]
    out.extend(replacements)
    return out


def _span_surface_fit_unreliable(fit: SurfaceFit) -> bool:
    diag = max(float(_fit_support_diag(fit)), 1e-12)
    return float(fit.fit_residual) / diag > 1e-3


def _linear_endpoint_span_curve(
    pair: Tuple[int, int],
    start: np.ndarray,
    end: np.ndarray,
    endpoint_vertex_indices: Tuple[int, int],
    fit_a: SurfaceFit,
    fit_b: SurfaceFit,
    loop_sampling_fallback: Optional[float],
) -> IntersectionCurve:
    start = np.asarray(start, dtype=np.float64).reshape(3)
    end = np.asarray(end, dtype=np.float64).reshape(3)
    length = float(np.linalg.norm(end - start))
    step = float(loop_sampling_fallback or 0.0)
    n = 2 if step <= 1e-15 else max(2, int(np.ceil(length / step)) + 1)
    pts = np.linspace(start, end, int(n), dtype=np.float64)
    return IntersectionCurve(
        patch_pair=(int(pair[0]), int(pair[1])),
        curve_kind="line",
        curve_points=pts,
        endpoints_on_boundary=np.vstack((pts[0], pts[-1])),
        curve_confidence="low",
        source_surface_types=(str(fit_a.surface_type), str(fit_b.surface_type)),
        endpoint_vertex_indices=(
            int(endpoint_vertex_indices[0]),
            int(endpoint_vertex_indices[1]),
        ),
    )


def _analytic_curve_line_frame(
    ac: AnalyticCurve,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if ac.kind != "line" or ac.line_point is None or ac.line_dir is None:
        return None
    lp = np.asarray(ac.line_point, dtype=np.float64)
    ld = np.asarray(ac.line_dir, dtype=np.float64)
    ln = float(np.linalg.norm(ld))
    if ln < 1e-15:
        return None
    return lp, ld / ln


def _vertex_label_set(
    boundary_vertex_labels: Mapping[int, Sequence[int]],
    vertex: int,
) -> Set[int]:
    return {int(x) for x in boundary_vertex_labels.get(int(vertex), [])}


def _vertex_has_patch_pair(
    labels: Set[int],
    pair: Tuple[int, int],
) -> bool:
    a_label, b_label = int(pair[0]), int(pair[1])
    return a_label in labels and b_label in labels


def _arc_label_runs(arcs: Sequence[BoundaryArc]) -> Dict[int, List[BoundaryArc]]:
    grouped: Dict[int, List[BoundaryArc]] = defaultdict(list)
    for arc in arcs:
        grouped[int(arc.patch_label)].append(arc)
    return grouped


def _arc_polyline_length(vertices: np.ndarray, arc: BoundaryArc) -> float:
    ids = [int(v) for v in arc.vertex_indices]
    if len(ids) < 2:
        return 0.0
    pts = vertices[np.asarray(ids, dtype=np.int64)]
    return _polyline_length(pts)


def _loop_perimeter(vertices: np.ndarray, loop: Sequence[int]) -> float:
    pts = vertices[np.asarray(loop, dtype=np.int64)]
    if pts.shape[0] < 2:
        return 0.0
    closed = np.vstack([pts, pts[:1]])
    return _polyline_length(closed)


def _arc_local_face_ownership(
    arc: BoundaryArc,
    face_labels: Mapping[int, int],
    boundary_edge_supports: Sequence[Sequence[int]],
) -> Tuple[int, int]:
    """返回弧段孔内邻接面中，属于本弧 source 聚类 / 其他聚类的计数。"""
    target = (
        int(arc.source_face_patch_label)
        if arc.source_face_patch_label is not None
        else int(arc.patch_label)
    )
    own = 0
    other = 0
    for edge_idx in arc.edge_indices:
        ei = int(edge_idx)
        if not (0 <= ei < len(boundary_edge_supports)):
            continue
        for fi in boundary_edge_supports[ei]:
            fi = int(fi)
            if fi not in face_labels:
                continue
            if int(face_labels[fi]) == target:
                own += 1
        else:
                other += 1
    return own, other


def _label_inward_ownership_ratio(
    label: int,
    arcs: Sequence[BoundaryArc],
    vertices: np.ndarray,
    loop: Sequence[int],
    hole_center: np.ndarray,
    patch_surface_fits: Mapping[int, SurfaceFit],
    *,
    loop_step: float,
) -> Tuple[int, int]:
    """
    沿孔内方向短步探测：probe 点最近属于哪个 patch 拟合曲面。
    返回 (owned_probes, total_probes)。
    """
    owned = 0
    total = 0
    all_labels = sorted(int(arc.patch_label) for arc in arcs)
    step = max(0.35 * float(loop_step), 1e-6)
    hole_center = np.asarray(hole_center, dtype=np.float64).reshape(3)
    n_loop = len(loop)
    for arc in arcs:
        if int(arc.patch_label) != int(label):
            continue
        for edge_idx in arc.edge_indices:
            ei = int(edge_idx)
            if not (0 <= ei < n_loop):
                continue
            u = int(loop[ei])
            v = int(loop[(ei + 1) % n_loop])
            mid = 0.5 * (
                np.asarray(vertices[u], dtype=np.float64)
                + np.asarray(vertices[v], dtype=np.float64)
            )
            inward = hole_center - mid
            ln = float(np.linalg.norm(inward))
            if ln <= 1e-15:
                continue
            inward = inward / ln
            probe = mid + step * inward
            best_label: Optional[int] = None
            best_dist = float("inf")
            for cand in all_labels:
                fit = patch_surface_fits.get(int(cand))
                if fit is None:
                    continue
                dist = _fit_distance_to_point(fit, probe)
                if dist < best_dist:
                    best_dist = dist
                    best_label = int(cand)
            if best_label is None:
                continue
            total += 1
            if best_label == int(label):
                owned += 1
    return owned, total


# ---------------------------------------------------------------------------
# L2 所有权：聚类 K → 补洞 M（一次性定稿）
# ---------------------------------------------------------------------------

# 内侧支撑探针：低于此值的 label 视为邻接条带（support），不参与补洞。
_INWARD_OWNERSHIP_ACTIVE_MIN = 0.34
# 孔边弧总顶点数不超过此值时，仅当同时缺承载/邻接混贴才视为退化条带。
_DEGENERATE_SUPPORT_STRIP_VERTICES = 2
# 孔内邻接面仍以本 label 为主时，短弧仍是独立承载面（case_0017 Y 形第三面）。
_DEGENERATE_STRIP_MIN_OWN_FACE_RATIO = 0.5


@dataclass(frozen=True)
class LabelCarrierMetrics:
    """单 label 在孔环局部的承载证据。"""

    label: int
    arc_length: float
    arc_fraction: float
    boundary_vertices: int
    own_face_ratio: float
    inward_ownership: float

    @property
    def is_degenerate_strip(self) -> bool:
        """
        邻接退化条带：孔边弧极短，且不具备本 patch 在孔局部的独立承载。

        仅 ``boundary_vertices <= 2`` 不足以判 support（case_0017）：
        若孔内邻接面仍属本 label（``own_face_ratio`` 高）且内侧探针有支撑，
        则该短弧是三面交汇角上的真实承载面，不是与其他 patch 无关的邻接条带。
        """
        if int(self.boundary_vertices) > _DEGENERATE_SUPPORT_STRIP_VERTICES:
            return False
        if (
            not self.lacks_inward_support
            and float(self.own_face_ratio) >= _DEGENERATE_STRIP_MIN_OWN_FACE_RATIO
        ):
            return False
        return True

    @property
    def lacks_inward_support(self) -> bool:
        return float(self.inward_ownership) < _INWARD_OWNERSHIP_ACTIVE_MIN


def _collect_label_carrier_metrics(
    vertices: np.ndarray,
    loop: Sequence[int],
    arcs: Sequence[BoundaryArc],
    face_labels: Mapping[int, int],
    boundary_edge_supports: Sequence[Sequence[int]],
    patch_surface_fits: Mapping[int, SurfaceFit],
    hole_center: np.ndarray,
    *,
    loop_step: float,
) -> Dict[int, LabelCarrierMetrics]:
    loop_perim = max(_loop_perimeter(vertices, loop), 1e-12)
    label_arcs = _arc_label_runs(arcs)
    out: Dict[int, LabelCarrierMetrics] = {}
    for label in sorted({int(arc.patch_label) for arc in arcs}):
        runs = label_arcs.get(int(label), [])
        arc_length = sum(_arc_polyline_length(vertices, arc) for arc in runs)
        boundary_vertices = sum(len(arc.vertex_indices) for arc in runs)
        own_faces = 0
        other_faces = 0
        for arc in runs:
            o, f = _arc_local_face_ownership(arc, face_labels, boundary_edge_supports)
            own_faces += int(o)
            other_faces += int(f)
        face_total = own_faces + other_faces
        own_ratio = float(own_faces) / float(face_total) if face_total > 0 else 0.0
        owned_probes, probe_total = _label_inward_ownership_ratio(
            int(label),
            arcs,
            vertices,
            loop,
            hole_center,
            patch_surface_fits,
            loop_step=float(loop_step),
        )
        inward = float(owned_probes) / float(probe_total) if probe_total > 0 else 0.0
        out[int(label)] = LabelCarrierMetrics(
            label=int(label),
            arc_length=float(arc_length),
            arc_fraction=float(arc_length) / float(loop_perim),
            boundary_vertices=int(boundary_vertices),
            own_face_ratio=float(own_ratio),
            inward_ownership=float(inward),
        )
    return out


def _decide_active_and_support_labels(
    metrics: Mapping[int, LabelCarrierMetrics],
) -> Tuple[Set[int], Set[int], str]:
    """
    由承载证据裁决 active / support。

    规则::
    - K≤1：全部 active
    - 退化条带：弧顶点≤2 **且**（内侧无支撑 **或** 孔内邻接面以其他 patch 为主）
    - 内侧无支撑（inward_ownership 低）→ support
    - 其余 → active
    - 若无一 active → 全部 support（L3 将显式失败）
    """
    all_labels = {int(x) for x in metrics.keys()}
    if len(all_labels) <= 1:
        return set(all_labels), set(), "single_label"

    active: Set[int] = set()
    support: Set[int] = set()
    for label, m in sorted(metrics.items()):
        if m.is_degenerate_strip or m.lacks_inward_support:
            support.add(int(label))
        else:
            active.add(int(label))

    if not active:
        return set(), all_labels, "no_opening_carrier"
    support.update(int(x) for x in all_labels - active)
    return active, support, "local_opening_carrier"


def _active_active_corner_vertices(
    arcs: Sequence[BoundaryArc],
    active_labels: Set[int],
) -> Set[int]:
    """孔环上 active–active 弧段交界处的角点顶点 id。"""
    active = {int(x) for x in active_labels}
    if len(arcs) < 2 or not active:
        return set()
    corners: Set[int] = set()
    n_arcs = len(arcs)
    for i, arc in enumerate(arcs):
        left_label = int(arc.patch_label)
        right_arc = arcs[(i + 1) % n_arcs]
        right_label = int(right_arc.patch_label)
        if left_label == right_label:
                    continue
        if left_label not in active or right_label not in active:
            continue
        if arc.vertex_indices:
            corners.add(int(arc.vertex_indices[-1]))
        elif right_arc.vertex_indices:
            corners.add(int(right_arc.vertex_indices[0]))
    return corners


def _active_support_junction_vertices(
    loop: Sequence[int],
    boundary_vertex_labels: Mapping[int, Sequence[int]],
    active_labels: Set[int],
    support_labels: Set[int],
) -> Set[int]:
    """
    active–support 接触角点：只连一个 active 的邻接条带角点，不得作剖分特征点。
    """
    active = {int(x) for x in active_labels}
    support = {int(x) for x in support_labels}
    if not active or not support:
        return set()
    demote: Set[int] = set()
    for vi in loop:
        labels = _vertex_label_set(boundary_vertex_labels, int(vi))
        if not (labels & active) or not (labels & support):
            continue
        if len(labels & active) >= 2:
            continue
        demote.add(int(vi))
    return demote


def _make_fill_classification_from_labels(
    *,
    active_labels: Set[int],
    support_labels: Set[int],
    arcs: Sequence[BoundaryArc],
    inactive_feature_points: Set[int],
    diagnostics_extra: Optional[Mapping[str, object]] = None,
) -> FillPatchClassification:
    label_arcs = _arc_label_runs(arcs)
    degenerate_paths: Dict[int, List[List[int]]] = {}
    for label in sorted(support_labels):
        paths = [
            [int(v) for v in arc.vertex_indices]
            for arc in label_arcs.get(int(label), [])
            if arc.vertex_indices
        ]
        if paths:
            degenerate_paths[int(label)] = paths

    active_feature_points: Set[int] = _active_active_corner_vertices(arcs, active_labels)

    inactive = set(int(v) for v in inactive_feature_points)
    active_feature_points -= inactive
    diagnostics: Dict[str, object] = {
        "active_fill_labels": sorted(int(x) for x in active_labels),
        "support_labels": sorted(int(x) for x in support_labels),
        "inactive_feature_points": sorted(inactive),
        "active_feature_points": sorted(int(x) for x in active_feature_points),
        "suppressed_pairs": [],
        "degenerate_label_paths": {
            int(lbl): [[int(v) for v in path] for path in paths]
            for lbl, paths in sorted(degenerate_paths.items())
        },
        "classification_source": "fill_ownership",
    }
    if diagnostics_extra:
        diagnostics.update(dict(diagnostics_extra))
    return FillPatchClassification(
        active_fill_labels=set(active_labels),
        support_labels=set(support_labels),
        degenerate_label_paths=degenerate_paths,
        inactive_feature_points=inactive,
        active_feature_points=active_feature_points,
        suppressed_pairs=set(),
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# S1 承载分类入口（调用 L2 证据收集 + 裁决）
# ---------------------------------------------------------------------------


def _classify_local_opening_carriers(
    vertices: np.ndarray,
    loop: Sequence[int],
    arcs: Sequence[BoundaryArc],
    face_labels: Mapping[int, int],
    boundary_edge_supports: Sequence[Sequence[int]],
    patch_surface_fits: Mapping[int, SurfaceFit],
    hole_center: np.ndarray,
    *,
    loop_step: float,
) -> FillPatchClassification:
    """
    判定孔环上哪些 patch label 局部承载孔口（active），哪些仅为邻接条带（support）。

    这是**该孔局部**结论；同一曲面在模型其他位置可有不同孔态。
    """
    all_labels = {int(arc.patch_label) for arc in arcs}
    if not arcs or len(all_labels) <= 1:
        return _make_fill_classification_from_labels(
            active_labels=set(all_labels),
            support_labels=set(),
            arcs=arcs,
            inactive_feature_points=set(),
            diagnostics_extra={"local_label_stats": {}, "decision_reason": "single_label"},
        )

    metrics = _collect_label_carrier_metrics(
        vertices,
        loop,
        arcs,
        face_labels,
        boundary_edge_supports,
        patch_surface_fits,
        hole_center,
        loop_step=float(loop_step),
    )
    active, support, reason = _decide_active_and_support_labels(metrics)
    stats = {
        int(k): {
            "arc_length": m.arc_length,
            "arc_fraction": m.arc_fraction,
            "boundary_vertices": m.boundary_vertices,
            "own_face_ratio": m.own_face_ratio,
            "inward_ownership": m.inward_ownership,
            "is_degenerate_strip": m.is_degenerate_strip,
            "lacks_inward_support": m.lacks_inward_support,
        }
        for k, m in metrics.items()
    }
    return _make_fill_classification_from_labels(
        active_labels=active,
        support_labels=support,
        arcs=arcs,
        inactive_feature_points=set(),
        diagnostics_extra={
            "cluster_labels": sorted(int(x) for x in all_labels),
            "opening_carriers": sorted(int(x) for x in active),
            "adjacency_only": sorted(int(x) for x in support),
            "decision_reason": reason,
            "local_label_stats": stats,
        },
    )


def _curve_pair_key(curve: IntersectionCurve) -> Tuple[int, int]:
    return tuple(sorted((int(curve.patch_pair[0]), int(curve.patch_pair[1]))))


def _finalize_layout_curves(
    curves: Sequence[IntersectionCurve],
    active_fill_labels: Set[int],
    support_labels: Set[int],
) -> Tuple[List[IntersectionCurve], List[IntersectionCurve]]:
    """
    L2 定稿后裁剪 layout 交线（下游 L3 只读）。

    - ``|M| = 0``：全部移除
    - ``|M| = 1``：全部移除（opening_carrier 整孔环）
    - ``|M| > 1``：仅保留 **active–active** 非退化交线
    """
    initial = list(curves)
    if not initial:
        return [], []
    active = {int(x) for x in active_fill_labels}
    support = {int(x) for x in support_labels}
    if len(active) <= 1:
        return [], list(initial)

    kept: List[IntersectionCurve] = []
    removed: List[IntersectionCurve] = []
    for curve in initial:
        if _is_degenerate_intersection_curve(curve):
            removed.append(curve)
            continue
        a, b = (int(curve.patch_pair[0]), int(curve.patch_pair[1]))
        if a in support and b in support:
            removed.append(curve)
            continue
        if a not in active or b not in active:
            removed.append(curve)
            continue
        kept.append(curve)
    return kept, removed


def _curve_endpoint_vertex_ids(
    curve: IntersectionCurve,
    vertices: np.ndarray,
    loop: Sequence[int],
    tol: float,
) -> Set[int]:
    """交线端点对应的孔环 mesh 顶点（含虚拟端点几何重合）。"""
    out: Set[int] = set()
    e0, e1 = (
        int(curve.endpoint_vertex_indices[0]),
        int(curve.endpoint_vertex_indices[1]),
    )
    if e0 >= 0:
        out.add(e0)
    if e1 >= 0:
        out.add(e1)
    pts = np.asarray(curve.curve_points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return out
    for point in (pts[0], pts[-1]):
        snapped = _nearest_loop_vertex_id(vertices, point, loop, tol)
        if snapped is not None:
            out.add(int(snapped))
    return out


def _curve_endpoint_match_tolerance(vertices: np.ndarray, loop: Sequence[int]) -> float:
    loop_pts = vertices[np.asarray(loop, dtype=np.int64)]
    diag = max(float(np.linalg.norm(np.ptp(loop_pts, axis=0))), 1e-12)
    return max(1e-9, 1e-8 * diag)


def _active_feature_points_from_kept_curves(
    kept_curves: Sequence[IntersectionCurve],
    vertices: np.ndarray,
    loop: Sequence[int],
    *,
    demoted: Set[int],
) -> Set[int]:
    tol = _curve_endpoint_match_tolerance(vertices, loop)
    active: Set[int] = set()
    for curve in kept_curves:
        for vertex in _curve_endpoint_vertex_ids(curve, vertices, loop, tol):
            if int(vertex) not in demoted:
                active.add(int(vertex))
    return active


def _demote_feature_points_for_layout(
    all_curves: Sequence[IntersectionCurve],
    kept_curves: Sequence[IntersectionCurve],
    loop: Sequence[int],
    boundary_vertex_labels: Mapping[int, Sequence[int]],
    active_labels: Set[int],
    support_labels: Set[int],
    candidate_vertices: Sequence[int],
    vertices: np.ndarray,
) -> Set[int]:
    """
    降级不应参与剖分的特征点：

    1. 仅关联已移除交线的顶点；
    2. active–support 接触角点（非 active–active 多面角）。
    """
    tol = _curve_endpoint_match_tolerance(vertices, loop)
    kept_incident_vertices: Set[int] = set()
    for curve in kept_curves:
        kept_incident_vertices.update(
            _curve_endpoint_vertex_ids(curve, vertices, loop, tol)
        )
    demoted = _active_support_junction_vertices(
        loop,
        boundary_vertex_labels,
        active_labels,
        support_labels,
    )
    # A vertex still used by any kept layout curve remains a legal L2 corner,
    # even if another coincident/duplicate curve incident to it was removed.
    demoted -= kept_incident_vertices
    for vertex in candidate_vertices:
        v = int(vertex)
        if v in kept_incident_vertices:
            continue
        incident = [
            curve
            for curve in all_curves
            if v in _curve_endpoint_vertex_ids(curve, vertices, loop, tol)
        ]
        if incident:
            demoted.add(v)
    return demoted


def _apply_adjacency_curve_and_feature_refinement(
    curves: Sequence[IntersectionCurve],
    fill_classification: FillPatchClassification,
    arcs: Sequence[BoundaryArc],
    feature_point_vertex_ids: Sequence[int],
    vertices: np.ndarray,
    loop: Sequence[int],
    boundary_vertex_labels: Mapping[int, Sequence[int]],
) -> Tuple[
    List[IntersectionCurve],
    FillPatchClassification,
    Set[int],
    Dict[str, object],
]:
    active_labels = set(int(x) for x in fill_classification.active_fill_labels)
    support_labels = set(int(x) for x in fill_classification.support_labels)
    initial_curves = list(curves)
    kept_curves, removed_curves = _finalize_layout_curves(
        initial_curves,
        active_labels,
        support_labels,
    )
    demoted_feature_points = _demote_feature_points_for_layout(
        initial_curves,
        kept_curves,
        loop,
        boundary_vertex_labels,
        active_labels,
        support_labels,
        feature_point_vertex_ids,
        vertices,
    )
    kept_curve_feature_points = _active_feature_points_from_kept_curves(
        kept_curves,
        vertices,
        loop,
        demoted=demoted_feature_points,
    )
    refined_classification = _make_fill_classification_from_labels(
        active_labels=set(fill_classification.active_fill_labels),
        support_labels=set(fill_classification.support_labels),
        arcs=arcs,
        inactive_feature_points=demoted_feature_points,
        diagnostics_extra={
            **dict(fill_classification.diagnostics),
            "demoted_feature_points": sorted(int(x) for x in demoted_feature_points),
            "kept_curve_feature_points": sorted(int(x) for x in kept_curve_feature_points),
            "initial_curve_pairs": [_curve_pair_key(c) for c in initial_curves],
            "kept_curve_pairs": [_curve_pair_key(c) for c in kept_curves],
            "removed_curve_pairs": [_curve_pair_key(c) for c in removed_curves],
        },
    )
    active_feature_points = (
        set(refined_classification.active_feature_points) | kept_curve_feature_points
    ) - set(demoted_feature_points)
    refined_classification = replace(
        refined_classification,
        active_feature_points=active_feature_points,
        diagnostics={
            **dict(refined_classification.diagnostics),
            "active_feature_points": sorted(int(x) for x in active_feature_points),
        },
    )
    prune_diag: Dict[str, object] = {
        "removed_intersection_curves": list(removed_curves),
        "demoted_feature_points": sorted(int(x) for x in demoted_feature_points),
        "removed_curve_pairs": [_curve_pair_key(c) for c in removed_curves],
        "kept_curve_pairs": [_curve_pair_key(c) for c in kept_curves],
        "n_curves_kept": int(len(kept_curves)),
        "n_curves_removed": int(len(removed_curves)),
    }
    return kept_curves, refined_classification, demoted_feature_points, prune_diag


def _ownership_snapshot_to_dict(snapshot: FillOwnershipSnapshot) -> Dict[str, object]:
    return {
        "cluster_labels": sorted(int(x) for x in snapshot.cluster_labels),
        "active_fill_labels": sorted(int(x) for x in snapshot.active_fill_labels),
        "support_labels": sorted(int(x) for x in snapshot.support_labels),
        "kept_curve_pairs": [list(p) for p in snapshot.kept_curve_pairs],
        "removed_curve_pairs": [list(p) for p in snapshot.removed_curve_pairs],
        "demoted_feature_points": sorted(int(x) for x in snapshot.demoted_feature_points),
        "active_feature_points": sorted(int(x) for x in snapshot.active_feature_points),
        "ownership_finalized": True,
    }


def _attach_ownership_to_arrangement_diagnostics(
    arrangement: FeatureArrangement,
    fill_classification: Optional[FillPatchClassification],
) -> None:
    """L3 只挂载 L2 定稿快照，禁止把分类中间态灌进 arrangement 全局 diagnostics。"""
    if fill_classification is None:
        return
    snap = fill_classification.ownership_snapshot
    if snap is not None:
        arrangement.diagnostics["ownership_snapshot"] = _ownership_snapshot_to_dict(snap)
    arrangement.diagnostics["active_fill_labels"] = sorted(
        int(x) for x in fill_classification.active_fill_labels
    )
    arrangement.diagnostics["support_labels"] = sorted(
        int(x) for x in fill_classification.support_labels
    )


def _finalize_fill_ownership(
    *,
    vertices: np.ndarray,
    faces_arr: np.ndarray,
    loop: Sequence[int],
    arcs: Sequence[BoundaryArc],
    face_labels: Mapping[int, int],
    boundary_edge_supports: Sequence[Sequence[int]],
    boundary_vertex_labels: Mapping[int, Sequence[int]],
    boundary_edge_labels: Sequence[int],
    patch_surface_fits: Mapping[int, SurfaceFit],
    hole_center: np.ndarray,
    loop_step: float,
    unique_patch_count: int,
    feature_point_vertex_ids: Sequence[int],
) -> Tuple[
    FillPatchClassification,
    List[BoundaryArc],
    List[IntersectionCurve],
    Dict[str, object],
    Optional[np.ndarray],
    str,
    List[BoundedCurveSegment],
    List[AnalyticCurve],
    Optional[object],
]:
    """
    L2 唯一裁决入口：聚类 K → 补洞 M，定稿后再恢复/裁剪交线与特征点。

    返回的 ``FillPatchClassification.ownership_snapshot`` 供 L3/L4 只读消费。
    """
    arcs_work = list(arcs)

    draft = _classify_local_opening_carriers(
        vertices,
        loop,
        arcs_work,
        face_labels,
        boundary_edge_supports,
        patch_surface_fits,
        hole_center,
        loop_step=float(loop_step),
    )
    cluster_labels = frozenset(int(x) for x in draft.diagnostics.get("cluster_labels", []))
    if not cluster_labels:
        cluster_labels = frozenset(
            int(x) for x in draft.active_fill_labels | draft.support_labels
        )

    junction_point: Optional[np.ndarray] = None
    junction_confidence = "none"
    bounded_segments: List[BoundedCurveSegment] = []
    analytic_curves: List[AnalyticCurve] = []
    initial_curves: List[IntersectionCurve] = []
    cavity_arrangement_result = None

    skip_intersection = (
        int(unique_patch_count) > 1 and len(draft.active_fill_labels) <= 1
    )
    if not skip_intersection:
        analytic_out = _try_recover_analytic_bounded_curves(
            vertices,
            faces_arr,
            loop,
            arcs_work,
            unique_patch_count,
            boundary_vertex_labels,
            patch_surface_fits,
            hole_center,
            float(loop_step) if loop_step > 1e-15 else None,
            fill_label_count=None,
        )
        if analytic_out is not None:
            (
                initial_curves,
                bounded_segments,
                analytic_curves,
                junction_point,
                junction_confidence,
                cavity_arrangement_result,
            ) = analytic_out
    initial_curves = _dedupe_curves_by_pair(list(initial_curves))

    kept_curves, fill_classification, demoted_feature_points, prune_diag = (
        _apply_adjacency_curve_and_feature_refinement(
            initial_curves,
            draft,
            arcs_work,
            feature_point_vertex_ids,
            vertices,
            loop,
            boundary_vertex_labels,
        )
    )
    snapshot = FillOwnershipSnapshot(
        cluster_labels=cluster_labels,
        active_fill_labels=frozenset(int(x) for x in fill_classification.active_fill_labels),
        support_labels=frozenset(int(x) for x in fill_classification.support_labels),
        kept_curve_pairs=tuple(
            _curve_pair_key(c) for c in kept_curves
        ),
        removed_curve_pairs=tuple(
            prune_diag.get("removed_curve_pairs", [])
        ),
        demoted_feature_points=frozenset(int(x) for x in demoted_feature_points),
        active_feature_points=frozenset(
            int(x) for x in fill_classification.active_feature_points
        ),
    )
    fill_classification = replace(
        fill_classification,
        ownership_snapshot=snapshot,
        diagnostics={
            **dict(fill_classification.diagnostics),
            "ownership_finalized": True,
            "classification_source": "finalize_fill_ownership",
        },
    )
    prune_diag["ownership_snapshot"] = _ownership_snapshot_to_dict(snapshot)
    prune_diag["skipped_intersection_recovery"] = bool(skip_intersection)
    if cavity_arrangement_result is not None:
        prune_diag["layout_source"] = "cavity_arrangement"
        prune_diag["cavity_layout"] = {
            "n_junction_nodes": len(cavity_arrangement_result.junction_nodes),
            "n_curves": len(cavity_arrangement_result.curves),
            "n_carriers": int(
                cavity_arrangement_result.diagnostics.get("n_carriers", 0)
            ),
            "n_carriers_raw": int(
                cavity_arrangement_result.diagnostics.get("n_carriers_raw", 0)
            ),
            "n_segments": int(
                cavity_arrangement_result.diagnostics.get("n_segments", 0)
            ),
            "n_bridge_segments": int(
                cavity_arrangement_result.diagnostics.get("n_bridge_segments", 0)
            ),
            "segment_pairs": list(
                cavity_arrangement_result.diagnostics.get("segment_pairs", [])
            ),
            "arrangement_valid": bool(
                cavity_arrangement_result.diagnostics.get("arrangement_valid", False)
            ),
            "cell_validation_valid": bool(
                cavity_arrangement_result.diagnostics.get("cell_validation_valid", False)
            ),
            "certificate_gate_valid": bool(
                cavity_arrangement_result.diagnostics.get("certificate_gate_valid", False)
            ),
            "edge_certificate_valid": bool(
                cavity_arrangement_result.diagnostics.get("edge_certificate_valid", False)
            ),
            "virtual_junction_certificate_valid": bool(
                cavity_arrangement_result.diagnostics.get(
                    "virtual_junction_certificate_valid",
                    False,
                )
            ),
            "carrier_certificates": list(
                cavity_arrangement_result.diagnostics.get("carrier_certificates", [])
            ),
            "boundary_pair_certificates": list(
                cavity_arrangement_result.diagnostics.get("boundary_pair_certificates", [])
            ),
            "protected_boundary_pairs": list(
                cavity_arrangement_result.diagnostics.get("protected_boundary_pairs", [])
            ),
            "junction_certificates": list(
                cavity_arrangement_result.diagnostics.get("junction_certificates", [])
            ),
            "edge_certificates": list(
                cavity_arrangement_result.diagnostics.get("edge_certificates", [])
            ),
            "cell_proof_report": list(
                cavity_arrangement_result.diagnostics.get("cell_proof_report", [])
            ),
            "surface_cell_validation": list(
                cavity_arrangement_result.diagnostics.get("surface_cell_validation", [])
            ),
            "internal_bridge_proofs": list(
                cavity_arrangement_result.diagnostics.get("internal_bridge_proofs", [])
            ),
            "incident_completion": list(
                cavity_arrangement_result.diagnostics.get("incident_completion", [])
            ),
            "candidate_selection": dict(
                cavity_arrangement_result.diagnostics.get("candidate_selection", {})
            ),
            "virtual_cell_refinement": dict(
                cavity_arrangement_result.diagnostics.get("virtual_cell_refinement", {})
            ),
            "carrier_branch_refinement": dict(
                cavity_arrangement_result.diagnostics.get("carrier_branch_refinement", {})
            ),
            "virtual_junction_prune": dict(
                cavity_arrangement_result.diagnostics.get("virtual_junction_prune", {})
            ),
            "n_candidate_edges": len(
                cavity_arrangement_result.diagnostics.get("candidate_edges", [])
            ),
            "n_candidate_nodes": len(
                cavity_arrangement_result.diagnostics.get("candidate_nodes", [])
            ),
            "virtual_sources": list(
                cavity_arrangement_result.diagnostics.get("virtual_sources", [])
            ),
            "layout_finalized": bool(
                cavity_arrangement_result.diagnostics.get("layout_finalized")
            ),
        }
    return (
        fill_classification,
        arcs_work,
        kept_curves,
        prune_diag,
        junction_point,
        junction_confidence,
        bounded_segments,
        analytic_curves,
        cavity_arrangement_result,
    )


def _nearest_loop_vertex_id(
    vertices: np.ndarray,
    point: np.ndarray,
    loop: Sequence[int],
    tol: float,
) -> Optional[int]:
    p = np.asarray(point, dtype=np.float64).reshape(3)
    best_vi: Optional[int] = None
    best_dist = float(tol)
    for vi in loop:
        dist = float(np.linalg.norm(vertices[int(vi)] - p))
        if dist <= best_dist:
            best_dist = dist
            best_vi = int(vi)
    return best_vi


def _snap_curve_endpoints_to_mesh_vertices(
    curves: Sequence[IntersectionCurve],
    vertices: np.ndarray,
    loop: Sequence[int],
) -> List[IntersectionCurve]:
    """虚拟端点若与孔环顶点重合，绑定到 mesh 顶点 id（避免 debug/下游歧义）。"""
    loop_pts = vertices[np.asarray(loop, dtype=np.int64)]
    diag = max(float(np.linalg.norm(np.ptp(loop_pts, axis=0))), 1e-12)
    tol = max(1e-9, 1e-8 * diag)
    out: List[IntersectionCurve] = []
    for curve in curves:
        pts = np.asarray(curve.curve_points, dtype=np.float64)
        e0, e1 = (
            int(curve.endpoint_vertex_indices[0]),
            int(curve.endpoint_vertex_indices[1]),
        )
        new_e0, new_e1 = e0, e1
        if pts.ndim == 2 and pts.shape[0] > 0:
            if e0 < 0:
                snapped = _nearest_loop_vertex_id(vertices, pts[0], loop, tol)
                if snapped is not None:
                    new_e0 = int(snapped)
            if e1 < 0:
                snapped = _nearest_loop_vertex_id(vertices, pts[-1], loop, tol)
                if snapped is not None:
                    new_e1 = int(snapped)
        if new_e0 != e0 or new_e1 != e1:
            out.append(
                replace(
                    curve,
                    endpoint_vertex_indices=(int(new_e0), int(new_e1)),
                )
            )
        else:
            out.append(curve)
    return out


def _junction_cluster_tolerance(
    vertices: np.ndarray,
    loop: Sequence[int],
    loop_sampling_fallback: Optional[float],
) -> float:
    """虚拟汇交点聚类容差：与 virtual bridge 共用，避免 bridge/snap 拓扑不一致。"""
    loop_pts = vertices[np.asarray(loop, dtype=np.int64)]
    diag = max(float(np.linalg.norm(np.ptp(loop_pts, axis=0))), 1e-12)
    return max(1e-6 * diag, 0.45 * float(loop_sampling_fallback or 0.0), 1e-9)



def _retained_cavity_junction_positions(
    cavity_result: Optional[object],
) -> List[np.ndarray]:
    if cavity_result is None:
        return []
    out: List[np.ndarray] = []
    for node in getattr(cavity_result, "junction_nodes", None) or []:
        if int(getattr(node, "vertex_id", 0)) >= 0:
            continue
        pos = np.asarray(getattr(node, "position", None), dtype=np.float64).reshape(3)
        if np.all(np.isfinite(pos)):
            out.append(pos)
    return out


def _truncate_virtual_endpoints_to_curve_arrangement(
    curves: Sequence[IntersectionCurve],
    patch_surface_fits: Mapping[int, SurfaceFit],
    hole_center: np.ndarray,
    loop_sampling_fallback: Optional[float],
    *,
    retained_junction_positions: Optional[Sequence[np.ndarray]] = None,
) -> List[IntersectionCurve]:
    """
    用解析交线之间的 arrangement 顶点截断虚拟端点。

    孔环非共面时，PCA 多边形裁剪可能把交线延到孔外。若一条“边界特征点→虚拟端点”
    曲线在到达虚拟端点前先遇到另一条相邻解析交线，则应在该 arrangement 顶点终止。

    当 ``retained_junction_positions`` 非空时，仅允许截断到 L2 已保留的汇交位置，
    避免把交线裁到已删除的 carrier 联交点（hole_test5 等台阶孔）。
    """
    retained: List[np.ndarray] = []
    if retained_junction_positions:
        for raw in retained_junction_positions:
            p = np.asarray(raw, dtype=np.float64).reshape(3)
            if np.all(np.isfinite(p)):
                retained.append(p)
    retained_match_tol = max(1e-6, 0.15 * float(loop_sampling_fallback or 0.0))
    if retained:
        retained_match_tol = max(
            retained_match_tol,
            0.45 * float(loop_sampling_fallback or 0.0),
        )
    analytic_by_pair: Dict[Tuple[int, int], AnalyticCurve] = {}
    for curve in curves:
        pair = tuple(sorted((int(curve.patch_pair[0]), int(curve.patch_pair[1]))))
        if pair in analytic_by_pair:
            continue
        if pair[0] not in patch_surface_fits or pair[1] not in patch_surface_fits:
            continue
        ac = analytic_intersection(patch_surface_fits[pair[0]], patch_surface_fits[pair[1]])
        if ac is not None:
            analytic_by_pair[pair] = ac

    out: List[IntersectionCurve] = []
    for curve in curves:
        pair = tuple(sorted((int(curve.patch_pair[0]), int(curve.patch_pair[1]))))
        pts = np.asarray(curve.curve_points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] < 2:
            out.append(curve)
            continue
        e0, e1 = (int(curve.endpoint_vertex_indices[0]), int(curve.endpoint_vertex_indices[1]))
        virtual_endpoint_idx: Optional[int] = None
        if e0 >= 0 and e1 < 0:
            fixed_point = pts[0]
            virtual_point = pts[-1]
            virtual_endpoint_idx = 1
        elif e1 >= 0 and e0 < 0:
            fixed_point = pts[-1]
            virtual_point = pts[0]
            virtual_endpoint_idx = 0
        else:
            out.append(curve)
            continue

        base_vec = np.asarray(virtual_point - fixed_point, dtype=np.float64)
        base_len2 = float(np.dot(base_vec, base_vec))
        if base_len2 <= 1e-20:
            out.append(curve)
            continue

        ac = analytic_by_pair.get(pair)
        if ac is None:
            out.append(curve)
            continue

        best_point: Optional[np.ndarray] = None
        best_t = 1.0
        for other in curves:
            if other is curve:
                continue
            other_pair = tuple(sorted((int(other.patch_pair[0]), int(other.patch_pair[1]))))
            if other_pair == pair or len(set(pair) & set(other_pair)) != 1:
                continue
            other_ac = analytic_by_pair.get(other_pair)
            if other_ac is None:
                continue
            candidate = intersect_analytic_curves(ac, other_ac, guide_point=hole_center)
            if candidate is None:
                continue
            p = np.asarray(candidate, dtype=np.float64).reshape(3)
            t = float(np.dot(p - fixed_point, base_vec) / base_len2)
            # 只接受真正位于曲线内部的 arrangement 截断点。过于靠近固定边界端的
            # 解析交点若过于靠近固定边界端，通常是邻域拟合抖动，不应截短孔内特征线。
            if not (0.08 < t < best_t - 1e-6 and t < 0.98):
                continue
            distance_to_segment = float(np.linalg.norm((fixed_point + t * base_vec) - p))
            tol = max(1e-6, 0.15 * float(loop_sampling_fallback or 0.0))
            if distance_to_segment > tol:
                continue
            if retained and not any(
                float(np.linalg.norm(p - q)) <= retained_match_tol for q in retained
            ):
                continue
            best_t = t
            best_point = p

        if best_point is None:
            out.append(curve)
            continue

        if virtual_endpoint_idx == 1:
            start, end = pts[0], best_point
        else:
            start, end = best_point, pts[-1]
        out.append(
            recover_curve_between_points(
                patch_surface_fits[pair[0]],
                patch_surface_fits[pair[1]],
                start,
                end,
                _feature_curve_guide_point(
                    start, end, hole_center, endpoint_vertex_indices=(e0, e1)
                ),
                endpoint_vertex_indices=(e0, e1),
                intersection_sampling_reference_step=loop_sampling_fallback,
            )
        )
    return out


def _curve_endpoint_tangent_line(
    curve: IntersectionCurve,
    endpoint_idx: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    pts = np.asarray(curve.curve_points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return None
    if int(endpoint_idx) == 0:
        p = pts[0]
        d = pts[1] - pts[0]
    else:
        p = pts[-1]
        d = pts[-1] - pts[-2]
    dn = float(np.linalg.norm(d))
    if dn <= 1e-15:
        return None
    return np.asarray(p, dtype=np.float64).reshape(3), np.asarray(d / dn, dtype=np.float64)


def _constrained_virtual_cluster_center(
    points: Sequence[np.ndarray],
    members: Sequence[Tuple[IntersectionCurve, int]],
    *,
    max_move: float,
) -> np.ndarray:
    anchor = np.mean(np.vstack([np.asarray(p, dtype=np.float64).reshape(3) for p in points]), axis=0)
    lines: List[Tuple[np.ndarray, np.ndarray]] = []
    for curve, endpoint_idx in members:
        line = _curve_endpoint_tangent_line(curve, int(endpoint_idx))
        if line is not None:
            lines.append(line)
    if len(lines) < 2:
        return np.asarray(anchor, dtype=np.float64)

    a_mat = np.zeros((3, 3), dtype=np.float64)
    b_vec = np.zeros(3, dtype=np.float64)
    eye = np.eye(3, dtype=np.float64)
    for p, d in lines:
        d = np.asarray(d, dtype=np.float64).reshape(3)
        d = d / max(float(np.linalg.norm(d)), 1e-15)
        projector = eye - np.outer(d, d)
        a_mat += projector
        b_vec += projector @ np.asarray(p, dtype=np.float64).reshape(3)

    # 保留一个弱 anchor，防止近似平行线把交点推离孔洞局部。
    anchor_weight = max(0.05 * float(len(lines)), 1e-6)
    a_mat += anchor_weight * eye
    b_vec += anchor_weight * anchor
    try:
        center = np.linalg.solve(a_mat, b_vec)
    except np.linalg.LinAlgError:
        center = np.linalg.lstsq(a_mat, b_vec, rcond=None)[0]
    if not np.all(np.isfinite(center)):
        return np.asarray(anchor, dtype=np.float64)
    move = float(np.linalg.norm(center - anchor))
    if max_move > 0.0 and move > max_move:
        center = anchor + (center - anchor) * (max_move / move)
    return np.asarray(center, dtype=np.float64)


def _virtual_endpoint_clusters(
    vertices: np.ndarray,
    loop: Sequence[int],
    curves: Sequence[IntersectionCurve],
    loop_sampling_fallback: Optional[float],
) -> List[Tuple[np.ndarray, Set[int], List[Tuple[IntersectionCurve, int]]]]:
    # 与 snap / virtual bridge 共用容差，保证拓扑阶段一致。
    tol = _junction_cluster_tolerance(vertices, loop, loop_sampling_fallback)
    clusters: List[Tuple[List[np.ndarray], Set[int], List[Tuple[IntersectionCurve, int]]]] = []
    for curve in curves:
        pts = np.asarray(curve.curve_points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] == 0:
            continue
        pair_labels = {int(curve.patch_pair[0]), int(curve.patch_pair[1])}
        endpoints = (
            (int(curve.endpoint_vertex_indices[0]), pts[0]),
            (int(curve.endpoint_vertex_indices[1]), pts[-1]),
        )
        for endpoint_idx, (endpoint_id, point) in enumerate(endpoints):
            if endpoint_id >= 0:
                continue
            p = np.asarray(point, dtype=np.float64).reshape(3)
            for cluster_points, labels, members in clusters:
                center = np.mean(np.vstack(cluster_points), axis=0)
                if float(np.linalg.norm(p - center)) <= tol:
                    cluster_points.append(p)
                    labels.update(pair_labels)
                    members.append((curve, endpoint_idx))
                    break
            else:
                clusters.append(([p], set(pair_labels), [(curve, endpoint_idx)]))
    result = [
        (
            _constrained_virtual_cluster_center(
                points,
                members,
                max_move=max(0.35 * float(loop_sampling_fallback or 0.0), 1e-9),
            ),
            labels,
            members,
        )
        for points, labels, members in clusters
        if len(points) >= 1
    ]
    return _coalesce_virtual_endpoint_clusters(
        result,
        loop_step=loop_sampling_fallback,
    )


def _coalesce_virtual_endpoint_clusters(
    clusters: Sequence[Tuple[np.ndarray, Set[int], List[Tuple[IntersectionCurve, int]]]],
    *,
    loop_step: Optional[float] = None,
) -> List[Tuple[np.ndarray, Set[int], List[Tuple[IntersectionCurve, int]]]]:
    """
    合并共享 patch 标签且相距不远的汇交簇（四面柱/平面汇交常产生近邻多簇）。
    """
    if len(clusters) < 2:
        return list(clusters)
    merge_tol = max(1e-6, 2.5 * float(loop_step or 0.0))
    work: List[Tuple[np.ndarray, Set[int], List[Tuple[IntersectionCurve, int]]]] = [
        (
            np.asarray(center, dtype=np.float64).reshape(3),
            set(int(x) for x in labels),
            list(members),
        )
        for center, labels, members in clusters
    ]
    merged = True
    while merged:
        merged = False
        for i in range(len(work)):
            if merged:
                break
            for j in range(i + 1, len(work)):
                ci, li, mi = work[i]
                cj, lj, mj = work[j]
                if not (li & lj):
                    continue
                if float(np.linalg.norm(ci - cj)) > merge_tol:
                    continue
                points = [ci, cj]
                members = list(mi) + list(mj)
                center = _constrained_virtual_cluster_center(
                    points,
                    members,
                    max_move=max(0.35 * float(loop_step or 0.0), 1e-9),
                )
                work[i] = (
                    np.asarray(center, dtype=np.float64).reshape(3),
                    li | lj,
                    members,
                )
                del work[j]
                merged = True
                break
    return work


def _snap_virtual_curve_endpoints_to_clusters(
    curves: Sequence[IntersectionCurve],
    clusters: Sequence[Tuple[np.ndarray, Set[int], List[Tuple[IntersectionCurve, int]]]],
) -> List[IntersectionCurve]:
    center_by_endpoint: Dict[Tuple[int, int], np.ndarray] = {}
    for center, _labels, members in clusters:
        c = np.asarray(center, dtype=np.float64).reshape(3)
        for curve, endpoint_idx in members:
            center_by_endpoint[(id(curve), int(endpoint_idx))] = c
    out: List[IntersectionCurve] = []
    for curve in curves:
        pts = np.asarray(curve.curve_points, dtype=np.float64).copy()
        if pts.ndim == 2 and pts.shape[0] > 0:
            p0 = center_by_endpoint.get((id(curve), 0))
            p1 = center_by_endpoint.get((id(curve), 1))
            if p0 is not None:
                pts[0] = p0
            if p1 is not None:
                pts[-1] = p1
        out.append(
            replace(
                curve,
                curve_points=pts,
                endpoints_on_boundary=np.vstack([pts[0], pts[-1]])
                if pts.ndim == 2 and pts.shape[0] > 0
                else curve.endpoints_on_boundary,
            )
        )
    return out


def _is_virtual_bridge_curve(curve: IntersectionCurve) -> bool:
    e0, e1 = (
        int(curve.endpoint_vertex_indices[0]),
        int(curve.endpoint_vertex_indices[1]),
    )
    return e0 < 0 and e1 < 0


def _try_recover_analytic_bounded_curves(
    vertices: np.ndarray,
    faces_arr: np.ndarray,
    loop: Sequence[int],
    arcs: Sequence[BoundaryArc],
    unique_patch_count: int,
    boundary_vertex_labels: Mapping[int, Sequence[int]],
    patch_surface_fits: Mapping[int, SurfaceFit],
    hole_center: np.ndarray,
    loop_sampling_fallback: Optional[float],
    *,
    fill_label_count: Optional[int] = None,
) -> Optional[Tuple[List[IntersectionCurve], List[BoundedCurveSegment], List[AnalyticCurve], Optional[np.ndarray], str]]:
    if unique_patch_count <= 1 or not arcs:
        return None
    if len(patch_surface_fits) < 2:
        return None
    pair_budget = (
        int(fill_label_count)
        if fill_label_count is not None
        else int(unique_patch_count)
    )
    require_all_pairs = pair_budget <= 3

    effective = _effective_boundary_vertex_labels(
        boundary_vertex_labels,
        patch_surface_fits,
    )
    pairs = _adjacent_patch_pairs_from_boundary(effective, arcs, patch_surface_fits)
    if len(patch_surface_fits) <= 6:
        all_pairs = [
            tuple(sorted((int(a), int(b))))
            for a, b in combinations(sorted(int(x) for x in patch_surface_fits), 2)
        ]
        seen_pairs = {tuple(sorted(pair)) for pair in pairs}
        for pair in all_pairs:
            if pair not in seen_pairs:
                pairs.append(pair)
                seen_pairs.add(pair)
    if not pairs:
        return None

    arc_hints = _build_arc_corner_hints(arcs, pairs, effective)

    from .hole_cavity_arrangement import (
        finalize_cavity_arrangement_layout,
        recover_cavity_restricted_curves,
    )

    cavity_result = recover_cavity_restricted_curves(
        patch_surface_fits,
        pairs,
        vertices,
        loop,
        hole_center=hole_center,
        loop_mean_edge=loop_sampling_fallback,
        vertex_labels=effective,
        arc_corner_hints=arc_hints,
    )
    curves = list(cavity_result.curves)
    segments = list(cavity_result.bounded_segments)
    analytics = list(cavity_result.analytic_curves)

    covered_pairs = {tuple(sorted(c.patch_pair)) for c in curves}
    incomplete_layout = bool(require_all_pairs and len(covered_pairs) < len(pairs))

    finalized = finalize_cavity_arrangement_layout(
        cavity_result,
        patch_surface_fits,
        hole_center=hole_center,
    )
    curves = list(finalized.curves)
    junction_point = finalized.junction_point
    junction_confidence = str(finalized.junction_confidence)

    covered_pairs = {tuple(sorted(c.patch_pair)) for c in curves}
    if require_all_pairs and len(covered_pairs) < len(pairs) and not incomplete_layout:
        return None
    return (
            curves,
        segments,
        analytics,
        junction_point,
        junction_confidence,
        finalized,
    )


class FillValidationError(ValueError):
    """分区补洞前硬验收失败（label 不全或子孔剖分不完整）。"""


def seam_constrained_edges_for_subhole(
    subhole: PreparedSubhole,
) -> FrozenSet[Tuple[int, int]]:
    """从 ``boundary_sources`` 导出交线接缝约束（负 source 链段）。"""
    sources = [int(x) for x in subhole.boundary_sources]
    n = int(subhole.closed_boundary_points.shape[0])
    if n < 2 or len(sources) < 2:
        return frozenset()
    out: Set[Tuple[int, int]] = set()
    for i in range(n):
        j = (i + 1) % n
        if i >= len(sources) or j >= len(sources):
            continue
        if int(sources[i]) < 0 or int(sources[j]) < 0:
            out.add((min(i, j), max(i, j)))
    return frozenset(out)


def validate_before_partitioned_fill(
    prepared: Sequence[PreparedSubhole],
    expected_labels: Set[int],
) -> None:
    """布尔硬验收：active label 子孔必须齐全。"""
    ok, reason = _accept_prepared_subholes(prepared, expected_labels)
    if not ok:
        raise FillValidationError(reason)


def _accept_prepared_subholes(
    prepared: Sequence[PreparedSubhole],
    expected_labels: Set[int],
) -> Tuple[bool, str]:
    if not expected_labels:
        return False, "no_expected_labels"
    got = {int(item.patch_label) for item in prepared}
    if got != expected_labels:
        return (
            False,
            (
                f"got_labels={sorted(got)} "
                f"expected_labels={sorted(expected_labels)} "
                f"missing={sorted(expected_labels - got)}"
            ),
        )
    return True, "ok"


def _assess_prepared_subholes_fill_ready(
    prepared: Sequence[PreparedSubhole],
    expected_labels: Set[int],
) -> Tuple[bool, str, List[Dict[str, object]]]:
    """L3 fill-ready gate: proven cells must already be triangulation-ready."""
    reports: List[Dict[str, object]] = []
    label_ok, label_reason = _accept_prepared_subholes(prepared, expected_labels)
    if not label_ok:
        return False, str(label_reason), reports
    for subhole in prepared:
        pts = np.asarray(subhole.closed_boundary_points, dtype=np.float64)
        uv = np.asarray(subhole.boundary_points_2d, dtype=np.float64)
        report: Dict[str, object] = {
            "patch_label": int(subhole.patch_label),
            "n_boundary_points": int(pts.shape[0]),
            "closure_kind": str(subhole.closure_kind),
            "parameterization_kind": str(subhole.parameterization_kind),
        }
        if pts.ndim != 2 or pts.shape[0] < 3:
            report.update({"ready": False, "reject_reason": "too_few_boundary_points"})
            reports.append(report)
            continue
        if uv.ndim != 2 or uv.shape[0] != pts.shape[0]:
            report.update({"ready": False, "reject_reason": "uv_boundary_size_mismatch"})
            reports.append(report)
            continue
        readiness = assess_patch_boundary_readiness(pts, uv)
        report.update(readiness)
        if not bool(readiness.get("ready", False)):
            report["reject_reason"] = "boundary_not_triangulation_ready"
        reports.append(report)
    failed = [item for item in reports if not bool(item.get("ready", False))]
    if failed:
        labels = [int(item.get("patch_label", -1)) for item in failed]
        return False, f"fill_ready_failed labels={labels}", reports
    return True, "ok", reports


def _build_fill_gate(
    prepared: Sequence[PreparedSubhole],
    fill_classification: FillPatchClassification,
    feature_arrangement: Optional[FeatureArrangement],
    *,
    unique_patch_count: int,
) -> FillGateResult:
    """S7 窄腰：汇总 expected/got/accepted，供 S8 与批处理直接读取。"""
    expected = frozenset(int(x) for x in fill_classification.active_fill_labels)
    pipeline_stage = FILL_STAGE_EXPORT_PREPARED
    reject_reason = ""
    if feature_arrangement is not None:
        diag = feature_arrangement.diagnostics
        exp_diag = diag.get("expected_labels")
        if isinstance(exp_diag, (list, tuple, set, frozenset)):
            expected = frozenset(int(x) for x in exp_diag)
        pipeline_stage = str(diag.get("fill_pipeline_stage") or pipeline_stage)
        reject_reason = str(
            diag.get("subhole_rejection") or diag.get("fill_reject_reason") or ""
        )
    elif unique_patch_count <= 1:
        pipeline_stage = FILL_STAGE_EXPORT_PREPARED

    got = frozenset(int(p.patch_label) for p in prepared)
    ok, reason = _accept_prepared_subholes(prepared, set(expected))
    accepted = bool(ok)
    if not accepted and not reject_reason:
        reject_reason = str(reason)
    if accepted:
        reject_reason = ""
    return FillGateResult(
        expected_labels=expected,
        got_labels=got,
        accepted=accepted,
        pipeline_stage=pipeline_stage,
        reject_reason=reject_reason,
    )




# ---------------------------------------------------------------------------
# 孔域剖分工作态（L1 → L2 → L3 → L5 单一状态对象）
# ---------------------------------------------------------------------------


@dataclass
class HoleFillWorkState:
    """analyze 管线唯一中间状态；各层只读写本对象，不平行维护多套叙事。"""

    mesh: HalfEdgeMesh
    loop: List[int]
    vertices: np.ndarray
    faces: np.ndarray
    scale: HoleScale
    # L1
    neighborhood_faces: List[int] = field(default_factory=list)
    boundary_half_edges: List[int] = field(default_factory=list)
    boundary_edge_labels: List[int] = field(default_factory=list)
    boundary_vertex_labels: Dict[int, List[int]] = field(default_factory=dict)
    boundary_arcs: List[BoundaryArc] = field(default_factory=list)
    arcs: List[BoundaryArc] = field(default_factory=list)
    patch_surface_fits: Dict[int, SurfaceFit] = field(default_factory=dict)
    surface_face_labels: Dict[int, int] = field(default_factory=dict)
    surface_patch_face_indices: Dict[int, List[int]] = field(default_factory=dict)
    face_labels: Dict[int, int] = field(default_factory=dict)
    boundary_edge_supports: List[List[int]] = field(default_factory=list)
    unique_patch_count: int = 1
    hole_center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    feature_point_positions: List[int] = field(default_factory=list)
    feature_point_vertex_ids: List[int] = field(default_factory=list)
    # L2（定稿）
    fill_classification: Optional[FillPatchClassification] = None
    ownership_snapshot: Optional[FillOwnershipSnapshot] = None
    intersection_curves: List[IntersectionCurve] = field(default_factory=list)
    bounded_segments: List[BoundedCurveSegment] = field(default_factory=list)
    analytic_curves: List[AnalyticCurve] = field(default_factory=list)
    junction_point: Optional[np.ndarray] = None
    junction_confidence: str = "none"
    prune_diagnostics: Dict[str, object] = field(default_factory=dict)
    cavity_arrangement_result: Optional[object] = None
    active_feature_point_positions: List[int] = field(default_factory=list)
    feature_edges: List[Tuple[int, int]] = field(default_factory=list)
    # L3 / L5
    prepared_subholes: List[PreparedSubhole] = field(default_factory=list)
    feature_arrangement: Optional[FeatureArrangement] = None
    partition_obstacles: List[PartitionObstacle] = field(default_factory=list)
    recovery_diagnostics: Dict[str, object] = field(default_factory=dict)
    fill_plan: Optional[FillPlan] = None
    skipped_intersection_recovery: bool = False


def _compute_hole_scale(vertices: np.ndarray, loop: Sequence[int]) -> HoleScale:
    loop_pts = vertices[np.asarray(loop, dtype=np.int64)]
    perimeter = max(_loop_perimeter(vertices, loop), 1e-12)
    mean_edge = _mean_hole_loop_edge_len(vertices, loop)
    diag = max(float(_bbox_diag(loop_pts)), 1e-12)
    return HoleScale(
        loop_perimeter=float(perimeter),
        mean_edge_length=float(max(mean_edge, 1e-15)),
        bbox_diag=float(diag),
    )


def _partition_obstacles_from_arrangement(
    feature_arrangement: Optional[FeatureArrangement],
) -> List[PartitionObstacle]:
    if feature_arrangement is None:
        return []
    raw = feature_arrangement.diagnostics.get("partition_obstacles_l3")
    if not isinstance(raw, list):
        return []
    out: List[PartitionObstacle] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        out.append(
            PartitionObstacle(
                kind=str(item.get("kind") or PARTITION_OBSTACLE_O4),
                label=(
                    int(item["label"])
                    if item.get("label") is not None
                    else None
                ),
                detail=str(item.get("detail") or ""),
            )
        )
    return out


def _extract_partition_obstacles(
    fill_classification: Optional[FillPatchClassification],
    prepared: Sequence[PreparedSubhole],
    feature_arrangement: Optional[FeatureArrangement],
) -> List[PartitionObstacle]:
    explicit = _partition_obstacles_from_arrangement(feature_arrangement)
    if explicit:
        return explicit
    if fill_classification is None:
        return []
    expected = {int(x) for x in fill_classification.active_fill_labels}
    got = {int(p.patch_label) for p in prepared}
    if not got and feature_arrangement is not None:
        got = {int(cell.patch_label) for cell in feature_arrangement.cells}
    missing = sorted(expected - got)
    if not missing:
        return []
    detail = ""
    if feature_arrangement is not None:
        diag = feature_arrangement.diagnostics
        detail = str(
            diag.get("subhole_rejection")
            or diag.get("fill_reject_reason")
            or ""
        )
    return [
        PartitionObstacle(
            kind=PARTITION_OBSTACLE_O1,
            label=int(label),
            detail=detail or f"missing_active_wedge label={label}",
        )
        for label in missing
    ]


# ---------------------------------------------------------------------------
# HoleAnalyzer — L1 → L2 → L3 → L5 编排
# ---------------------------------------------------------------------------


class HoleAnalyzer:
    """分析孔洞边界附近的局部面域结构。"""

    def __init__(
        self,
        *,
        neighborhood_rings: int = 4,
        normal_angle_deg: float = 35.0,
        feature_point_normal_deg: float = 30.0,
        max_surface_patches: int = 7,
        collect_diagnostics: bool = True,
    ) -> None:
        self.neighborhood_rings = max(1, int(neighborhood_rings))
        self.normal_angle_deg = float(normal_angle_deg)
        self.feature_point_normal_deg = float(feature_point_normal_deg)
        self.max_surface_patches = max(1, int(max_surface_patches))
        self.collect_diagnostics = bool(collect_diagnostics)

    def analyze(self, mesh: HalfEdgeMesh, loop: Sequence[int]) -> HoleAnalysis:
        loop_list = [int(v) for v in loop]
        if len(loop_list) < 3:
            raise ValueError("loop 顶点数不足 3，无法分析孔洞结构")
        return self._run_analysis_pipeline(mesh, loop_list)

    def _pipeline_l1_boundary(self, state: HoleFillWorkState) -> None:
        """L1：邻域感知、拟合、孔边弧 — 不涉及补洞份数判定。"""
        mesh = state.mesh
        loop = state.loop
        vertices = state.vertices
        faces = state.faces
        face_normals, _ = _compute_face_normals_and_areas(vertices, faces)
        vertex_to_faces = _build_vertex_to_faces(len(vertices), faces)
        face_neighbors = _build_face_edge_neighbors(faces)
        boundary_half_edges = _boundary_half_edges_for_loop(mesh, loop)
        boundary_edge_supports = _boundary_edge_incident_face_supports(
            loop,
            boundary_half_edges,
            mesh,
            vertex_to_faces,
        )
        boundary_support_faces = _flatten_boundary_edge_supports(boundary_edge_supports)
        neighborhood_faces = _collect_neighborhood_faces(
            loop,
            vertex_to_faces,
            face_neighbors,
            self.neighborhood_rings,
        )
        neighborhood_faces = sorted(
            set(int(fi) for fi in neighborhood_faces)
            | set(int(fi) for fi in boundary_support_faces)
        )
        boundary_seed_faces = boundary_support_faces or _boundary_seed_faces(loop, vertex_to_faces)
        face_labels = _cluster_faces_by_normal_connectivity(
            neighborhood_faces,
            face_neighbors,
            face_normals,
            self.max_surface_patches,
            self.normal_angle_deg,
            boundary_seed_faces=boundary_seed_faces,
        )
        face_labels, patch_face_indices, patch_surface_fits = _refine_patch_partition_with_surface_fit(
            vertices,
            faces,
            face_labels,
            self.max_surface_patches,
        )
        patch_surface_fits = _upgrade_patch_surface_fits_semiglobal(
            vertices,
            faces,
            face_neighbors,
            patch_face_indices,
            patch_surface_fits,
        )
        cluster_surface_fits = dict(patch_surface_fits)

        feature_point_positions = _feature_point_candidates(
            vertices,
            faces,
            loop,
            vertex_to_faces,
            self.feature_point_normal_deg,
        )
        initial_arcs = _extract_boundary_arcs_from_feature_points(
            loop,
            feature_point_positions,
        )
        boundary_arcs, patch_surface_fits, arc_edge_labels = _arc_semiglobal_surface_fits(
            vertices,
            faces,
            loop,
            boundary_half_edges,
            mesh,
            face_labels,
            vertex_to_faces,
            boundary_edge_supports,
            face_neighbors,
            initial_arcs,
            patch_surface_fits,
            allowed_faces=set(int(fi) for fi in neighborhood_faces),
        )
        boundary_edge_labels = (
            [int(x) for x in arc_edge_labels]
            if arc_edge_labels
            else [0] * len(loop)
        )
        source_to_surface_id: Dict[int, int] = {}
        if boundary_arcs:
            boundary_edge_labels, boundary_arcs, patch_surface_fits = _merge_equivalent_boundary_arc_labels(
                boundary_edge_labels,
                boundary_arcs,
                patch_surface_fits,
            )
            patch_surface_fits = _arc_fit_subset(boundary_arcs, patch_surface_fits)
            (
                boundary_edge_labels,
                boundary_arcs,
                patch_surface_fits,
                source_to_surface_id,
            ) = _relabel_boundary_by_surface_id(
                loop,
                boundary_edge_labels,
                boundary_arcs,
                patch_surface_fits,
            )

        surface_face_labels = _surface_face_labels_from_fit_distance(
            vertices,
            faces,
            face_labels,
            cluster_surface_fits,
            patch_surface_fits,
            source_to_surface_id,
        )
        _force_boundary_arc_seed_surface_labels(
            surface_face_labels,
            boundary_edge_supports,
            boundary_arcs,
        )
        surface_patch_face_indices = _build_patch_face_indices(surface_face_labels)

        boundary_vertex_labels = _boundary_vertex_patch_labels(loop, boundary_edge_labels)
        if boundary_arcs:
            boundary_vertex_labels = _augment_boundary_vertex_labels_from_surface_faces(
                boundary_vertex_labels,
                loop,
                vertex_to_faces,
                surface_face_labels,
            )
        arcs = (
            boundary_arcs
            if boundary_arcs
            else _extract_boundary_arcs_from_feature_points(loop, feature_point_positions)
        )

        state.neighborhood_faces = list(neighborhood_faces)
        state.boundary_half_edges = list(boundary_half_edges)
        state.boundary_edge_supports = [
            [int(fi) for fi in row] for row in boundary_edge_supports
        ]
        state.boundary_edge_labels = boundary_edge_labels
        state.boundary_vertex_labels = dict(boundary_vertex_labels)
        state.boundary_arcs = list(boundary_arcs)
        state.arcs = list(arcs)
        state.patch_surface_fits = dict(patch_surface_fits)
        state.surface_face_labels = dict(surface_face_labels)
        state.surface_patch_face_indices = dict(surface_patch_face_indices)
        state.face_labels = dict(face_labels)
        state.unique_patch_count = len(set(int(x) for x in boundary_edge_labels))
        state.hole_center = _loop_centroid(vertices, loop)
        state.feature_point_positions = list(feature_point_positions)
        state.feature_point_vertex_ids = [
            int(loop[int(i)]) for i in feature_point_positions
        ]
        state.feature_edges = _feature_edge_candidates(loop, feature_point_positions)

    def _pipeline_l2_ownership_and_curves(self, state: HoleFillWorkState) -> None:
        """L2：一次性定稿 K→M，再恢复/裁剪交线；下游只读 ownership_snapshot。"""
        vertices = state.vertices
        loop = state.loop
        arcs = state.arcs
        patch_surface_fits = state.patch_surface_fits
        hole_center = state.hole_center
        loop_step = float(state.scale.mean_edge_length)
        faces_arr = np.asarray(state.faces, dtype=np.int64)

        (
            fill_classification,
            arcs_refined,
            display_curves,
            prune_diagnostics,
            junction_point,
            junction_confidence,
            bounded_segments,
            analytic_curves,
            cavity_result,
        ) = _finalize_fill_ownership(
            vertices=vertices,
            faces_arr=faces_arr,
            loop=loop,
            arcs=arcs,
            face_labels=state.face_labels,
            boundary_edge_supports=state.boundary_edge_supports,
            boundary_vertex_labels=state.boundary_vertex_labels,
            boundary_edge_labels=state.boundary_edge_labels,
            patch_surface_fits=patch_surface_fits,
            hole_center=hole_center,
            loop_step=loop_step,
            unique_patch_count=state.unique_patch_count,
            feature_point_vertex_ids=state.feature_point_vertex_ids,
        )
        state.arcs = list(arcs_refined)
        state.boundary_arcs = list(arcs_refined)
        state.fill_classification = fill_classification
        state.ownership_snapshot = fill_classification.ownership_snapshot
        state.bounded_segments = list(bounded_segments)
        state.analytic_curves = list(analytic_curves)
        state.junction_point = junction_point
        state.junction_confidence = str(junction_confidence)

        display_curves = _snap_curve_endpoints_to_mesh_vertices(
            list(display_curves),
            vertices,
            loop,
        )
        if cavity_result is not None:
            state.cavity_arrangement_result = cavity_result
        layout_finalized = bool(
            cavity_result is not None
            and bool(
                (getattr(cavity_result, "diagnostics", None) or {}).get(
                    "layout_finalized", False
                )
            )
        )
        if not layout_finalized:
            retained_junctions = _retained_cavity_junction_positions(cavity_result)
            display_curves = _truncate_virtual_endpoints_to_curve_arrangement(
                list(display_curves),
                patch_surface_fits,
                hole_center,
                float(loop_step) if loop_step > 1e-15 else None,
                retained_junction_positions=retained_junctions or None,
            )
        display_curves = _supplement_boundary_arc_endpoint_span_curves(
            vertices,
            arcs_refined,
            display_curves,
            patch_surface_fits,
            state.boundary_vertex_labels,
            hole_center,
            float(loop_step) if loop_step > 1e-15 else None,
            set(int(x) for x in fill_classification.active_fill_labels),
        )
        inactive_feature_vertices = set(
            int(v) for v in (state.ownership_snapshot.demoted_feature_points if state.ownership_snapshot else ())
        )
        active_feature_vertices = sorted(
            int(v)
            for v in fill_classification.active_feature_points
            if int(v) not in inactive_feature_vertices
        )
        loop_index_by_vertex = {int(v): int(i) for i, v in enumerate(loop)}
        state.active_feature_point_positions = sorted(
            {
                int(loop_index_by_vertex[int(v)])
                for v in active_feature_vertices
                if int(v) in loop_index_by_vertex
            }
        )
        state.feature_point_vertex_ids = list(active_feature_vertices)
        state.intersection_curves = _filter_layout_curves(
            _resample_sparse_arrangement_curves(
            vertices,
            loop,
                display_curves,
            patch_surface_fits,
        )
        )
        state.prune_diagnostics = dict(prune_diagnostics)
        state.skipped_intersection_recovery = bool(
            prune_diagnostics.get("skipped_intersection_recovery")
        )
        state.feature_edges = _feature_edge_candidates(
            loop,
            state.active_feature_point_positions,
        )

    def _pipeline_l3_partition(self, state: HoleFillWorkState) -> None:
        """L3：在 ownership 与交线定稿后构造子孔剖分。"""
        fill_classification = state.fill_classification
        if fill_classification is None:
            return
        layout_curves = list(state.intersection_curves)
        prepared, feature_arrangement = _prepare_subholes(
            state.vertices,
            state.faces,
            state.loop,
            state.arcs,
            state.unique_patch_count,
            state.boundary_vertex_labels,
            layout_curves,
            state.patch_surface_fits,
            fill_classification=fill_classification,
        )
        state.prepared_subholes = list(prepared)
        state.feature_arrangement = feature_arrangement
        state.partition_obstacles = _extract_partition_obstacles(
            fill_classification,
            prepared,
            feature_arrangement,
        )
        state.fill_plan = build_fill_plan(
            state.unique_patch_count,
            fill_classification,
            skipped_intersection_recovery=state.skipped_intersection_recovery,
        )

        prune_diagnostics = dict(state.prune_diagnostics)
        prune_diagnostics["layout_curve_pairs"] = [
            _curve_pair_key(c) for c in layout_curves
        ]
        active_labels = sorted(int(x) for x in fill_classification.active_fill_labels)
        partition_mode = "curve_arc_partition"
        if feature_arrangement is not None:
            part = feature_arrangement.diagnostics.get("subhole_partition")
            if part is not None:
                partition_mode = str(part)
        prune_diagnostics["carrier_boundary_mode"] = (
            "full_loop" if len(active_labels) == 1 else partition_mode
        )
        state.prune_diagnostics = prune_diagnostics
        ownership_diag: Dict[str, object] = {}
        if state.ownership_snapshot is not None:
            ownership_diag = _ownership_snapshot_to_dict(state.ownership_snapshot)
        l1_feature_vertex_ids = sorted(
            int(state.loop[int(i)])
            for i in state.feature_point_positions
            if 0 <= int(i) < len(state.loop)
        )
        endpoint_remap_diag: Optional[Dict[str, object]] = None
        if feature_arrangement is not None:
            cap = feature_arrangement.diagnostics.get("curve_arc_partition")
            if isinstance(cap, dict):
                raw_remap = cap.get("endpoint_remap")
                if isinstance(raw_remap, dict) and raw_remap:
                    endpoint_remap_diag = {int(k): int(v) for k, v in raw_remap.items()}
        state.recovery_diagnostics = {
            "mode": str(state.fill_plan.fill_strategy),
            "fill_strategy": str(state.fill_plan.fill_strategy),
            "pipeline": "L1_L2_L3",
            "n_curves": int(len(layout_curves)),
            "n_layout_curves": int(len(layout_curves)),
            "fill_label_count": int(len(active_labels)),
            "cluster_patch_count": int(state.unique_patch_count),
            "boundary_patch_count": int(state.fill_plan.boundary_patch_count),
            "skipped_intersection_recovery": bool(
                state.fill_plan.skipped_intersection_recovery
            ),
            "l1_feature_point_vertex_ids": l1_feature_vertex_ids,
            "layout_curve_endpoints": [
                [
                    int(c.endpoint_vertex_indices[0]),
                    int(c.endpoint_vertex_indices[1]),
                ]
                for c in layout_curves
            ],
            **ownership_diag,
            **prune_diagnostics,
            "partition_obstacles": [
                {"kind": o.kind, "label": o.label, "detail": o.detail}
                for o in state.partition_obstacles
            ],
        }
        if endpoint_remap_diag is not None:
            state.recovery_diagnostics["endpoint_remap"] = endpoint_remap_diag
        if feature_arrangement is not None:
            feature_arrangement.diagnostics["curve_recovery"] = dict(
                state.recovery_diagnostics
            )
            feature_arrangement.diagnostics["partition_obstacles"] = [
                {"kind": o.kind, "label": o.label, "detail": o.detail}
                for o in state.partition_obstacles
            ]

    def _pipeline_package_analysis(self, state: HoleFillWorkState) -> HoleAnalysis:
        """L5：打包对外契约（HoleAnalysis + fill_gate）。"""
        fill_classification = state.fill_classification
        if fill_classification is None:
            raise RuntimeError("L2 ownership 未产出 fill_classification")
        curves = list(state.intersection_curves)
        junction_point, junction_confidence = _coerce_junction_for_analysis(
            curves,
            state.junction_point,
            state.junction_confidence,
        )
        fill_gate = _build_fill_gate(
            state.prepared_subholes,
            fill_classification,
            state.feature_arrangement,
            unique_patch_count=state.unique_patch_count,
        )
        diagnostics: Optional[AnalysisDiagnostics] = None
        if self.collect_diagnostics:
            diagnostics = build_analysis_diagnostics(
                loop=state.loop,
                boundary_edge_labels=state.boundary_edge_labels,
                active_feature_point_positions=state.active_feature_point_positions,
                feature_point_vertex_ids=state.feature_point_vertex_ids,
                feature_edges=state.feature_edges,
                boundary_vertex_labels=state.boundary_vertex_labels,
                arcs=state.arcs,
                vertices=state.vertices,
                curves=curves,
                bounded_segments=state.bounded_segments,
                analytic_curves=state.analytic_curves,
                junction_point=junction_point,
                junction_confidence=junction_confidence,
                prepared=state.prepared_subholes,
                patch_surface_fits=state.patch_surface_fits,
                recovery_diagnostics=state.recovery_diagnostics,
                neighborhood_face_indices=state.neighborhood_faces,
                surface_patch_labels=dict(
                    sorted(state.surface_face_labels.items(), key=lambda x: x[0])
                ),
                patch_face_indices=state.surface_patch_face_indices,
                boundary_half_edges=state.boundary_half_edges,
                feature_arrangement=state.feature_arrangement,
            )
        return HoleAnalysis(
            boundary_vertices=list(state.loop),
            patch_surface_fits=state.patch_surface_fits,
            boundary_edge_patch_labels=state.boundary_edge_labels,
            boundary_vertex_patch_labels=state.boundary_vertex_labels,
            boundary_arcs=state.arcs,
            intersection_curves=curves,
            junction_point=None
            if junction_point is None
            else np.asarray(junction_point, dtype=np.float64),
            junction_confidence=junction_confidence,
            prepared_subholes=state.prepared_subholes,
            hole_type=_infer_public_hole_type(state.unique_patch_count),
            fill_plan=state.fill_plan,
            fill_classification=fill_classification,
            fill_gate=fill_gate,
            hole_scale=state.scale,
            partition_obstacles=list(state.partition_obstacles),
            diagnostics=diagnostics,
        )

    def _run_analysis_pipeline(self, mesh: HalfEdgeMesh, loop: List[int]) -> HoleAnalysis:
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        state = HoleFillWorkState(
            mesh=mesh,
            loop=loop,
            vertices=vertices,
            faces=faces,
            scale=_compute_hole_scale(vertices, loop),
        )
        self._pipeline_l1_boundary(state)
        self._pipeline_l2_ownership_and_curves(state)
        self._pipeline_l3_partition(state)
        return self._pipeline_package_analysis(state)
