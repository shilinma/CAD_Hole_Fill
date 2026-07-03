#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python run_cad_fill.py hole_data/fandisk/case_0019.obj --hole-clean --debug
用法 python run_cad_fill.py hole_data/hole_cases_cadxxxxxxxx/case_0000.obj --hole-clean --debug

"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# 保证从任意工作目录可直接运行
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import trimesh
from trimesh.path.entities import Line
from trimesh.path.path import Path3D

from libs.hole_patch_triangulation import (
    assess_patch_boundary_readiness,
    triangulate_hole_patch,
    triangulate_ordered_hole_boundary,
)
from libs.hole_analyzer import (
    FillValidationError,
    HoleAnalyzer,
    _arrangement_endpoint_sources_for_curve,
    _estimate_subhole_reference_normal,
    seam_constrained_edges_for_subhole,
    validate_before_partitioned_fill,
)
from libs.hole_analysis_types import PARTITION_OBSTACLE_O3, PARTITION_OBSTACLE_O4
from libs.hole_boundary_clean import clean_hole_boundary_tooth_faces
from libs.hole_detector import HoleDetector
from libs.surface_parameterization import lift_parameter_point, parameterize_boundary
from libs.surface_fitting import (
    SurfaceFit,
    is_analytic_surface_type,
    project_point_to_surface,
)


def _parse_rgba(color: str) -> np.ndarray:
    parts = [x.strip() for x in color.split(",")]
    if len(parts) not in {3, 4}:
        raise ValueError("颜色需要是 'R,G,B' 或 'R,G,B,A' 格式")
    values = [int(x) for x in parts]
    if any(v < 0 or v > 255 for v in values):
        raise ValueError("颜色通道必须在 0..255 范围内")
    if len(values) == 3:
        values.append(255)
    return np.array(values, dtype=np.uint8)


def _apply_patch_face_color(
    mesh: trimesh.Trimesh,
    patch_face_indices: np.ndarray,
    patch_color: np.ndarray,
) -> trimesh.Trimesh:
    if patch_face_indices.size == 0:
        return mesh
    face_count = int(mesh.faces.shape[0])
    face_colors = None
    if hasattr(mesh.visual, "face_colors") and len(mesh.visual.face_colors) == face_count:
        face_colors = np.array(mesh.visual.face_colors, dtype=np.uint8, copy=True)
    else:
        face_colors = np.tile(
            np.array([200, 200, 200, 255], dtype=np.uint8).reshape(1, 4),
            (face_count, 1),
        )
    face_colors[np.asarray(patch_face_indices, dtype=np.int64)] = np.asarray(
        patch_color, dtype=np.uint8
    ).reshape(1, 4)
    mesh.visual.face_colors = face_colors
    return mesh


def _make_colored_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    color: np.ndarray,
) -> trimesh.Trimesh:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if int(faces.shape[0]) > 0:
        mesh.visual.face_colors = np.tile(
            np.asarray(color, dtype=np.uint8).reshape(1, 4),
            (int(faces.shape[0]), 1),
        )
    return mesh


def _make_original_debug_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> trimesh.Trimesh:
    """GLB 底模：不着色面片，仅保留原始三角网格。"""
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _surface_debug_color(surface_type: str, confidence: str) -> np.ndarray:
    base = {
        "plane": np.array([70, 160, 255, 210], dtype=np.uint8),
        "cylinder": np.array([255, 170, 70, 210], dtype=np.uint8),
        "sphere": np.array([90, 210, 120, 210], dtype=np.uint8),
        "cone": np.array([200, 110, 255, 210], dtype=np.uint8),
        "transition_surface": np.array([240, 120, 170, 190], dtype=np.uint8),
        "freeform_fallback": np.array([180, 180, 180, 180], dtype=np.uint8),
    }.get(surface_type, np.array([180, 180, 180, 180], dtype=np.uint8))
    alpha = {
        "high": 220,
        "medium": 190,
        "low": 150,
    }.get(confidence, int(base[3]))
    out = base.copy()
    out[3] = np.uint8(alpha)
    return out


def _curve_debug_color(confidence: str) -> np.ndarray:
    return {
        "high": np.array([255, 60, 60, 255], dtype=np.uint8),
        "medium": np.array([255, 140, 40, 255], dtype=np.uint8),
        "low": np.array([255, 220, 80, 255], dtype=np.uint8),
    }.get(confidence, np.array([255, 80, 80, 255], dtype=np.uint8))


def _estimated_debug_radius(vertices: np.ndarray) -> float:
    diag = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    return max(1e-4, 0.003 * max(diag, 1.0))


def _debug_cylinder_segment(
    a: np.ndarray,
    b: np.ndarray,
    radius: float,
    *,
    sections: int = 8,
) -> trimesh.Trimesh | None:
    """细圆柱表示线段，用于 debug 折线与参考弦。"""
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    d = b - a
    h = float(np.linalg.norm(d))
    if h < 1e-12 * max(float(np.linalg.norm(a)), 1.0):
        return None
    direction = d / h
    cyl = trimesh.creation.cylinder(
        radius=max(radius, 1e-8),
        height=h,
        sections=int(sections),
    )
    rotation, _angle = trimesh.geometry.align_vectors(
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
        direction,
        return_angle=True,
    )
    cyl.apply_transform(rotation)
    cyl.apply_translation(0.5 * (a + b))
    return cyl


def _debug_polyline_as_tubes(
    points: np.ndarray,
    tube_radius: float,
    face_color: np.ndarray,
) -> trimesh.Trimesh | None:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 3:
        return None
    parts: list[trimesh.Trimesh] = []
    for i in range(int(pts.shape[0]) - 1):
        seg = _debug_cylinder_segment(pts[i], pts[i + 1], tube_radius)
        if seg is None:
            continue
        fc = np.asarray(face_color, dtype=np.uint8).reshape(1, 4)
        seg.visual.face_colors = np.tile(fc, (int(seg.faces.shape[0]), 1))
        parts.append(seg)
    if not parts:
        return None
    return trimesh.util.concatenate(tuple(parts))


def _boundary_feature_point_debug_color() -> np.ndarray:
    """孔边界上检测到的原始特征点（mesh 顶点）。"""
    return np.array([40, 220, 255, 255], dtype=np.uint8)


def _demoted_feature_point_debug_color() -> np.ndarray:
    """已降为普通边界点的原特征点。"""
    return np.array([150, 150, 165, 180], dtype=np.uint8)


def _removed_intersection_curve_debug_color() -> np.ndarray:
    """因 patch 不可达而被修剪掉的交线。"""
    return np.array([120, 120, 130, 120], dtype=np.uint8)


def _recovered_intersection_point_debug_color() -> np.ndarray:
    """分析阶段后求交点（线线汇交、虚拟端点、几何端点与 mesh 错位等）。"""
    return np.array([255, 165, 50, 255], dtype=np.uint8)


def _prepared_subhole_debug_color(label: int) -> np.ndarray:
    """L3 实际送入 L4 三角化的子孔边界。"""
    palette = (
        np.array([210, 60, 255, 255], dtype=np.uint8),
        np.array([60, 255, 150, 255], dtype=np.uint8),
        np.array([80, 120, 255, 255], dtype=np.uint8),
        np.array([255, 80, 180, 255], dtype=np.uint8),
        np.array([120, 255, 255, 255], dtype=np.uint8),
        np.array([255, 210, 80, 255], dtype=np.uint8),
    )
    return palette[int(label) % len(palette)]


def _debug_point_tolerance(vertices: np.ndarray) -> float:
    diag = float(np.linalg.norm(np.ptp(np.asarray(vertices, dtype=np.float64), axis=0)))
    return max(1e-9, 1e-6 * max(diag, 1.0))


def _dedupe_xyz_points(
    points: Iterable[np.ndarray],
    *,
    tol: float,
) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for raw in points:
        p = np.asarray(raw, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(p)):
            continue
        if any(float(np.linalg.norm(p - q)) <= tol for q in out):
            continue
        out.append(p)
    return out


def _recovery_diagnostics(analysis) -> dict:
    diag = getattr(analysis, "diagnostics", None)
    if diag is None:
        return {}
    return dict(getattr(diag, "recovery_diagnostics", None) or {})


def _endpoint_remap_diagnostics(analysis) -> dict:
    recovery = _recovery_diagnostics(analysis)
    raw = recovery.get("endpoint_remap")
    if isinstance(raw, dict):
        return {int(k): int(v) for k, v in raw.items()}
    diag = getattr(analysis, "diagnostics", None)
    if diag is None:
        return {}
    arrangement = getattr(diag, "feature_arrangement", None)
    if arrangement is None:
        return {}
    cap = getattr(arrangement, "diagnostics", None) or {}
    part = cap.get("curve_arc_partition")
    if not isinstance(part, dict):
        return {}
    ref = part.get("endpoint_remap")
    return {int(k): int(v) for k, v in ref.items()} if isinstance(ref, dict) else {}


def _curve_arc_partition_diagnostics(analysis) -> dict:
    diag = getattr(analysis, "diagnostics", None)
    if diag is None:
        return {}
    arrangement = getattr(diag, "feature_arrangement", None)
    if arrangement is None:
        return {}
    cap = getattr(arrangement, "diagnostics", None) or {}
    part = cap.get("curve_arc_partition")
    return dict(part) if isinstance(part, dict) else {}


def _arrangement_validation_diagnostics(analysis) -> dict:
    recovery = _recovery_diagnostics(analysis)
    cavity_layout = recovery.get("cavity_layout")
    return dict(cavity_layout) if isinstance(cavity_layout, dict) else {}


def _arrangement_gate_required(analysis) -> bool:
    """Only hard-gate arrangement proof when final fill uses arrangement curves."""
    subholes = list(getattr(analysis, "prepared_subholes", None) or [])
    if any(str(getattr(sub, "closure_kind", "")) == "curve_arc_partition" for sub in subholes):
        return True
    curves = list(getattr(analysis, "intersection_curves", None) or [])
    if curves:
        return True
    return False


def _final_prepared_subholes_ready(analysis) -> bool:
    fill_gate = getattr(analysis, "fill_gate", None)
    if fill_gate is None or not bool(getattr(fill_gate, "accepted", False)):
        return False
    prepared = list(getattr(analysis, "prepared_subholes", None) or [])
    if not prepared:
        return False
    for subhole in prepared:
        readiness = assess_patch_boundary_readiness(
            np.asarray(subhole.closed_boundary_points, dtype=np.float64),
            np.asarray(subhole.boundary_points_2d, dtype=np.float64),
        )
        if not bool(readiness.get("ready", False)):
            return False
    return True


def _arrangement_invalid_superseded_by_l3(analysis) -> bool:
    """L3 endpoint remap can supersede stale L2 cell proof if final subholes are ready."""
    if not _endpoint_remap_diagnostics(analysis):
        return False
    return _final_prepared_subholes_ready(analysis)


def _arrangement_invalid_superseded_by_ready_cells(analysis, cavity_layout: dict) -> bool:
    """Final L3 subholes may supersede stale arrangement certificates once cells prove valid."""
    if not bool(cavity_layout.get("cell_validation_valid", False)):
        return False
    return _final_prepared_subholes_ready(analysis)


def _arrangement_invalid_superseded_by_support_subholes(analysis) -> bool:
    """L3-added support-arc subholes cover support strips absent from stale L2 cell proof."""
    part = _curve_arc_partition_diagnostics(analysis)
    support_subholes = part.get("support_bridge_subholes")
    if not isinstance(support_subholes, list) or not support_subholes:
        return False
    return _final_prepared_subholes_ready(analysis)


def _final_l3_layout_coverage_diagnostics(analysis) -> dict:
    """Prove final L3 rings cover all active layout curve endpoints.

    L2 cell proof can become stale after L3 demotion, endpoint refinement, or support
    bridge splitting. This certificate uses the actual ``PreparedSubhole`` rings that
    L4 consumes, so it may supersede dangling-only L2 failures without hiding real
    readiness or residual-loop errors.
    """
    if not _final_prepared_subholes_ready(analysis):
        return {"accepted": False, "reason": "final_subholes_not_ready"}
    fill_gate = getattr(analysis, "fill_gate", None)
    expected = {
        int(x)
        for x in getattr(fill_gate, "expected_labels", frozenset())
    }
    got = {int(x) for x in getattr(fill_gate, "got_labels", frozenset())}
    if not expected or got != expected:
        return {
            "accepted": False,
            "reason": "label_mismatch",
            "expected_labels": sorted(expected),
            "got_labels": sorted(got),
        }
    prepared = list(getattr(analysis, "prepared_subholes", None) or [])
    sources_by_label: Dict[int, Set[int]] = defaultdict(set)
    subholes_by_label: Dict[int, int] = defaultdict(int)
    for subhole in prepared:
        label = int(getattr(subhole, "patch_label", -1))
        if label not in expected:
            continue
        subholes_by_label[label] += 1
        sources_by_label[label].update(
            int(src)
            for src in getattr(subhole, "boundary_sources", []) or []
        )
    if set(sources_by_label) != expected:
        return {
            "accepted": False,
            "reason": "missing_label_sources",
            "expected_labels": sorted(expected),
            "source_labels": sorted(sources_by_label),
        }

    endpoint_remap = _endpoint_remap_diagnostics(analysis)
    curves = list(getattr(analysis, "intersection_curves", None) or [])
    required: List[Dict[str, object]] = []
    missing: List[Dict[str, object]] = []
    for curve_idx, curve in enumerate(curves):
        pair = tuple(int(x) for x in getattr(curve, "patch_pair", ()))
        active_labels = [label for label in pair if label in expected]
        if not active_labels:
            continue
        e0, e1 = _arrangement_endpoint_sources_for_curve(curve, int(curve_idx))
        endpoints = []
        for endpoint in (int(e0), int(e1)):
            if endpoint >= 0:
                endpoint = int(endpoint_remap.get(endpoint, endpoint))
            endpoints.append(int(endpoint))
        for label in active_labels:
            label_sources = sources_by_label.get(int(label), set())
            absent = [int(src) for src in endpoints if int(src) not in label_sources]
            record = {
                "curve_index": int(curve_idx),
                "label": int(label),
                "patch_pair": [int(x) for x in pair],
                "endpoints": [int(x) for x in endpoints],
            }
            required.append(record)
            if absent:
                missing.append({**record, "missing": absent})
    if not required:
        return {"accepted": False, "reason": "no_active_layout_curves"}
    if missing:
        return {
            "accepted": False,
            "reason": "missing_layout_endpoint_sources",
            "missing": missing,
        }
    return {
        "accepted": True,
        "expected_labels": sorted(expected),
        "subholes_by_label": {
            int(label): int(count) for label, count in sorted(subholes_by_label.items())
        },
        "covered_layout_endpoints": required,
    }


def _arrangement_invalid_superseded_by_final_l3_coverage(analysis) -> bool:
    diag = _final_l3_layout_coverage_diagnostics(analysis)
    return bool(diag.get("accepted", False))


def _arrangement_invalid_superseded_by_complete_mesh_pairs(analysis, cavity_layout: dict) -> bool:
    """A complete mesh-mesh pair cover is validated by final L3 PreparedSubholes."""
    certificates = cavity_layout.get("boundary_pair_certificates")
    if not isinstance(certificates, list) or not certificates:
        return False
    accepted: set[tuple[int, int]] = set()
    covered: set[int] = set()
    for record in certificates:
        if not isinstance(record, dict) or not bool(record.get("accepted", False)):
            continue
        anchors = record.get("anchor_vertices")
        if not isinstance(anchors, (list, tuple)) or len(anchors) != 2:
            continue
        a, b = int(anchors[0]), int(anchors[1])
        if a == b:
            return False
        accepted.add((a, b) if a < b else (b, a))
        covered.update((a, b))
    fill_classification = getattr(analysis, "fill_classification", None)
    feature_points = set(
        int(v)
        for v in getattr(fill_classification, "active_feature_points", set())
    )
    if not feature_points:
        return False
    if len(feature_points) < 6 or len(feature_points) % 2 != 0:
        return False
    if covered != feature_points:
        return False
    if len(accepted) * 2 != len(feature_points):
        return False
    return _final_prepared_subholes_ready(analysis)


def _raise_if_invalid_arrangement(analysis) -> None:
    if not _arrangement_gate_required(analysis):
        return
    cavity_layout = _arrangement_validation_diagnostics(analysis)
    if not cavity_layout:
        return
    if bool(cavity_layout.get("arrangement_valid", True)):
        return
    if _arrangement_invalid_superseded_by_l3(analysis):
        return
    if _arrangement_invalid_superseded_by_ready_cells(analysis, cavity_layout):
        return
    if _arrangement_invalid_superseded_by_support_subholes(analysis):
        return
    if _arrangement_invalid_superseded_by_final_l3_coverage(analysis):
        return
    if _arrangement_invalid_superseded_by_complete_mesh_pairs(analysis, cavity_layout):
        return
    invalid = [
        item
        for item in cavity_layout.get("surface_cell_validation", []) or []
        if isinstance(item, dict) and not bool(item.get("valid", False))
    ]
    detail = "; ".join(
        (
            f"label={int(item.get('label', -1))} "
            f"dangling={item.get('dangling_nodes', [])} "
            f"self_cross={item.get('parameter_self_intersections', 0)}"
        )
        for item in invalid
    )
    if not detail:
        detail = "surface_cell_validation_failed"
    raise ValueError(
        "O4_non_wedge: 参数域 trimming-cell 验收失败，禁止进入 L4 补洞: "
        + detail
    )


def _collect_layout_curve_endpoints(analysis) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for curve in getattr(analysis, "intersection_curves", None) or []:
        e0 = int(curve.endpoint_vertex_indices[0])
        e1 = int(curve.endpoint_vertex_indices[1])
        out.append((e0, e1))
    return out


def _print_analysis_topology_summary(analysis) -> None:
    """打印特征点、layout 端点及 L3 端点替换摘要。"""
    fill_plan = getattr(analysis, "fill_plan", None)
    if fill_plan is not None:
        print(
            f"FillPlan: strategy={fill_plan.fill_strategy} "
            f"K={fill_plan.boundary_patch_count} "
            f"|M|={len(fill_plan.active_fill_labels)} "
            f"skip_intersection={fill_plan.skipped_intersection_recovery}"
        )
    recovery = _recovery_diagnostics(analysis)

    l1_fps = sorted(
        int(x) for x in recovery.get("l1_feature_point_vertex_ids", []) or []
    )
    active_fps = sorted(
        int(x)
        for x in (
            recovery.get("active_feature_points")
            or getattr(analysis, "feature_point_candidates", None)
            or []
        )
    )
    demoted_fps = sorted(int(x) for x in recovery.get("demoted_feature_points", []) or [])

    print(f"特征点(L1): {l1_fps if l1_fps else '—'}")
    print(f"特征点(L2活跃): {active_fps if active_fps else '—'}")
    if demoted_fps:
        print(f"特征点(已降级): {demoted_fps}")

    raw_eps = recovery.get("layout_curve_endpoints")
    if isinstance(raw_eps, list) and raw_eps:
        endpoints = [
            (int(pair[0]), int(pair[1]))
            for pair in raw_eps
            if isinstance(pair, (list, tuple)) and len(pair) >= 2
        ]
    else:
        endpoints = _collect_layout_curve_endpoints(analysis)
    if endpoints:
        ep_text = ", ".join(f"({a},{b})" for a, b in endpoints)
    else:
        ep_text = "—"
    print(f"Layout 交线端点: {ep_text}")

    ref = _endpoint_remap_diagnostics(analysis)
    if ref:
        parts = [f"{int(k)}→{int(v)}" for k, v in sorted(ref.items())]
        print(f"L3 端点替换: {'; '.join(parts)}")
    else:
        print("L3 端点替换: 无")

    subholes = getattr(analysis, "prepared_subholes", None) or []
    wedge_parts: List[str] = []
    for subhole in subholes:
        fp = getattr(subhole, "feature_point_vertex_indices", None)
        if fp is None:
            continue
        wedge_parts.append(
            f"label={int(subhole.patch_label)} ({int(fp[0])},{int(fp[1])})"
        )
    if wedge_parts:
        print(f"子孔楔角: {'; '.join(wedge_parts)}")


def _analysis_debug_legend_text() -> str:
    return (
        "调试图例（analysis_scene.glb / intersection_curves.ply）\n"
        "  橙/黄管  L3 使用的 layout 交线（confidence 着色）\n"
        "  灰管    ownership 移除的交线（active/support 不可达）\n"
        "  青球    L1 法向折角特征点候选（mesh 顶点）\n"
        "  灰球    已降级边界点（demoted）\n"
        "  橙球    L3 使用的虚拟汇交/几何端点\n"
        "  紫/绿/蓝粗管  L3 accepted prepared_subhole 边界（送入 L4 三角化）\n"
    )


def _collect_boundary_feature_debug_points(
    vertices: np.ndarray,
    analysis,
) -> List[np.ndarray]:
    """孔环上保留的活跃特征点（mesh 顶点）。"""
    tol = _debug_point_tolerance(vertices)
    nv = int(np.asarray(vertices).shape[0])
    raw = [
        np.asarray(vertices[int(vi)], dtype=np.float64)
        for vi in getattr(analysis, "feature_point_candidates", None) or []
        if 0 <= int(vi) < nv
    ]
    return _dedupe_xyz_points(raw, tol=tol)


def _collect_demoted_feature_debug_points(
    vertices: np.ndarray,
    analysis,
) -> List[np.ndarray]:
    """已降为普通边界点的原特征点。"""
    tol = _debug_point_tolerance(vertices)
    nv = int(np.asarray(vertices).shape[0])
    recovery = _recovery_diagnostics(analysis)
    demoted = (
        recovery.get("demoted_feature_points")
        or recovery.get("inactive_feature_points")
        or []
    )
    raw = [
        np.asarray(vertices[int(vi)], dtype=np.float64)
        for vi in demoted
        if 0 <= int(vi) < nv
    ]
    return _dedupe_xyz_points(raw, tol=tol)


def _collect_removed_intersection_debug_curves(analysis) -> list:
    recovery = _recovery_diagnostics(analysis)
    removed = recovery.get("removed_intersection_curves")
    if isinstance(removed, list) and removed:
        return list(removed)
    return []


def _layout_sources_used_by_prepared_subholes(analysis) -> Set[int]:
    """Negative boundary_sources actually consumed by accepted L3 subholes."""
    used: Set[int] = set()
    for subhole in getattr(analysis, "prepared_subholes", None) or []:
        for raw in getattr(subhole, "boundary_sources", None) or ():
            src = int(raw)
            if src < 0:
                used.add(src)
    return used


def _filter_layout_curves_for_debug(analysis) -> list:
    """
    Keep only layout curves that L3/L4 actually traverse.

    L2 may still retain pruned/optional arrangement segments for diagnostics;
    debug export must not visualize dangling mesh↔virtual or virtual↔virtual stubs.
    """
    curves = list(getattr(analysis, "intersection_curves", None) or [])
    if not curves:
        return curves
    used_sources = _layout_sources_used_by_prepared_subholes(analysis)
    if not used_sources:
        recovery = _recovery_diagnostics(analysis)
        cavity_layout = recovery.get("cavity_layout")
        if isinstance(cavity_layout, dict):
            raw_sources = cavity_layout.get("virtual_sources")
            if isinstance(raw_sources, list) and raw_sources:
                used_sources = {int(x) for x in raw_sources}
    if not used_sources:
        return curves

    filtered: list = []
    for curve_idx, curve in enumerate(curves):
        e0, e1 = _arrangement_endpoint_sources_for_curve(curve, int(curve_idx))
        v0, v1 = int(e0), int(e1)
        if v0 >= 0 and v1 >= 0:
            filtered.append(curve)
            continue
        if v0 < 0 and v1 < 0:
            if {v0, v1}.issubset(used_sources):
                filtered.append(curve)
            continue
        virtual = v0 if v0 < 0 else v1
        if int(virtual) in used_sources:
            filtered.append(curve)
    return filtered


def _collect_recovered_intersection_debug_points(
    vertices: np.ndarray,
    analysis,
) -> List[np.ndarray]:
    """后求交点：非边界特征点的汇交/虚拟端点/折线几何端点。"""
    from libs.surface_intersections import _point_in_loop_polygon_3d

    loop = [int(v) for v in getattr(analysis, "boundary_vertices", []) or []]
    tol = _debug_point_tolerance(vertices)
    loop_pts = (
        np.asarray(vertices, dtype=np.float64)[np.asarray(loop, dtype=np.int64)]
        if loop
        else np.zeros((0, 3), dtype=np.float64)
    )
    loop_diag = (
        max(float(np.linalg.norm(np.ptp(loop_pts, axis=0))), 1e-12)
        if loop_pts.size
        else max(float(np.linalg.norm(np.ptp(np.asarray(vertices, dtype=np.float64), axis=0))), 1.0)
    )
    loop_step = 0.0
    if loop_pts.shape[0] >= 2:
        loop_step = float(
            np.mean(np.linalg.norm(loop_pts - np.roll(loop_pts, -1, axis=0), axis=1))
        )
    virtual_merge_tol = max(1e-6 * loop_diag, 0.45 * loop_step, tol)
    nv = int(np.asarray(vertices).shape[0])
    fp_set = {
        int(vi)
        for vi in getattr(analysis, "feature_point_candidates", None) or []
    }
    boundary_pts = _collect_boundary_feature_debug_points(vertices, analysis)
    demoted_pts = _collect_demoted_feature_debug_points(vertices, analysis)
    candidates: List[np.ndarray] = []
    curves = _filter_layout_curves_for_debug(analysis)
    used_virtual_sources = _layout_sources_used_by_prepared_subholes(analysis)

    def _near_boundary_feature(p: np.ndarray) -> bool:
        return any(float(np.linalg.norm(p - q)) <= tol for q in boundary_pts)

    def _near_demoted_feature(p: np.ndarray) -> bool:
        return any(float(np.linalg.norm(p - q)) <= tol for q in demoted_pts)

    def _add(pt: np.ndarray, *, require_in_loop: bool = False) -> None:
        p = np.asarray(pt, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(p)):
            return
        if require_in_loop and len(loop) >= 3 and not _point_in_loop_polygon_3d(
            vertices, loop, p
        ):
            return
        if _near_boundary_feature(p) or _near_demoted_feature(p):
            return
        for idx, q in enumerate(candidates):
            if float(np.linalg.norm(p - q)) <= virtual_merge_tol:
                candidates[idx] = 0.5 * (q + p)
                return
        candidates.append(p)

    # 已接受 L3 子孔时，仅显示其 boundary_sources 使用的虚拟汇交，避免 prune 后残留黄球。
    if used_virtual_sources:
        for curve in curves:
            pts = np.asarray(curve.curve_points, dtype=np.float64)
            if pts.ndim != 2 or pts.shape[0] == 0:
                continue
            e0, e1 = (
                int(curve.endpoint_vertex_indices[0]),
                int(curve.endpoint_vertex_indices[1]),
            )
            if int(e0) in used_virtual_sources:
                _add(pts[0])
            if int(e1) in used_virtual_sources:
                _add(pts[-1])
    elif curves and len(loop) >= 3:
        from libs.hole_analyzer import _mean_hole_loop_edge_len, _virtual_endpoint_clusters

        loop_sampling = float(loop_step) if loop_step > 1e-15 else None
        if loop_sampling is None:
            loop_sampling = _mean_hole_loop_edge_len(vertices, loop)
        for center, _labels, _members in _virtual_endpoint_clusters(
            vertices, loop, curves, loop_sampling
        ):
            explicit_sources = {
                int(member[0].endpoint_vertex_indices[int(member[1])])
                for member in _members
                if int(member[0].endpoint_vertex_indices[int(member[1])]) < -900_000
            }
            if len(explicit_sources) > 1:
                continue
            _add(np.asarray(center, dtype=np.float64))

    for curve in curves:
        pts = np.asarray(curve.curve_points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] == 0:
            continue
        e0, e1 = (
            int(curve.endpoint_vertex_indices[0]),
            int(curve.endpoint_vertex_indices[1]),
        )
        if e0 < 0:
            _add(pts[0])
        elif e0 >= nv or int(e0) not in fp_set:
            _add(pts[0])
        elif float(np.linalg.norm(pts[0] - vertices[int(e0)])) > tol:
            _add(pts[0])
        if e1 < 0:
            _add(pts[-1])
        elif e1 >= nv or int(e1) not in fp_set:
            _add(pts[-1])
        elif float(np.linalg.norm(pts[-1] - vertices[int(e1)])) > tol:
            _add(pts[-1])

    has_virtual_bridge = any(
        int(curve.endpoint_vertex_indices[0]) < 0
        and int(curve.endpoint_vertex_indices[1]) < 0
        for curve in curves
    )
    if getattr(analysis, "junction_point", None) is not None and not has_virtual_bridge:
        _add(np.asarray(analysis.junction_point, dtype=np.float64), require_in_loop=True)

    return candidates


def _append_debug_point_markers(
    meshes: list[trimesh.Trimesh],
    points: Sequence[np.ndarray],
    *,
    radius: float,
    color_rgba: np.ndarray,
    subdivisions: int = 2,
) -> None:
    if not points:
        return
    marker = trimesh.creation.icosphere(subdivisions=int(subdivisions), radius=float(radius))
    marker.visual.face_colors = np.tile(
        np.asarray(color_rgba, dtype=np.uint8).reshape(1, 4),
        (int(marker.faces.shape[0]), 1),
    )
    for pt in points:
        meshes.append(_translated_mesh(marker.copy(), pt))


def _chord_reference_debug_color() -> np.ndarray:
    return np.array([190, 190, 205, 210], dtype=np.uint8)


def _intersection_curve_path3d(points: np.ndarray) -> Path3D | None:
    v = np.asarray(points, dtype=np.float64)
    if v.ndim != 2 or v.shape[0] < 2 or v.shape[1] != 3:
        return None
    n = int(v.shape[0])
    return Path3D(
        entities=[Line(np.arange(n, dtype=np.int64))],
        vertices=v,
        process=False,
        metadata={"kind": "intersection_polyline"},
    )


def _translated_mesh(template: trimesh.Trimesh, center: np.ndarray) -> trimesh.Trimesh:
    mesh = template.copy()
    mesh.apply_translation(np.asarray(center, dtype=np.float64).reshape(3))
    return mesh


def _concatenate_meshes(
    meshes: Iterable[trimesh.Trimesh],
) -> trimesh.Trimesh | None:
    valid = [
        mesh
        for mesh in meshes
        if isinstance(mesh, trimesh.Trimesh)
        and int(mesh.vertices.shape[0]) > 0
        and int(mesh.faces.shape[0]) > 0
    ]
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]
    return trimesh.util.concatenate(tuple(valid))


def _fit_patch_debug_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    fit: SurfaceFit,
) -> trimesh.Trimesh | None:
    face_ids = [int(fi) for fi in fit.support_face_indices]
    vertex_ids = [int(vi) for vi in fit.support_vertex_indices]
    if not face_ids or not vertex_ids:
        return None
    remap = {old: new for new, old in enumerate(vertex_ids)}
    projected = np.array(
        [project_point_to_surface(fit, vertices[idx]) for idx in vertex_ids],
        dtype=np.float64,
    )
    local_faces = []
    for fi in face_ids:
        tri = faces[int(fi)]
        try:
            local_faces.append([remap[int(v)] for v in tri])
        except KeyError:
            continue
    if not local_faces:
        return None
    return _make_colored_mesh(
        projected,
        np.asarray(local_faces, dtype=np.int64),
        _surface_debug_color(fit.surface_type, fit.fit_confidence),
    )


def _build_surface_fit_debug_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    analysis,
) -> trimesh.Trimesh | None:
    meshes = []
    for label in sorted(analysis.patch_surface_fits):
        fit = analysis.patch_surface_fits[int(label)]
        patch_mesh = _fit_patch_debug_mesh(vertices, faces, fit)
        if patch_mesh is not None:
            meshes.append(patch_mesh)
    return _concatenate_meshes(meshes)


def _build_boundary_edge_support_debug_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    analysis,
) -> trimesh.Trimesh | None:
    loop = [int(v) for v in getattr(analysis, "boundary_vertices", [])]
    if len(loop) < 2:
        return None
    face_sets: dict[int, set[int]] = defaultdict(set)
    for fi, tri in enumerate(np.asarray(faces, dtype=np.int64)):
        for vi in tri:
            face_sets[int(vi)].add(int(fi))

    support_faces: set[int] = set()
    n = len(loop)
    for i in range(n):
        u = int(loop[i])
        v = int(loop[(i + 1) % n])
        support_faces.update(face_sets.get(u, set()) & face_sets.get(v, set()))
    if not support_faces:
        return None

    face_ids = sorted(int(fi) for fi in support_faces)
    vertex_ids = sorted({int(v) for fi in face_ids for v in faces[int(fi)]})
    remap = {old: new for new, old in enumerate(vertex_ids)}
    local_faces = []
    for fi in face_ids:
        try:
            local_faces.append([remap[int(v)] for v in faces[int(fi)]])
        except KeyError:
            continue
    if not local_faces:
        return None
    return _make_colored_mesh(
        np.asarray(vertices, dtype=np.float64)[np.asarray(vertex_ids, dtype=np.int64)],
        np.asarray(local_faces, dtype=np.int64),
        np.array([255, 245, 80, 235], dtype=np.uint8),
    )


def _build_intersection_curve_debug_mesh(
    vertices: np.ndarray,
    analysis,
) -> trimesh.Trimesh | None:
    """交线 debug：橙/黄管=保留交线；灰管=已修剪；青/灰/橙球=特征点层级。"""
    base_r = _estimated_debug_radius(vertices)
    tube_r = max(1e-4, 0.52 * base_r)
    removed_tube_r = max(1e-4, 0.38 * base_r)
    fp_r = max(1e-4, 1.05 * base_r)
    demoted_r = max(1e-4, 0.82 * base_r)
    recovered_r = max(1e-4, 1.25 * base_r)
    meshes: list[trimesh.Trimesh] = []

    _append_debug_point_markers(
        meshes,
        _collect_boundary_feature_debug_points(vertices, analysis),
        radius=fp_r,
        color_rgba=_boundary_feature_point_debug_color(),
    )
    _append_debug_point_markers(
        meshes,
        _collect_demoted_feature_debug_points(vertices, analysis),
        radius=demoted_r,
        color_rgba=_demoted_feature_point_debug_color(),
    )
    _append_debug_point_markers(
        meshes,
        _collect_recovered_intersection_debug_points(vertices, analysis),
        radius=recovered_r,
        color_rgba=_recovered_intersection_point_debug_color(),
        subdivisions=3,
    )

    for curve in _collect_removed_intersection_debug_curves(analysis):
        pts = np.asarray(curve.curve_points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] == 0:
            continue
        poly = _debug_polyline_as_tubes(
            pts,
            removed_tube_r,
            _removed_intersection_curve_debug_color(),
        )
        if poly is not None:
            meshes.append(poly)

    for curve in _filter_layout_curves_for_debug(analysis):
        pts = np.asarray(curve.curve_points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] == 0:
            continue
        col = _curve_debug_color(curve.curve_confidence)
        poly = _debug_polyline_as_tubes(pts, tube_r, col)
        if poly is not None:
            meshes.append(poly)

    return _concatenate_meshes(meshes)


def _build_prepared_subhole_debug_mesh(
    vertices: np.ndarray,
    analysis,
) -> trimesh.Trimesh | None:
    """L3 accepted prepared_subholes: these are the boundaries actually triangulated in L4."""
    base_r = _estimated_debug_radius(vertices)
    tube_r = max(1e-4, 0.82 * base_r)
    meshes: list[trimesh.Trimesh] = []
    for subhole in getattr(analysis, "prepared_subholes", None) or []:
        pts = np.asarray(
            getattr(subhole, "closed_boundary_points", None),
            dtype=np.float64,
        )
        if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 3:
            continue
        if float(np.linalg.norm(pts[0] - pts[-1])) > max(1e-9, 1e-6 * max(float(np.linalg.norm(np.ptp(pts, axis=0))), 1.0)):
            pts = np.vstack([pts, pts[0]])
        color = _prepared_subhole_debug_color(int(getattr(subhole, "patch_label", 0)))
        poly = _debug_polyline_as_tubes(pts, tube_r, color)
        if poly is not None:
            meshes.append(poly)
    return _concatenate_meshes(meshes)


def _export_analysis_debug_visuals(
    vertices: np.ndarray,
    faces: np.ndarray,
    analysis,
    output_path: Path,
) -> list[Path]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base = output_path.with_suffix("")
    exported: list[Path] = []

    surface_mesh = _build_surface_fit_debug_mesh(vertices, faces, analysis)
    if surface_mesh is not None:
        surface_path = base.with_name(f"{base.name}_fit_surfaces.ply")
        surface_mesh.export(str(surface_path))
        exported.append(surface_path)

    support_mesh = _build_boundary_edge_support_debug_mesh(vertices, faces, analysis)
    if support_mesh is not None:
        support_path = base.with_name(f"{base.name}_boundary_edge_support_faces.ply")
        support_mesh.export(str(support_path))
        exported.append(support_path)

    curve_mesh = _build_intersection_curve_debug_mesh(vertices, analysis)
    curve_path = base.with_name(f"{base.name}_intersection_curves.ply")
    if curve_path.exists():
        curve_path.unlink()
    prepared_mesh = _build_prepared_subhole_debug_mesh(vertices, analysis)
    if prepared_mesh is not None:
        prepared_path = base.with_name(f"{base.name}_prepared_subholes.ply")
        prepared_mesh.export(str(prepared_path))
        exported.append(prepared_path)

    scene = trimesh.Scene()
    scene.add_geometry(_make_original_debug_mesh(vertices, faces), geom_name="original_mesh")
    if curve_mesh is not None:
        scene.add_geometry(curve_mesh, geom_name="intersection_curves_tubes")
    if prepared_mesh is not None:
        scene.add_geometry(prepared_mesh, geom_name="prepared_subhole_boundaries")
    scene_path = base.with_name(f"{base.name}_analysis_scene.glb")
    scene.export(str(scene_path))
    exported.append(scene_path)

    legend_path = base.with_name(f"{base.name}_debug_legend.txt")
    legend_path.write_text(_analysis_debug_legend_text(), encoding="utf-8")
    exported.append(legend_path)

    return exported


def _triangle_unit_normal(
    vertices: np.ndarray,
    faces: np.ndarray,
    fi: int,
) -> np.ndarray:
    tri = faces[int(fi)]
    p0 = vertices[int(tri[0])]
    p1 = vertices[int(tri[1])]
    p2 = vertices[int(tri[2])]
    n = np.cross(p1 - p0, p2 - p0)
    ln = float(np.linalg.norm(n))
    if ln < 1e-15:
        return np.zeros(3, dtype=np.float64)
    return n / ln


def _build_undirected_edge_to_faces(
    faces: np.ndarray,
) -> dict[Tuple[int, int], list[int]]:
    d: dict[Tuple[int, int], list[int]] = defaultdict(list)
    nf = int(faces.shape[0])
    for fi in range(nf):
        a, b, c = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
        for u, v in ((a, b), (b, c), (c, a)):
            k = (u, v) if u < v else (v, u)
            d[k].append(int(fi))
    return d


def _patch_face_connected_components(
    patch_set: Set[int],
    faces: np.ndarray,
    edge_to_faces: dict[Tuple[int, int], list[int]],
) -> list[Set[int]]:
    neigh: dict[int, list[int]] = {int(fi): [] for fi in patch_set}
    for fi in patch_set:
        tri = faces[int(fi)]
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, v in ((a, b), (b, c), (c, a)):
            k = (u, v) if u < v else (v, u)
            for fo in edge_to_faces.get(k, []):
                o = int(fo)
                if o != int(fi) and o in patch_set:
                    neigh[int(fi)].append(o)
    visited: set[int] = set()
    comps: list[Set[int]] = []
    for seed in patch_set:
        if seed in visited:
            continue
        comp: Set[int] = set()
        stack = [int(seed)]
        while stack:
            u = int(stack.pop())
            if u in visited:
                continue
            visited.add(u)
            comp.add(u)
            for w in neigh[u]:
                if w not in visited:
                    stack.append(int(w))
        comps.append(comp)
    return comps


def _fix_patch_face_winding(
    vertices: np.ndarray,
    faces: np.ndarray,
    patch_face_indices: np.ndarray,
    boundary_vertices: set[int],
) -> np.ndarray:
    """
    使补丁法向与邻接原始网格一致。

    ``sharp_edge_crossing`` 等多子孔合并时，补丁三角在「仅补丁–补丁共边」下可能分成多个连通块；
    若仍用全局平均法向做一次翻转，会出现某一整块与旧面仍反向。此处按连通分量分别：
    用与该分量**共边的旧面**法向估参考方向，再决定只翻转该分量内的三角形。
    """
    idx = np.asarray(patch_face_indices, dtype=np.int64).ravel()
    if idx.size == 0:
        return faces
    patch_set: Set[int] = {int(x) for x in idx.tolist()}
    out = np.asarray(faces, dtype=np.int64, copy=True)
    v = np.asarray(vertices, dtype=np.float64)
    edge_to_faces = _build_undirected_edge_to_faces(out)
    for comp in _patch_face_connected_components(patch_set, out, edge_to_faces):
        mesh_touch: Set[int] = set()
        for fi in comp:
            tri = out[int(fi)]
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            for u, w in ((a, b), (b, c), (c, a)):
                k = (u, w) if u < w else (w, u)
                for fo in edge_to_faces.get(k, []):
                    o = int(fo)
                    if o not in patch_set:
                        mesh_touch.add(o)

        mesh_normals: list[np.ndarray] = []
        if mesh_touch:
            for mfi in mesh_touch:
                mesh_normals.append(_triangle_unit_normal(v, out, mfi))
        else:
            for mfi in range(int(out.shape[0])):
                if mfi in patch_set:
                    continue
                tri = out[mfi]
                if any(int(x) in boundary_vertices for x in tri):
                    mesh_normals.append(_triangle_unit_normal(v, out, mfi))

        if not mesh_normals:
            continue
        ref = np.mean(np.stack(mesh_normals, axis=0), axis=0)
        ref = ref / (float(np.linalg.norm(ref)) + 1e-12)

        acc = np.zeros(3, dtype=np.float64)
        for fi in comp:
            acc += _triangle_unit_normal(v, out, int(fi))
        pmean = acc / (float(np.linalg.norm(acc)) + 1e-12)

        if float(np.dot(ref, pmean)) < 0:
            for fi in comp:
                out[int(fi)] = out[int(fi)][[0, 2, 1]]

    return out


def _propagate_patch_face_winding(
    vertices: np.ndarray,
    faces: np.ndarray,
    patch_face_indices: np.ndarray,
    boundary_vertices: set[int],
) -> np.ndarray:
    if patch_face_indices.size == 0:
        return faces
    out = faces.copy()
    patch_set = {int(x) for x in patch_face_indices.tolist()}
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for fi in patch_set:
        tri = out[int(fi)]
        for i in range(3):
            a = int(tri[i])
            b = int(tri[(i + 1) % 3])
            key = (a, b) if a < b else (b, a)
            edge_to_faces.setdefault(key, []).append(int(fi))

    neighbors: dict[int, list[tuple[int, tuple[int, int]]]] = {fi: [] for fi in patch_set}
    for edge, owners in edge_to_faces.items():
        if len(owners) != 2:
            continue
        a, b = owners
        neighbors[a].append((b, edge))
        neighbors[b].append((a, edge))

    visited: set[int] = set()
    for seed in list(patch_set):
        if seed in visited:
            continue
        queue = [seed]
        visited.add(seed)
        while queue:
            cur = queue.pop(0)
            tri_cur = out[int(cur)]
            for nb, edge in neighbors.get(cur, []):
                if nb in visited:
                    continue
                tri_nb = out[int(nb)]

                def edge_dir(tri: np.ndarray, e: tuple[int, int]) -> int:
                    for i in range(3):
                        u = int(tri[i])
                        v = int(tri[(i + 1) % 3])
                        if {u, v} == {int(e[0]), int(e[1])}:
                            return 1 if (u, v) == e else -1
                    return 0

                dir_cur = edge_dir(tri_cur, edge)
                dir_nb = edge_dir(tri_nb, edge)
                if dir_cur == dir_nb and dir_cur != 0:
                    out[int(nb)] = out[int(nb)][[0, 2, 1]]
                visited.add(int(nb))
                queue.append(int(nb))

    return _fix_patch_face_winding(vertices, out, patch_face_indices, boundary_vertices)


def _inject_hole_clean_metadata(
    repaired: trimesh.Trimesh,
    meta: Optional[Dict[str, Any]],
) -> trimesh.Trimesh:
    if meta is not None:
        repaired.metadata["hole_boundary_clean"] = meta
    return repaired


def _vertex_key(point: np.ndarray, tol: float) -> tuple[int, int, int]:
    p = np.asarray(point, dtype=np.float64)
    scale = max(float(tol), 1e-12)
    return tuple(int(np.round(float(x) / scale)) for x in p)


def _undirected_edge_key(u: int, v: int) -> tuple[int, int]:
    u, v = int(u), int(v)
    return (u, v) if u < v else (v, u)


def _face_undirected_edge_counts(faces: np.ndarray) -> dict[tuple[int, int], int]:
    ec: dict[tuple[int, int], int] = defaultdict(int)
    nf = int(faces.shape[0])
    for fi in range(nf):
        a, b, c = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
        for u, v in ((a, b), (b, c), (c, a)):
            ec[_undirected_edge_key(u, v)] += 1
    return ec


def _boundary_vertex_set(faces: np.ndarray) -> set[int]:
    ec = _face_undirected_edge_counts(faces)
    bdry_edges = [e for e, c in ec.items() if c == 1]
    verts: set[int] = set()
    for u, v in bdry_edges:
        verts.add(u)
        verts.add(v)
    return verts


class _SeamVertexUnionFind:
    """按「小下标为根」合并，便于稳定压缩顶点。"""

    __slots__ = ("p",)

    def __init__(self, n: int) -> None:
        self.p = np.arange(int(n), dtype=np.int64)

    def find(self, x: int) -> int:
        x = int(x)
        px = int(self.p[x])
        if px != x:
            self.p[x] = self.find(px)
        return int(self.p[x])

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if ra < rb:
            self.p[rb] = ra
        else:
            self.p[ra] = rb


def _weld_coincident_boundary_vertices(
    vertices: np.ndarray,
    faces: np.ndarray,
    patch_face_mask: np.ndarray,
    weld_tol: float,
    *,
    hole_boundary_vertices: Optional[set[int]] = None,
    max_passes: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[set[int]]]:
    """
    在开边界上合并距离 < weld_tol 的顶点，闭合多子孔拼接后残留的双重边界（缝）。

    仅对当前仍落在开边界上的顶点建邻，不碰纯内部顶点，避免破坏原网格其他区域。

    若提供 ``hole_boundary_vertices``（孔洞环顶点），在顶点位移与压缩后仍会返回对应的
    新索引集合，供 ``_propagate_patch_face_winding`` 使用。
    """
    v = np.asarray(vertices, dtype=np.float64, copy=True)
    f = np.asarray(faces, dtype=np.int64, copy=True)
    mask = np.asarray(patch_face_mask, dtype=bool).copy()
    track: Optional[set[int]] = None
    if hole_boundary_vertices is not None:
        track = {int(x) for x in hole_boundary_vertices}
    if (
        v.size == 0
        or f.size == 0
        or not np.isfinite(weld_tol)
        or float(weld_tol) <= 1e-15
    ):
        return v, f, mask, track

    inv_w = 1.0 / float(weld_tol)
    for _ in range(int(max_passes)):
        bset = _boundary_vertex_set(f)
        if len(bset) < 2:
            break
        n_v = int(v.shape[0])
        uf = _SeamVertexUnionFind(n_v)
        buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for vi in bset:
            if vi < 0 or vi >= n_v:
                continue
            p = v[vi]
            cell = (
                int(np.floor(p[0] * inv_w)),
                int(np.floor(p[1] * inv_w)),
                int(np.floor(p[2] * inv_w)),
            )
            buckets[cell].append(int(vi))
        merged_any = False
        for cell, ids in buckets.items():
            cx, cy, cz = cell
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        ncell = (cx + dx, cy + dy, cz + dz)
                        oth = buckets.get(ncell)
                        if oth is None:
                            continue
                        for i in ids:
                            for j in oth:
                                if i >= j:
                                    continue
                                if float(np.linalg.norm(v[i] - v[j])) <= float(weld_tol):
                                    uf.union(i, j)
                                    merged_any = True

        if not merged_any:
            break

        remap = np.array([uf.find(i) for i in range(n_v)], dtype=np.int64)
        if track is not None:
            track = {
                int(remap[x])
                for x in track
                if 0 <= int(x) < n_v
            }
        f = remap[f]
        keep_faces = np.array(
            [
                fi
                for fi in range(int(f.shape[0]))
                if len(
                    {int(f[fi, 0]), int(f[fi, 1]), int(f[fi, 2])}
                )
                == 3
            ],
            dtype=np.int64,
        )
        f = f[keep_faces]
        mask = mask[keep_faces]

        used = np.zeros(n_v, dtype=bool)
        used[f.ravel()] = True
        kept = np.flatnonzero(used)
        old_to_new = np.full(n_v, -1, dtype=np.int64)
        old_to_new[kept] = np.arange(len(kept), dtype=np.int64)
        v = v[kept]
        f = old_to_new[f]
        if track is not None:
            track = {
                int(old_to_new[x])
                for x in track
                if 0 <= int(x) < len(old_to_new) and int(old_to_new[x]) >= 0
            }

    return v, f, mask, track



def _format_residual_boundary_diagnostics(
    residual_loops: Sequence[Sequence[int]],
    boundary_loop: Sequence[int],
) -> str:
    """L4 覆盖缺口诊断：列出残余环顶点及是否在原孔环上。"""
    hole_set = {int(v) for v in boundary_loop}
    parts: List[str] = []
    for loop in residual_loops:
        vids = [int(v) for v in loop]
        on_hole = [vid for vid in vids if vid in hole_set]
        off_hole = [vid for vid in vids if vid not in hole_set]
        parts.append(
            f"loop={vids} on_hole={on_hole} new_indices={off_hole}"
        )
    return "; ".join(parts)


def _merge_partitioned_subholes(
    mesh: trimesh.Trimesh,
    analysis,
    boundary_loop: list[int],
    *,
    fix_orientation: bool,
    color_patch: bool,
    patch_color: np.ndarray,
) -> trimesh.Trimesh:
    """
    L4 分区补洞合并：三角化各子孔 → 并入网格 → 焊接 → 检测覆盖缺口。

    Seam 契约：孔环 ``boundary_sources`` 顶点必须复用原 mesh 索引，
    禁止曲面投影漂移在汇交角产生重复顶点。
    """
    vertices = mesh.vertices.copy()
    faces = mesh.faces.copy()
    diag = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    tol = max(1e-9, 1e-8 * max(diag, 1.0))
    boundary_set = set(boundary_loop)
    coord_map = {
        _vertex_key(vertices[i], tol): i
        for i in range(len(vertices))
    }
    feature_source_map: dict[int, int] = {}

    def get_or_add_vertex(
        point: np.ndarray,
        preferred: int | None = None,
        *,
        snap_hole_boundary: bool = False,
    ) -> int:
        nonlocal vertices
        if preferred is not None and int(preferred) < 0 and int(preferred) in feature_source_map:
            return int(feature_source_map[int(preferred)])
        if preferred is not None and preferred >= 0:
            pref = int(preferred)
            # L4 seam：孔环顶点为唯一索引承载，避免投影漂移复制汇交角
            if pref in boundary_set and pref < len(vertices):
                return pref
            if pref < len(vertices) and float(np.linalg.norm(vertices[pref] - point)) <= tol * 4.0:
                return pref
        if snap_hole_boundary and boundary_set:
            best_hv: int | None = None
            best_d = float("inf")
            for hv in boundary_set:
                hi = int(hv)
                if hi < 0 or hi >= len(vertices):
                    continue
                d = float(np.linalg.norm(vertices[hi] - point))
                if d < best_d and d <= tol * 4.0:
                    best_d = d
                    best_hv = hi
            if best_hv is not None:
                return int(best_hv)
        key = _vertex_key(point, tol)
        if key in coord_map:
            idx = int(coord_map[key])
            if float(np.linalg.norm(vertices[idx] - point)) <= tol * 4.0:
                if preferred is not None and preferred < 0:
                    feature_source_map[int(preferred)] = idx
                return idx
        idx = int(vertices.shape[0])
        vertices = np.vstack([vertices, np.asarray(point, dtype=np.float64).reshape(1, 3)])
        coord_map[key] = idx
        if preferred is not None and preferred < 0:
            feature_source_map[int(preferred)] = idx
        return idx

    new_face_blocks: list[np.ndarray] = []
    for subhole in analysis.prepared_subholes:
        boundary_points = np.asarray(subhole.closed_boundary_points, dtype=np.float64)
        if boundary_points.shape[0] < 3:
            continue
        fit = analysis.patch_surface_fits.get(int(subhole.patch_label))
        boundary_count = int(boundary_points.shape[0])
        sources = list(subhole.boundary_sources)
        boundary_uv = None
        try:
            if fit is not None:
                param = parameterize_boundary(
                    fit,
                    boundary_points,
                    reference_normal=subhole.reference_normal,
                )
                boundary_uv = np.asarray(param.uv_boundary_points, dtype=np.float64)
                if boundary_uv.shape[0] != boundary_points.shape[0]:
                    raise ValueError("parameterize_boundary 与边界点数不一致")
                readiness = assess_patch_boundary_readiness(
                    boundary_points,
                    boundary_uv,
                )
                if not bool(readiness.get("ready")):
                    raise ValueError(
                        "子孔边界不可三角化: "
                        f"patch_label={int(subhole.patch_label)} "
                        f"readiness={readiness}"
                    )
                boundary_count = int(readiness.get("n_boundary_sanitized", boundary_points.shape[0]))
                param_kind = str(param.kind)
                density_scale = (
                    0.92
                    if param_kind
                    in {
                        "cylinder",
                        "cylinder_tangent",
                        "cylinder_local_plane",
                        "sphere_tangent",
                        "cone",
                    }
                    else 1.0
                )
                seam_edges = seam_constrained_edges_for_subhole(subhole)
                v_patch, f_patch = triangulate_hole_patch(
                    boundary_points,
                    subhole.closed_boundary_edges,
                    boundary_points_2d=boundary_uv,
                    lift_point_from_2d=lambda uv, fit=fit, param=param: lift_parameter_point(
                        fit,
                        param,
                        uv,
                    ),
                    boundary_sources=sources,
                    closure_kind=subhole.closure_kind,
                    parameterization_kind=param_kind,
                    open_boundary_count=int(subhole.boundary_points.shape[0]),
                    feature_point_vertex_indices=subhole.feature_point_vertex_indices,
                    reference_normal=subhole.reference_normal,
                    height_scale=0.8660254037844386 * density_scale,
                    seam_constrained_edges=seam_edges,
                )
            else:
                v_patch, f_patch = triangulate_ordered_hole_boundary(
                    boundary_points,
                    boundary_sources=subhole.boundary_sources,
                    closure_kind=subhole.closure_kind,
                    parameterization_kind=subhole.parameterization_kind,
                    open_boundary_count=int(subhole.boundary_points.shape[0]),
                    feature_point_vertex_indices=subhole.feature_point_vertex_indices,
                    reference_normal=subhole.reference_normal,
                )
                boundary_count = int(boundary_points.shape[0])
        except (RuntimeError, ValueError) as exc:
            readiness = assess_patch_boundary_readiness(boundary_points, boundary_uv)
            raise ValueError(
                "分区子孔三角化失败: "
                f"patch_label={int(subhole.patch_label)} "
                f"reason={exc} readiness={readiness}"
            ) from exc
        if fit is not None and is_analytic_surface_type(fit.surface_type):
            for i in range(boundary_count, int(v_patch.shape[0])):
                v_patch[i] = project_point_to_surface(
                    fit, np.asarray(v_patch[i], dtype=np.float64)
                )
        local_to_global = np.empty(v_patch.shape[0], dtype=np.int64)
        ring_boundary = np.asarray(boundary_points, dtype=np.float64)

        for i in range(v_patch.shape[0]):
            preferred = sources[i] if i < boundary_count and i < len(sources) else None
            if i < boundary_count:
                point = np.asarray(ring_boundary[i], dtype=np.float64)
                if (
                    preferred is not None
                    and int(preferred) >= 0
                    and int(preferred) in boundary_set
                ):
                    point = np.asarray(vertices[int(preferred)], dtype=np.float64)
            else:
                point = np.asarray(v_patch[i], dtype=np.float64)
            local_to_global[i] = get_or_add_vertex(
                point,
                preferred,
                snap_hole_boundary=(i < boundary_count),
            )
        new_face_blocks.append(local_to_global[f_patch])

    if not new_face_blocks:
        raise ValueError("多面域孔洞未生成任何可三角化的闭合子孔")

    n_faces_old = int(faces.shape[0])
    faces = np.vstack([faces] + new_face_blocks)
    patch_face_mask = np.zeros(int(faces.shape[0]), dtype=bool)
    patch_face_mask[n_faces_old:] = True
    weld_tol = max(1e-9, 1e-6 * max(diag, 1.0))
    hole_bdry_track = {int(x) for x in boundary_loop}
    vertices, faces, patch_face_mask, hole_bdry_track = _weld_coincident_boundary_vertices(
        vertices,
        faces,
        patch_face_mask,
        weld_tol,
        hole_boundary_vertices=hole_bdry_track,
        max_passes=4,
    )
    patch_idx = np.nonzero(patch_face_mask)[0].astype(np.int64)
    if fix_orientation:
        faces = _propagate_patch_face_winding(
            vertices, faces, patch_idx, hole_bdry_track
        )

    det_res = HoleDetector()
    _he_res, loops_res = det_res.detect_with_half_edge(
        np.asarray(vertices, dtype=np.float64),
        np.asarray(faces, dtype=np.int64),
    )
    residual_loops = [
        [int(x) for x in rem_loop_raw]
        for rem_loop_raw in loops_res
        if 3 <= len(rem_loop_raw) <= len(boundary_loop)
    ]
    if residual_loops:
        detail = _format_residual_boundary_diagnostics(residual_loops, boundary_loop)
        raise ValueError(
            "O3_coverage_gap: 子孔焊接后仍有未覆盖边界环（禁止 residual 静默补带）: "
            f"n_residual={len(residual_loops)} "
            f"lens={[len(x) for x in residual_loops]} "
            f"detail={detail}"
        )

    repaired = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if color_patch:
        repaired = _apply_patch_face_color(repaired, patch_idx, patch_color)
    return repaired


def _analysis_ready_for_fill(analysis: Any) -> bool:
    """L4 是否可走 prepared_subholes 统一补洞入口（读 fill_gate，不读 hole_type）。"""
    fill_gate = getattr(analysis, "fill_gate", None)
    return fill_gate is not None and bool(fill_gate.accepted)


def _expected_partition_labels(analysis: Any) -> Set[int]:
    fill_gate = getattr(analysis, "fill_gate", None)
    if fill_gate is not None and fill_gate.expected_labels:
        return {int(x) for x in fill_gate.expected_labels}
    fill_classification = getattr(analysis, "fill_classification", None)
    if fill_classification is not None:
        active = getattr(fill_classification, "active_fill_labels", None)
        if active:
            return {int(x) for x in active}
    diagnostics = getattr(analysis, "diagnostics", None)
    recovery_diag = (
        getattr(diagnostics, "recovery_diagnostics", None) if diagnostics is not None else None
    )
    if isinstance(recovery_diag, dict):
        active = recovery_diag.get("active_fill_labels")
        if active:
            return {int(x) for x in active}
    arcs = getattr(analysis, "boundary_arcs", None) or []
    return {int(arc.patch_label) for arc in arcs}


def _partition_fill_rejection_message(analysis: Any) -> Optional[str]:
    obstacles = list(getattr(analysis, "partition_obstacles", None) or [])
    explicit = [
        o
        for o in obstacles
        if str(getattr(o, "kind", ""))
        in {PARTITION_OBSTACLE_O3, PARTITION_OBSTACLE_O4}
    ]
    if explicit:
        return "; ".join(
            f"{o.kind} label={o.label}: {o.detail}" for o in explicit
        )
    fill_gate = getattr(analysis, "fill_gate", None)
    if fill_gate is not None and fill_gate.reject_reason:
        return str(fill_gate.reject_reason)
    return None


def _validate_partitioned_subholes(analysis: Any) -> None:
    expected = _expected_partition_labels(analysis)
    if not expected:
        raise ValueError("多面域孔洞缺少活跃补洞 label，无法分区补洞")
    prepared = list(getattr(analysis, "prepared_subholes", None) or [])
    try:
        validate_before_partitioned_fill(prepared, expected)
    except FillValidationError as exc:
        explicit = _partition_fill_rejection_message(analysis)
        raise ValueError(explicit or str(exc)) from exc


def run_cad_fill(
    input_path: str | Path,
    output_path: str | Path,
    *,
    mesh: Optional[trimesh.Trimesh] = None,
    loop_index: int = 0,
    fix_orientation: bool = True,
    color_patch: bool = False,
    patch_color: tuple[int, int, int, int] = (255, 215, 0, 255),
    debug_analysis: bool = False,
    hole_clean: bool = False,
) -> trimesh.Trimesh:
    """
    加载网格，填充第 ``loop_index`` 个孔洞，写出 ``output_path``。

    补洞路由（与 ``hole_analyzer`` FillPlan 契约一致）::

    - L3 产出 ``prepared_subholes`` 且 ``fill_gate.accepted`` → L4 统一 merge
    - 分析或 fill_gate 失败 → 显式 ``ValueError``

    ``hole_clean=True`` 时先在全网格上迭代删除「齿状三角形」（至少两条边在开边界上），
    再检测孔洞并补洞；孔洞顺序可能与清理前不同（多孔时 ``--loop`` 宜复查）。

    若传入 ``mesh``，则不再从 ``input_path`` 读取（仍用于日志/调试），便于调用方先清理再补洞。

    返回修补后的 ``Trimesh``（与写出文件一致）；若执行了清理，
    ``mesh.metadata['hole_boundary_clean']`` 含迭代次数与删除面数。
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    if mesh is None:
        mesh = trimesh.load(str(input_path), force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    else:
        mesh = mesh.copy()

    boundary_clean_meta: Optional[Dict[str, Any]] = None
    if hole_clean:
        clean_res = clean_hole_boundary_tooth_faces(
            np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.faces, dtype=np.int64),
        )
        mesh = trimesh.Trimesh(
            vertices=clean_res.vertices,
            faces=clean_res.faces,
            process=False,
        )
        boundary_clean_meta = {
            "iterations": int(clean_res.iterations),
            "faces_removed_total": int(clean_res.faces_removed_total),
        }

    detector = HoleDetector()
    analyzer = HoleAnalyzer()
    he_mesh, loops = detector.detect_with_half_edge(
        np.array(mesh.vertices, dtype=np.float64),
        np.array(mesh.faces, dtype=np.int64),
    )
    if not loops:
        raise ValueError("未检测到孔洞")
    if loop_index < 0 or loop_index >= len(loops):
        raise IndexError(f"孔洞索引 {loop_index} 超出范围 0..{len(loops) - 1}")

    boundary_loop = [int(v) for v in loops[loop_index]]
    n_b = len(boundary_loop)
    if n_b < 3:
        raise ValueError("孔洞边界顶点数不足 3")

    analysis = analyzer.analyze(he_mesh, boundary_loop)
    analysis_confidence = float(getattr(analysis, "analysis_confidence", 1.0))
    _print_analysis_topology_summary(analysis)
    if debug_analysis:
        for path in _export_analysis_debug_visuals(
            np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.faces, dtype=np.int64),
            analysis,
            output_path,
        ):
            print(f"已保存调试可视化: {path}")
        print(
            "调试图例: 橙/黄管=保留交线；灰管=已修剪交线；青球=L1 特征点；"
            "灰球=已降级；橙球=后求交点；紫/绿/蓝粗管=L3 实际补洞边界。"
        )
    _raise_if_invalid_arrangement(analysis)

    if not _analysis_ready_for_fill(analysis):
        fill_gate = getattr(analysis, "fill_gate", None)
        reject = (
            _partition_fill_rejection_message(analysis)
            or (str(fill_gate.reject_reason) if fill_gate else "")
            or "fill_gate_not_accepted"
        )
        raise ValueError(f"补洞分析未通过: {reject}")

    if analysis_confidence < 0.25 and not debug_analysis:
        raise RuntimeError(
            f"分析置信度过低 ({analysis_confidence:.2f})，已跳过自动补洞；"
            "请使用 --debug-analysis 查看分析结果。"
        )

    _validate_partitioned_subholes(analysis)
    repaired = _merge_partitioned_subholes(
        mesh,
        analysis,
        boundary_loop,
        fix_orientation=fix_orientation,
        color_patch=color_patch,
        patch_color=np.asarray(patch_color, dtype=np.uint8),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    repaired.export(str(output_path))
    return _inject_hole_clean_metadata(repaired, boundary_clean_meta)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="CAD 孔洞填充（孔洞分析 + 单面孔前沿推进初始三角化）",
    )
    parser.add_argument("input", help="输入 OBJ/STL/PLY 等")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="输出路径（默认：输入名_cad_fill.obj）",
    )
    parser.add_argument(
        "--loop",
        type=int,
        default=0,
        help="填充第几个孔洞（默认 0）",
    )
    parser.add_argument(
        "--no-orient-fix",
        action="store_true",
        help="禁用按边界法向校正补丁三角形朝向",
    )
    parser.add_argument(
        "--color-patch",
        action="store_true",
        help="导出时给补洞新增面片着色",
    )
    parser.add_argument(
        "--patch-color",
        default="255,215,0,255",
        help="补洞区域颜色，格式 R,G,B 或 R,G,B,A，默认 255,215,0,255",
    )
    parser.add_argument(
        "--debug-analysis",
        action="store_true",
        help="额外导出拟合曲面与交线的调试可视化 ply",
    )
    parser.add_argument(
        "--hole-clean",
        action="store_true",
        help="补洞前迭代删除齿状边界三角形（≥2 条开边界边），平滑孔洞边界",
    )
    args = parser.parse_args(argv)

    inp = Path(args.input)
    out = args.output
    if out is None:
        stem = inp.stem
        default_suffix = ".ply" if args.color_patch else ".obj"
        out = _ROOT / "out" / f"{stem}_cad_fill{default_suffix}"
    else:
        out = Path(out)

    try:
        patch_color = _parse_rgba(args.patch_color)
        repaired = run_cad_fill(
            inp,
            out,
            loop_index=args.loop,
            fix_orientation=not args.no_orient_fix,
            color_patch=args.color_patch,
            patch_color=tuple(int(x) for x in patch_color.tolist()),
            debug_analysis=args.debug_analysis,
            hole_clean=args.hole_clean,
        )
    except (ValueError, IndexError) as e:
        print(e, file=sys.stderr)
        return 1

    print(
        f"已保存: {out} "
        f"({repaired.vertices.shape[0]} 顶点, {repaired.faces.shape[0]} 面)"
    )
    hc = repaired.metadata.get("hole_boundary_clean")
    if hc is not None:
        print(
            f"孔洞边界清理: 迭代 {hc['iterations']} 次, "
            f"累计删除齿状面 {hc['faces_removed_total']} 个"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
