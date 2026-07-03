#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""孔洞分析调试 / 论文 / 批处理统计（与补洞核心解耦）。"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .feature_graph import (
    FeatureArrangement,
    FeatureGraph,
    FeaturePoint,
    PatchCell,
    aggregate_clip_confidence,
    aggregate_curve_confidence,
    polygon_area_3d,
    validate_patch_cell,
)
from .hole_analysis_types import (
    CLOSED_PAIR_CURVE_MULTI_PATCH,
    MULTI_PATCH_GENERIC,
    S2C_MULTI_CORNER_PAIR,
    SHARP_EDGE_CROSSING,
    SINGLE_PATCH,
    TRIPLE_SURFACE_JUNCTION,
    AnalysisDiagnostics,
    BoundaryArc,
    HoleType,
    PreparedSubhole,
)
from .surface_fitting import SurfaceFit
from .surface_intersections import (
    AnalyticCurve,
    BoundedCurveSegment,
    IntersectionCurve,
)


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


def _is_s2c_multi_corner_hole(
    vertices: np.ndarray,
    loop: Sequence[int],
    boundary_edge_labels: Sequence[int],
    feature_point_positions: Sequence[int],
    arcs: Sequence[BoundaryArc],
) -> bool:
    _ = vertices, loop
    if len(feature_point_positions) != 4 or len(arcs) != 4:
        return False
    unique_patch_count = len(set(int(x) for x in boundary_edge_labels))
    return unique_patch_count == 2


def _matches_closed_pair_from_curves(
    arcs: Sequence[BoundaryArc],
    curves: Sequence[IntersectionCurve],
    patch_surface_fits: Optional[Mapping[int, SurfaceFit]] = None,
) -> bool:
    _ = patch_surface_fits
    if len(arcs) < 3 or len(curves) < 2:
        return False

    unique_labels = sorted({int(arc.patch_label) for arc in arcs})
    if len(unique_labels) < 3:
        return False

    analytic_pair_curves = [
        curve
        for curve in curves
        if int(curve.endpoint_vertex_indices[0]) >= 0
        and int(curve.endpoint_vertex_indices[1]) >= 0
    ]
    if len(analytic_pair_curves) < 2:
        return False

    covered_pairs = {
        tuple(sorted((int(curve.patch_pair[0]), int(curve.patch_pair[1]))))
        for curve in analytic_pair_curves
    }
    if len(covered_pairs) < 2:
        return False

    for label in unique_labels:
        incident_count = 0
        for arc in arcs:
            if int(arc.patch_label) != label:
                continue
            start = int(arc.vertex_indices[0])
            end = int(arc.vertex_indices[-1])
            if any(start in curve.endpoint_vertex_indices for curve in analytic_pair_curves):
                incident_count += 1
            if any(end in curve.endpoint_vertex_indices for curve in analytic_pair_curves):
                incident_count += 1
        if incident_count < 2:
            return False

    return True


def infer_template_hint(
    loop: Sequence[int],
    boundary_edge_labels: Sequence[int],
    feature_point_positions: Sequence[int],
    arcs: Sequence[BoundaryArc],
    vertices: np.ndarray,
    curves: Sequence[IntersectionCurve],
    patch_surface_fits: Mapping[int, SurfaceFit],
) -> HoleType:
    """由边界拓扑与恢复结果推断详细模板标签（仅 diagnostics，不参与决策）。"""
    unique_patch_count = len(set(int(x) for x in boundary_edge_labels))
    if unique_patch_count <= 1:
        return SINGLE_PATCH
    if unique_patch_count > 3:
        return MULTI_PATCH_GENERIC
    if _matches_closed_pair_from_curves(arcs, curves, patch_surface_fits):
        return CLOSED_PAIR_CURVE_MULTI_PATCH
    if _is_s2c_multi_corner_hole(
        vertices, loop, boundary_edge_labels, feature_point_positions, arcs
    ):
        return S2C_MULTI_CORNER_PAIR
    if unique_patch_count == 2:
        return SHARP_EDGE_CROSSING
    if unique_patch_count >= 3 or len(feature_point_positions) >= 3:
        return TRIPLE_SURFACE_JUNCTION
    return SHARP_EDGE_CROSSING


def build_feature_graph(
    loop: Sequence[int],
    feature_point_positions: Sequence[int],
    feature_point_vertex_ids: Sequence[int],
    boundary_vertex_labels: Mapping[int, Sequence[int]],
    arcs: Sequence[BoundaryArc],
    curves: Sequence[IntersectionCurve],
    bounded_segments: Sequence[BoundedCurveSegment],
    analytic_curves: Sequence[AnalyticCurve],
    junction_point: Optional[np.ndarray],
    junction_confidence: str,
    template_hint: str,
    prepared: Sequence[PreparedSubhole],
    arrangement: Optional[FeatureArrangement] = None,
) -> FeatureGraph:
    fp_set = {int(v) for v in feature_point_vertex_ids}
    feature_points: List[FeaturePoint] = []
    for i, vi in enumerate(loop):
        v = int(vi)
        if v not in fp_set:
            continue
        labels = tuple(sorted(int(x) for x in boundary_vertex_labels.get(v, [])))
        feature_points.append(
            FeaturePoint(vertex_id=v, surface_labels=labels, loop_index=i)
        )

    clip_confidences = [str(s.clip_confidence) for s in bounded_segments]
    if bounded_segments:
        confidence = aggregate_clip_confidence(bounded_segments)
    elif curves:
        confidence = aggregate_curve_confidence(curves)
    elif len(prepared) == 1 and not curves:
        confidence = 0.9
    else:
        confidence = 0.0

    cells = _cells_from_prepared_subholes(prepared)

    return FeatureGraph(
        loop=[int(v) for v in loop],
        feature_points=feature_points,
        arc_count=len(arcs),
        analytic_segments=list(bounded_segments),
        analytic_curves=list(analytic_curves),
        intersection_curves=list(curves),
        junction_point=(
            None
            if junction_point is None
            else np.asarray(junction_point, dtype=np.float64)
        ),
        junction_confidence=str(junction_confidence),
        cells=cells,
        confidence=float(confidence),
        template_hint=str(template_hint),
        clip_confidences=clip_confidences,
        arrangement=arrangement,
    )


def build_analysis_diagnostics(
    *,
    loop: Sequence[int],
    boundary_edge_labels: Sequence[int],
    active_feature_point_positions: Sequence[int],
    feature_point_vertex_ids: Sequence[int],
    feature_edges: Sequence[Tuple[int, int]],
    boundary_vertex_labels: Mapping[int, Sequence[int]],
    arcs: Sequence[BoundaryArc],
    vertices: np.ndarray,
    curves: Sequence[IntersectionCurve],
    bounded_segments: Sequence[BoundedCurveSegment],
    analytic_curves: Sequence[AnalyticCurve],
    junction_point: Optional[np.ndarray],
    junction_confidence: str,
    prepared: Sequence[PreparedSubhole],
    patch_surface_fits: Mapping[int, SurfaceFit],
    recovery_diagnostics: Mapping[str, object],
    neighborhood_face_indices: Sequence[int],
    surface_patch_labels: Mapping[int, int],
    patch_face_indices: Mapping[int, List[int]],
    boundary_half_edges: Sequence[int],
    feature_arrangement: Optional[FeatureArrangement],
) -> AnalysisDiagnostics:
    template_hint = str(
        infer_template_hint(
            loop,
            boundary_edge_labels,
            active_feature_point_positions,
            arcs,
            vertices,
            curves,
            patch_surface_fits,
        )
    )
    feature_graph = build_feature_graph(
        loop,
        active_feature_point_positions,
        feature_point_vertex_ids,
        boundary_vertex_labels,
        arcs,
        curves,
        bounded_segments,
        analytic_curves,
        junction_point,
        junction_confidence,
        template_hint,
        prepared,
        feature_arrangement,
    )
    recovery_mode = str(recovery_diagnostics.get("mode", ""))
    return AnalysisDiagnostics(
        template_hint=template_hint,
        analysis_confidence=float(feature_graph.confidence),
        feature_graph=feature_graph,
        recovery_mode=recovery_mode,
        recovery_diagnostics=dict(recovery_diagnostics),
        neighborhood_face_indices=[int(x) for x in neighborhood_face_indices],
        surface_patch_labels={int(k): int(v) for k, v in surface_patch_labels.items()},
        patch_face_indices={
            int(k): [int(x) for x in vals] for k, vals in patch_face_indices.items()
        },
        feature_point_candidates=[int(x) for x in feature_point_vertex_ids],
        feature_edge_candidates=[(int(a), int(b)) for a, b in feature_edges],
        boundary_half_edges=[int(x) for x in boundary_half_edges],
        bounded_segments=list(bounded_segments),
        analytic_curves=list(analytic_curves),
        feature_arrangement=feature_arrangement,
    )
