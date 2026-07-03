#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""孔洞特征图：解析有界交线段 + patch cell 划分。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from .surface_intersections import AnalyticCurve, BoundedCurveSegment, IntersectionCurve


@dataclass
class FeaturePoint:
    """孔边界上的特征点（由相邻内侧三角面法向折角判定）。"""

    vertex_id: int
    surface_labels: Tuple[int, ...]
    loop_index: int = -1


@dataclass
class PatchCell:
    patch_label: int
    boundary_arc_indices: List[int]
    segment_indices: List[int]
    closed_loop_vertex_ids: List[int] = field(default_factory=list)
    area: float = 0.0
    is_active: bool = True
    inactive_reason: str = ""


@dataclass
class GraphNode:
    """Surface-labeled feature arrangement node."""

    id: int
    point3d: np.ndarray
    uv: np.ndarray
    kind: str
    source: int = -1
    incident_labels: Tuple[int, ...] = ()
    confidence: float = 1.0


@dataclass
class GraphEdge:
    """Directed geometric edge candidate used to extract patch cells."""

    id: int
    node0: int
    node1: int
    points3d: np.ndarray
    sources: List[int]
    kind: str
    surface_pair: Tuple[int, ...]
    left_label: Optional[int] = None
    right_label: Optional[int] = None
    source_curve_id: int = -1
    confidence: float = 1.0


@dataclass
class GraphCell:
    """Closed graph face with an assigned surface patch label."""

    id: int
    patch_label: int
    edge_cycle: List[Tuple[int, bool]]
    closed_points: np.ndarray
    sources: List[int]
    area: float
    uv_area: float
    validity_score: float
    is_valid: bool = True
    invalid_reason: str = ""


@dataclass
class FeatureArrangement:
    """Embedded arrangement used before exporting PreparedSubhole objects."""

    nodes: List[GraphNode] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)
    cells: List[GraphCell] = field(default_factory=list)
    euler_residual: int = 0
    diagnostics: Dict[str, object] = field(default_factory=dict)


@dataclass
class FeatureGraph:
    loop: List[int]
    feature_points: List[FeaturePoint]
    arc_count: int
    analytic_segments: List[BoundedCurveSegment]
    analytic_curves: List[AnalyticCurve]
    intersection_curves: List[IntersectionCurve]
    junction_point: Optional[np.ndarray]
    junction_confidence: str
    cells: List[PatchCell]
    confidence: float
    template_hint: str
    clip_confidences: List[str] = field(default_factory=list)
    arrangement: Optional[FeatureArrangement] = None


def polygon_area_3d(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 3:
        return 0.0
    center = np.mean(pts, axis=0)
    area_vec = np.sum(
        np.cross(pts - center, np.roll(pts, -1, axis=0) - center),
        axis=0,
    )
    return 0.5 * float(np.linalg.norm(area_vec))


def signed_polygon_area_2d(poly: np.ndarray) -> float:
    pts = np.asarray(poly, dtype=np.float64)
    if pts.shape[0] < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _orient_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _segments_cross_proper_2d(
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
    return (o1 * o2 < -eps) and (o3 * o4 < -eps)


def polygon_is_simple_2d(poly: np.ndarray, eps: float = 1e-10) -> bool:
    pts = np.asarray(poly, dtype=np.float64)
    n = int(pts.shape[0])
    if n < 3:
        return False
    for i in range(n):
        i1 = (i + 1) % n
        for j in range(i + 1, n):
            j1 = (j + 1) % n
            if i1 == j or j1 == i:
                continue
            if i == 0 and j1 == 0:
                continue
            if _segments_cross_proper_2d(pts[i], pts[i1], pts[j], pts[j1], eps):
                return False
    return True


def project_points_to_best_plane(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project points to a stable local 2D frame for graph embedding."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros(3), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    center = np.mean(pts, axis=0)
    rel = pts - center.reshape(1, 3)
    if pts.shape[0] < 3:
        return rel[:, :2], center, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    _, _, vh = np.linalg.svd(rel, full_matrices=False)
    u_axis = np.asarray(vh[0], dtype=np.float64)
    v_axis = np.asarray(vh[1], dtype=np.float64)
    return np.column_stack([rel @ u_axis, rel @ v_axis]), center, u_axis, v_axis


def _edge_points(edge: GraphEdge, forward: bool) -> np.ndarray:
    pts = np.asarray(edge.points3d, dtype=np.float64)
    return pts if forward else pts[::-1].copy()


def _edge_sources(edge: GraphEdge, forward: bool) -> List[int]:
    src = [int(x) for x in edge.sources]
    return src if forward else src[::-1]


def assemble_cell_boundary(
    arrangement: FeatureArrangement,
    edge_cycle: Sequence[Tuple[int, bool]],
) -> Tuple[np.ndarray, List[int]]:
    points: List[np.ndarray] = []
    sources: List[int] = []
    for edge_id, forward in edge_cycle:
        edge = arrangement.edges[int(edge_id)]
        pts = _edge_points(edge, bool(forward))
        src = _edge_sources(edge, bool(forward))
        if pts.shape[0] == 0:
            continue
        start = 1 if points and float(np.linalg.norm(points[-1] - pts[0])) < 1e-9 else 0
        for i in range(start, int(pts.shape[0])):
            points.append(np.asarray(pts[i], dtype=np.float64))
            sources.append(int(src[i]) if i < len(src) else -1)
    if len(points) >= 2 and float(np.linalg.norm(points[0] - points[-1])) < 1e-9:
        points.pop()
        sources.pop()
    return np.asarray(points, dtype=np.float64), sources


def _cell_label_from_edges(
    arrangement: FeatureArrangement,
    edge_cycle: Sequence[Tuple[int, bool]],
) -> Optional[int]:
    arc_labels: List[int] = []
    curve_pairs: List[Tuple[int, int]] = []
    for edge_id, forward in edge_cycle:
        edge = arrangement.edges[int(edge_id)]
        if edge.kind == "boundary_arc" and edge.left_label is not None:
            arc_labels.append(int(edge.left_label))
        elif edge.kind == "intersection_curve":
            left = edge.left_label
            right = edge.right_label
            if left is not None and right is not None:
                curve_pairs.append((int(left), int(right)))
    if curve_pairs:
        arc_set = set(arc_labels)
        overlap = {
            int(label)
            for left, right in curve_pairs
            for label in (left, right)
            if int(label) in arc_set
        }
        if overlap:
            return max(overlap)
        if len(edge_cycle) == 2 and len(arc_labels) == 1:
            return int(arc_labels[0])
    if not arc_labels:
        side_labels: List[int] = []
        for edge_id, forward in edge_cycle:
            edge = arrangement.edges[int(edge_id)]
            if edge.kind == "intersection_curve":
                label = edge.left_label if forward else edge.right_label
                if label is not None:
                    side_labels.append(int(label))
        if not side_labels:
            return None
        counts = defaultdict(int)
        for label in side_labels:
            counts[int(label)] += 1
        return max(sorted(counts), key=lambda x: counts[x])
    counts: Dict[int, int] = defaultdict(int)
    for label in arc_labels:
        counts[int(label)] += 1
    return max(sorted(counts), key=lambda x: counts[x])


def _build_arrangement_outgoing(
    arrangement: FeatureArrangement,
    *,
    tie_epsilon: float = 1e-12,
) -> Dict[int, List[Tuple[float, int, bool]]]:
    """
    CCW-sorted outgoing halfedges with comb nesting for parallel edges.

    When a boundary arc and an intersection curve share endpoints, their
    halfangles coincide; without a consistent tie order the DCEL walk merges
    all wedges into one degenerate outer cycle.
    """
    pair_edges: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for edge in arrangement.edges:
        n0, n1 = int(edge.node0), int(edge.node1)
        pair_edges[(min(n0, n1), max(n0, n1))].append(int(edge.id))

    edge_rank: Dict[int, int] = {}
    pair_count: Dict[Tuple[int, int], int] = {}
    for pair, eids in pair_edges.items():
        pair_count[pair] = len(eids)
        for rank, eid in enumerate(
            sorted(eids, key=lambda x: (str(arrangement.edges[x].kind), x))
        ):
            edge_rank[eid] = rank

    outgoing: Dict[int, List[Tuple[float, int, bool]]] = defaultdict(list)
    for edge in arrangement.edges:
        n0, n1 = int(edge.node0), int(edge.node1)
        p0 = arrangement.nodes[n0].uv
        p1 = arrangement.nodes[n1].uv
        base_forward = float(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
        base_backward = float(np.arctan2(p0[1] - p1[1], p0[0] - p1[0]))
        u, v = min(n0, n1), max(n0, n1)
        pair = (u, v)
        rk = edge_rank[int(edge.id)]
        nc = pair_count[pair]
        # u->v uses increasing rank; v->u uses decreasing rank so wedges nest.
        forward_offset = rk if n0 == u and n1 == v else (nc - 1 - rk)
        backward_offset = rk if n1 == u and n0 == v else (nc - 1 - rk)
        outgoing[n0].append(
            (base_forward + forward_offset * tie_epsilon, int(edge.id), True)
        )
        outgoing[n1].append(
            (base_backward + backward_offset * tie_epsilon, int(edge.id), False)
        )
    for node_id in outgoing:
        outgoing[node_id].sort(key=lambda item: item[0])
    return outgoing


def extract_cells_from_arrangement(
    arrangement: FeatureArrangement,
    *,
    min_area: float = 1e-10,
) -> List[GraphCell]:
    """Extract face cycles from an embedded graph using a DCEL-style traversal.

    [当前未使用] 生产剖分已改为 ``curve_arc_partition``，见 ``hole_analyzer``。
    """
    if not arrangement.nodes or not arrangement.edges:
        arrangement.cells = []
        return []

    outgoing = _build_arrangement_outgoing(arrangement)

    def head(edge_id: int, forward: bool) -> int:
        e = arrangement.edges[int(edge_id)]
        return int(e.node1 if forward else e.node0)

    def next_halfedge(edge_id: int, forward: bool) -> Tuple[int, bool]:
        at = head(edge_id, forward)
        rev = (int(edge_id), not bool(forward))
        entries = outgoing.get(at, [])
        rev_idx = next((i for i, item in enumerate(entries) if (item[1], item[2]) == rev), -1)
        if rev_idx < 0 or not entries:
            return rev
        # Pick previous in cyclic angular order to keep the face on the left.
        nxt = entries[(rev_idx - 1) % len(entries)]
        return int(nxt[1]), bool(nxt[2])

    visited: Set[Tuple[int, bool]] = set()
    cells: List[GraphCell] = []
    for edge in arrangement.edges:
        for forward in (True, False):
            start = (int(edge.id), bool(forward))
            if start in visited:
                continue
            cur = start
            cycle: List[Tuple[int, bool]] = []
            for _ in range(max(8, 4 * len(arrangement.edges) + 8)):
                if cur in visited and cur != start:
                    break
                visited.add(cur)
                cycle.append(cur)
                cur = next_halfedge(cur[0], cur[1])
                if cur == start:
                    break
            if cur != start or len(cycle) < 2:
                continue
            pts, sources = assemble_cell_boundary(arrangement, cycle)
            if pts.shape[0] < 3:
                continue
            uv, _center, _u, _v = project_points_to_best_plane(pts)
            uv_area = signed_polygon_area_2d(uv)
            if abs(uv_area) <= min_area:
                continue
            eps = max(1e-12, 1e-10 * max(float(np.linalg.norm(np.ptp(uv, axis=0))), 1.0))
            if not polygon_is_simple_2d(uv, eps):
                continue
            patch_label = _cell_label_from_edges(arrangement, cycle)
            if patch_label is None:
                continue
            area = polygon_area_3d(pts)
            if area <= min_area:
                continue
            cells.append(
                GraphCell(
                    id=len(cells),
                    patch_label=int(patch_label),
                    edge_cycle=list(cycle),
                    closed_points=pts,
                    sources=sources,
                    area=float(area),
                    uv_area=float(abs(uv_area)),
                    validity_score=1.0,
                )
            )

    arrangement.cells = cells
    v_count = len(arrangement.nodes)
    e_count = len(arrangement.edges)
    arrangement.euler_residual = int(v_count - e_count + len(cells) - 1)
    arrangement.diagnostics["node_count"] = v_count
    arrangement.diagnostics["edge_count"] = e_count
    arrangement.diagnostics["cell_count"] = len(cells)
    arrangement.diagnostics["euler_residual"] = arrangement.euler_residual
    return cells


def validate_patch_cell(
    cell: PatchCell,
    *,
    max_area: float,
) -> PatchCell:
    """判定 patch cell 是否应参与补洞。"""
    if not cell.closed_loop_vertex_ids or len(cell.closed_loop_vertex_ids) < 3:
        cell.is_active = False
        cell.inactive_reason = "too_few_vertices"
        return cell
    pts = cell.closed_loop_vertex_ids  # placeholder; caller sets area on points array
    _ = pts
    if cell.area <= 0.0:
        cell.is_active = False
        cell.inactive_reason = "zero_area"
        return cell
    if max_area > 0.0 and cell.area < 0.02 * max_area:
        n = len(cell.closed_loop_vertex_ids)
        if n <= 3:
            cell.is_active = False
            cell.inactive_reason = "degenerate_small_cell"
    return cell


def aggregate_curve_confidence(curves: Sequence[IntersectionCurve]) -> float:
    if not curves:
        return 0.0
    weights = {"high": 1.0, "medium": 0.65, "low": 0.35, "none": 0.0}
    vals = [weights.get(str(c.curve_confidence), 0.5) for c in curves]
    return float(sum(vals) / len(vals))


def aggregate_clip_confidence(segments: Sequence[BoundedCurveSegment]) -> float:
    if not segments:
        return 0.0
    weights = {
        "high": 1.0,
        "medium": 0.65,
        "low": 0.35,
        "corner_endpoints": 0.55,
        "corner_junction": 0.45,
        "none": 0.0,
    }
    vals = [weights.get(str(s.clip_confidence), 0.5) for s in segments]
    return float(sum(vals) / len(vals))
