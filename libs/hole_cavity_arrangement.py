#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
孔腔约束的特征 arrangement（L2 核心）

拓扑规则（无模板）
-----------------
给定孔腔 C、解析面 {S_ℓ}、全部 carrier L_ij = S_i ∩ S_j：

1. **节点** V = ∂H 多标签角点 ∪ { p ∈ C : ∃ 两 carrier 在 p 解析相交且 ‖p−c‖ 有界 }
2. **边**  沿每条 L_ij，取落在该线上的节点，按参数排序，在相邻节点间连边；
   若仅有 mesh 角点而无内部节点，则沿 carrier 朝孔心延伸至腔体内汇交点。
3. **Γ_ij** = L_ij 上所有边的并（restrict 到 C）。

节点由 carrier 联立确定，禁止 per-pair 独立延伸后再距离聚类。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from .surface_fitting import SurfaceFit
from .surface_intersections import (
    AnalyticCurve,
    BoundedCurveSegment,
    IntersectionCurve,
    analytic_intersection,
    bounded_segment_to_intersection_curve,
    feature_curve_sample_count,
    intersect_analytic_curves,
    recover_curve_between_points,
    _analytic_curve_as_line,
    _hole_inward_junction_on_analytic,
    _point_in_loop_polygon_3d,
    _surface_residual_and_gradient,
    _transition_corners_for_pair,
)


@dataclass(frozen=True)
class HoleCavity:
    """孔腔 C：由孔环投影多边形与孔心锚定。"""

    center: np.ndarray
    loop: Tuple[int, ...]
    mean_edge_length: float
    bbox_diag: float
    junction_cluster_tol: float

    def contains(self, vertices: np.ndarray, point: np.ndarray) -> bool:
        return _point_in_loop_polygon_3d(
            np.asarray(vertices, dtype=np.float64),
            self.loop,
            np.asarray(point, dtype=np.float64).reshape(3),
        )


@dataclass
class ArrangementNode:
    """Arrangement 图节点：mesh 角点 (vertex_id≥0) 或腔体内汇交 (vertex_id<0)。"""

    node_id: int
    position: np.ndarray
    vertex_id: int
    incident_pairs: Set[Tuple[int, int]] = field(default_factory=set)


# 兼容旧名
CavityJunctionNode = ArrangementNode


@dataclass
class CavityArrangementResult:
    """孔腔 arrangement 恢复结果（供 L2/L3 消费）。"""

    cavity: HoleCavity
    curves: List[IntersectionCurve]
    bounded_segments: List[BoundedCurveSegment]
    analytic_curves: List[AnalyticCurve]
    junction_nodes: List[ArrangementNode]
    junction_point: Optional[np.ndarray]
    junction_confidence: str
    diagnostics: Dict[str, object] = field(default_factory=dict)


@dataclass
class CarrierCertificate:
    """几何证书：carrier 是否属于 cavity 局部结构。"""

    pair: Tuple[int, int]
    certified: bool
    cavity_distance: float
    sample_in_cavity: bool
    reject_reason: str = ""


@dataclass
class JunctionCertificate:
    """几何证书：junction 是否为 cavity 内合法汇交点。"""

    node_id: int
    vertex_id: int
    certified: bool
    in_cavity: bool
    max_surface_residual: float
    n_incident_carriers: int
    n_supported_labels: int
    reject_reason: str = ""


@dataclass
class EdgeCertificate:
    """几何证书：topology segment 是否属于 cavity cell complex。"""

    edge_id: int
    pair: Tuple[int, int]
    certified: bool
    midpoint_in_cavity: bool
    max_endpoint_carrier_dist: float
    optional_edge: bool
    reject_reason: str = ""


def _residual_tol(cavity: HoleCavity) -> float:
    return max(0.02 * float(cavity.bbox_diag), 0.35 * float(cavity.mean_edge_length), 1e-6)


def _labels_for_pair(
    pair: Tuple[int, int],
    patch_surface_fits: Mapping[int, SurfaceFit],
) -> List[int]:
    return [int(l) for l in pair if int(l) in patch_surface_fits]


def _max_surface_residual_at_point(
    point: np.ndarray,
    labels: Sequence[int],
    patch_surface_fits: Mapping[int, SurfaceFit],
) -> float:
    p = np.asarray(point, dtype=np.float64).reshape(3)
    residuals: List[float] = []
    for label in labels:
        fit = patch_surface_fits.get(int(label))
        if fit is None:
            continue
        residual, _ = _surface_residual_and_gradient(fit, p)
        residuals.append(abs(float(residual)))
    return max(residuals) if residuals else float("inf")


def _certify_carrier(
    pair: Tuple[int, int],
    analytic: AnalyticCurve,
    cavity: HoleCavity,
    vertices: np.ndarray,
) -> CarrierCertificate:
    guide = np.asarray(cavity.center, dtype=np.float64).reshape(3)
    dist = _distance_point_to_carrier(guide, analytic, guide)
    line = _analytic_curve_as_line(analytic, guide)
    sample_in_cavity = False
    if line is not None:
        lp, ld = line
        rel = guide - np.asarray(lp, dtype=np.float64).reshape(3)
        t = float(np.dot(rel, ld))
        sample = np.asarray(lp, dtype=np.float64).reshape(3) + t * ld
        sample_in_cavity = bool(cavity.contains(vertices, sample))
    tol = max(_line_tol(cavity), _residual_tol(cavity))
    certified = bool(sample_in_cavity or dist <= tol)
    reject = ""
    if not certified:
        reject = "carrier_far_from_cavity"
    return CarrierCertificate(
        pair=tuple(sorted((int(pair[0]), int(pair[1])))),
        certified=certified,
        cavity_distance=float(dist),
        sample_in_cavity=bool(sample_in_cavity),
        reject_reason=reject,
    )


def _gate_certified_carriers(
    carriers: Mapping[Tuple[int, int], AnalyticCurve],
    cavity: HoleCavity,
    vertices: np.ndarray,
    *,
    filter_carriers: bool = False,
) -> Tuple[Dict[Tuple[int, int], AnalyticCurve], List[Dict[str, object]]]:
    reports: List[Dict[str, object]] = []
    gated: Dict[Tuple[int, int], AnalyticCurve] = {}
    for pair, analytic in carriers.items():
        cert = _certify_carrier(pair, analytic, cavity, vertices)
        reports.append(
            {
                "pair": list(cert.pair),
                "certified": bool(cert.certified),
                "cavity_distance": float(cert.cavity_distance),
                "sample_in_cavity": bool(cert.sample_in_cavity),
                "reject_reason": str(cert.reject_reason),
            }
        )
        if not filter_carriers or cert.certified:
            gated[tuple(sorted(pair))] = analytic
    if not filter_carriers:
        gated = dict(carriers)
    return gated, reports


def _certify_junction(
    node: ArrangementNode,
    carriers: Mapping[Tuple[int, int], AnalyticCurve],
    cavity: HoleCavity,
    vertices: np.ndarray,
    patch_surface_fits: Mapping[int, SurfaceFit],
) -> JunctionCertificate:
    pos = np.asarray(node.position, dtype=np.float64).reshape(3)
    if int(node.vertex_id) >= 0:
        labels = sorted(
            {
                int(l)
                for pair in node.incident_pairs
                for l in pair
                if int(l) in patch_surface_fits
            }
        )
        return JunctionCertificate(
            node_id=int(node.node_id),
            vertex_id=int(node.vertex_id),
            certified=True,
            in_cavity=True,
            max_surface_residual=_max_surface_residual_at_point(
                pos, labels, patch_surface_fits
            ),
            n_incident_carriers=len(node.incident_pairs),
            n_supported_labels=len(labels),
            reject_reason="",
        )

    in_cavity = bool(cavity.contains(vertices, pos))
    tol = _line_tol(cavity)
    incident_pairs = set(node.incident_pairs)
    geometric_ok = 0
    supported_labels: Set[int] = set()
    for pair in sorted(incident_pairs):
        ac = carriers.get(tuple(sorted(pair)))
        if ac is None:
            continue
        supported_labels.update(int(l) for l in pair)
        if _distance_point_to_carrier(pos, ac, cavity.center) <= tol:
            geometric_ok += 1
    max_res = _max_surface_residual_at_point(
        pos, sorted(supported_labels), patch_surface_fits
    )
    res_tol = _residual_tol(cavity)
    n_topological = len(incident_pairs)
    certified = (
        in_cavity
        and max_res <= res_tol * 2.0
        and (geometric_ok >= 2 or (n_topological >= 2 and geometric_ok >= 1))
    )
    reject = ""
    if not in_cavity:
        reject = "outside_cavity"
    elif max_res > res_tol * 2.0:
        reject = "surface_residual_too_large"
    elif geometric_ok < 1 and n_topological < 2:
        reject = "insufficient_carrier_support"
    return JunctionCertificate(
        node_id=int(node.node_id),
        vertex_id=int(node.vertex_id),
        certified=bool(certified),
        in_cavity=bool(in_cavity),
        max_surface_residual=float(max_res),
        n_incident_carriers=int(max(geometric_ok, n_topological)),
        n_supported_labels=len(supported_labels),
        reject_reason=reject,
    )


def _gate_certified_junctions(
    nodes: Sequence[ArrangementNode],
    carriers: Mapping[Tuple[int, int], AnalyticCurve],
    cavity: HoleCavity,
    vertices: np.ndarray,
    patch_surface_fits: Mapping[int, SurfaceFit],
    *,
    filter_nodes: bool = False,
) -> Tuple[List[ArrangementNode], List[Dict[str, object]]]:
    reports: List[Dict[str, object]] = []
    kept: List[ArrangementNode] = []
    for node in nodes:
        cert = _certify_junction(node, carriers, cavity, vertices, patch_surface_fits)
        reports.append(
            {
                "node_id": int(cert.node_id),
                "vertex_id": int(cert.vertex_id),
                "certified": bool(cert.certified),
                "in_cavity": bool(cert.in_cavity),
                "max_surface_residual": float(cert.max_surface_residual),
                "n_incident_carriers": int(cert.n_incident_carriers),
                "n_supported_labels": int(cert.n_supported_labels),
                "reject_reason": str(cert.reject_reason),
            }
        )
        if not filter_nodes:
            kept.append(node)
            continue
        if int(node.vertex_id) >= 0:
            kept.append(node)
        elif cert.in_cavity and cert.max_surface_residual <= _residual_tol(cavity) * 3.0:
            kept.append(node)
    return kept, reports


def _certify_edge_segment(
    edge_id: int,
    seg: BoundedCurveSegment,
    cavity: HoleCavity,
    vertices: np.ndarray,
    *,
    optional_edge: bool,
) -> EdgeCertificate:
    pts = np.asarray(seg.curve_points, dtype=np.float64)
    pair = tuple(sorted(int(x) for x in seg.analytic.patch_pair))
    if pts.ndim != 2 or pts.shape[0] < 2:
        return EdgeCertificate(
            edge_id=int(edge_id),
            pair=pair,
            certified=False,
            midpoint_in_cavity=False,
            max_endpoint_carrier_dist=float("inf"),
            optional_edge=bool(optional_edge),
            reject_reason="degenerate_segment",
        )
    mid = np.mean(pts, axis=0)
    midpoint_in_cavity = bool(cavity.contains(vertices, mid))
    guide = cavity.center
    dists = [
        _distance_point_to_carrier(pts[0], seg.analytic, guide),
        _distance_point_to_carrier(pts[-1], seg.analytic, guide),
    ]
    max_dist = float(max(dists))
    tol = max(_line_tol(cavity), _residual_tol(cavity))
    certified = bool(midpoint_in_cavity and max_dist <= 2.5 * tol)
    reject = ""
    if not midpoint_in_cavity:
        reject = "midpoint_outside_cavity"
    elif max_dist > 2.5 * tol:
        reject = "endpoints_off_carrier"
    return EdgeCertificate(
        edge_id=int(edge_id),
        pair=pair,
        certified=bool(certified),
        midpoint_in_cavity=bool(midpoint_in_cavity),
        max_endpoint_carrier_dist=float(max_dist),
        optional_edge=bool(optional_edge),
        reject_reason=reject,
    )


def build_hole_cavity(
    vertices: np.ndarray,
    loop: Sequence[int],
    hole_center: np.ndarray,
    *,
    loop_mean_edge: Optional[float] = None,
) -> HoleCavity:
    verts = np.asarray(vertices, dtype=np.float64)
    loop_tuple = tuple(int(v) for v in loop)
    loop_pts = verts[np.asarray(loop_tuple, dtype=np.int64)]
    mean_edge = float(loop_mean_edge or 0.0)
    if mean_edge <= 1e-15 and len(loop_tuple) >= 2:
        perim = 0.0
        for i in range(len(loop_tuple)):
            a = int(loop_tuple[i])
            b = int(loop_tuple[(i + 1) % len(loop_tuple)])
            perim += float(np.linalg.norm(verts[b] - verts[a]))
        mean_edge = perim / float(len(loop_tuple))
    diag = max(float(np.linalg.norm(np.ptp(loop_pts, axis=0))), 1e-12)
    cluster_tol = max(1e-6, 0.45 * float(mean_edge), 0.015 * float(diag))
    return HoleCavity(
        center=np.asarray(hole_center, dtype=np.float64).reshape(3),
        loop=loop_tuple,
        mean_edge_length=max(mean_edge, 1e-15),
        bbox_diag=float(diag),
        junction_cluster_tol=float(cluster_tol),
    )


def _loop_set(loop: Sequence[int]) -> Set[int]:
    return {int(v) for v in loop}


def _line_tol(cavity: HoleCavity) -> float:
    return max(
        0.45 * float(cavity.mean_edge_length),
        0.04 * float(cavity.bbox_diag),
        1e-5 * float(cavity.bbox_diag),
    )


def _node_merge_tol(cavity: HoleCavity) -> float:
    return max(float(cavity.junction_cluster_tol), 0.06 * float(cavity.bbox_diag))


def _max_extent(cavity: HoleCavity) -> float:
    return 2.5 * float(cavity.bbox_diag)


def _distance_point_to_carrier(point: np.ndarray, analytic: AnalyticCurve, guide: np.ndarray) -> float:
    guide_pt = np.asarray(guide, dtype=np.float64).reshape(3)
    line = _analytic_curve_as_line(analytic, guide_pt)
    if line is None:
        return float("inf")
    lp, ld = line
    rel = np.asarray(point, dtype=np.float64).reshape(3) - np.asarray(lp, dtype=np.float64)
    return float(np.linalg.norm(rel - float(np.dot(rel, ld)) * ld))


def _param_on_carrier(point: np.ndarray, analytic: AnalyticCurve, guide: np.ndarray) -> float:
    line = _analytic_curve_as_line(analytic, np.asarray(guide, dtype=np.float64).reshape(3))
    if line is None:
        return 0.0
    lp, ld = line
    rel = np.asarray(point, dtype=np.float64).reshape(3) - np.asarray(lp, dtype=np.float64)
    return float(np.dot(rel, ld))


def _active_labels(
    patch_surface_fits: Mapping[int, SurfaceFit],
    patch_pairs: Iterable[Tuple[int, int]],
) -> List[int]:
    labels = {int(l) for l in patch_surface_fits}
    for a, b in patch_pairs:
        labels.add(int(a))
        labels.add(int(b))
    return sorted(labels)


def _all_carrier_pairs(labels: Sequence[int]) -> List[Tuple[int, int]]:
    return [tuple(sorted((int(a), int(b)))) for a, b in combinations(labels, 2)]


def _build_carrier_map(
    patch_surface_fits: Mapping[int, SurfaceFit],
    labels: Sequence[int],
) -> Dict[Tuple[int, int], AnalyticCurve]:
    out: Dict[Tuple[int, int], AnalyticCurve] = {}
    for pair in _all_carrier_pairs(labels):
        if pair[0] not in patch_surface_fits or pair[1] not in patch_surface_fits:
            continue
        ac = analytic_intersection(patch_surface_fits[pair[0]], patch_surface_fits[pair[1]])
        if ac is not None:
            out[pair] = ac
    return out


def _mesh_corner_vertices(
    loop: Sequence[int],
    vertex_labels: Mapping[int, Sequence[int]],
    *,
    arc_corner_hints: Optional[Mapping[Tuple[int, int], Sequence[int]]] = None,
) -> Set[int]:
    loop_set = _loop_set(loop)
    corners: Set[int] = set()
    if arc_corner_hints:
        for verts in arc_corner_hints.values():
            corners.update(int(v) for v in verts if int(v) in loop_set)
    if len(corners) >= 2:
        return corners
    for pair in combinations(_active_labels_from_corners(vertex_labels), 2):
        for v in _transition_corners_for_pair(tuple(sorted(pair)), vertex_labels):
            if int(v) in loop_set:
                corners.add(int(v))
    for v, labels in vertex_labels.items():
        if int(v) not in loop_set:
            continue
        if len({int(x) for x in labels}) >= 2:
            corners.add(int(v))
    return corners


def _active_labels_from_corners(vertex_labels: Mapping[int, Sequence[int]]) -> List[int]:
    labels: Set[int] = set()
    for values in vertex_labels.values():
        labels.update(int(x) for x in values)
    return sorted(labels)


def _collect_corner_traces(
    mesh_corners: Set[int],
    carriers: Mapping[Tuple[int, int], AnalyticCurve],
    cavity: HoleCavity,
    vertices: np.ndarray,
    loop: Sequence[int],
    vertex_labels: Mapping[int, Sequence[int]],
    *,
    arc_corner_hints: Optional[Mapping[Tuple[int, int], Sequence[int]]] = None,
    protected_boundary_pairs: Optional[Set[Tuple[int, int]]] = None,
) -> List[Tuple[np.ndarray, int, Tuple[int, int]]]:
    """角点沿 incident carrier 朝孔心的追踪点 (pos, corner_vertex_id, pair)。"""
    out: List[Tuple[np.ndarray, int, Tuple[int, int]]] = []
    protected = {tuple(sorted(p)) for p in (protected_boundary_pairs or set())}
    for corner in sorted(mesh_corners):
        for pair, analytic in carriers.items():
            key = tuple(sorted(pair))
            if key in protected:
                continue
            if not _corner_on_pair(
                int(corner),
                pair,
                vertex_labels,
                arc_corner_hints=arc_corner_hints,
            ):
                continue
            pos = _trace_virtual_on_carrier(
                int(corner), analytic, cavity, vertices, loop
            )
            out.append(
                (
                    np.asarray(pos, dtype=np.float64).reshape(3),
                    int(corner),
                    tuple(sorted(pair)),
                )
            )
    return out


def _boundary_pair_certificate_records(
    carriers: Mapping[Tuple[int, int], AnalyticCurve],
    mesh_corners: Set[int],
    cavity: HoleCavity,
    vertices: np.ndarray,
    vertex_labels: Mapping[int, Sequence[int]],
    *,
    arc_corner_hints: Optional[Mapping[Tuple[int, int], Sequence[int]]] = None,
) -> Tuple[Dict[Tuple[int, int], Tuple[int, int]], List[Dict[str, object]]]:
    """
    L2 boundary-paired carrier certificate.

    If a carrier has exactly two boundary anchors, those anchors already define
    the bounded feature interval; they should not be pulled inward to a virtual
    junction.
    """
    accepted: Dict[Tuple[int, int], Tuple[int, int]] = {}
    reports: List[Dict[str, object]] = []
    two_anchor_candidates: List[Tuple[Tuple[int, int], Tuple[int, int], int]] = []
    for pair in sorted({tuple(sorted(p)) for p in carriers}):
        anchors = [
            int(v)
            for v in sorted(mesh_corners)
            if _corner_on_pair(
                int(v),
                pair,
                vertex_labels,
                arc_corner_hints=arc_corner_hints,
            )
        ]
        record: Dict[str, object] = {
            "pair": [int(pair[0]), int(pair[1])],
            "anchor_vertices": list(anchors),
            "accepted": False,
        }
        if len(anchors) != 2:
            record["reject_reason"] = "not_exactly_two_boundary_anchors"
            reports.append(record)
            continue
        two_anchor_candidates.append(
            (pair, (int(anchors[0]), int(anchors[1])), len(reports))
        )
        p0 = np.asarray(vertices[int(anchors[0])], dtype=np.float64).reshape(3)
        p1 = np.asarray(vertices[int(anchors[1])], dtype=np.float64).reshape(3)
        mid = 0.5 * (p0 + p1)
        if not cavity.contains(vertices, mid):
            record["reject_reason"] = "midpoint_outside_cavity"
            reports.append(record)
            continue
        accepted[pair] = (int(anchors[0]), int(anchors[1]))
        record["accepted"] = True
        record["reject_reason"] = ""
        reports.append(record)
    if len(mesh_corners) >= 6 and len(mesh_corners) % 2 == 0:
        target = {int(v) for v in mesh_corners}
        best_cover: Optional[List[Tuple[Tuple[int, int], Tuple[int, int], int]]] = None

        def _search(
            start: int,
            chosen: List[Tuple[Tuple[int, int], Tuple[int, int], int]],
            used: Set[int],
        ) -> None:
            nonlocal best_cover
            if best_cover is not None:
                return
            if used == target:
                best_cover = list(chosen)
                return
            if len(used) >= len(target):
                return
            for idx in range(start, len(two_anchor_candidates)):
                pair, anchors, report_idx = two_anchor_candidates[idx]
                aset = {int(anchors[0]), int(anchors[1])}
                if used & aset:
                    continue
                chosen.append((pair, anchors, report_idx))
                _search(idx + 1, chosen, used | aset)
                chosen.pop()

        _search(0, [], set())
        if best_cover is not None and len(best_cover) * 2 == len(target):
            accepted = {}
            accepted_pairs = {tuple(pair) for pair, _anchors, _idx in best_cover}
            for pair, anchors, report_idx in best_cover:
                accepted[tuple(pair)] = (int(anchors[0]), int(anchors[1]))
                reports[report_idx]["accepted"] = True
                reports[report_idx]["reject_reason"] = ""
                reports[report_idx]["accepted_by_complete_pair_cover"] = True
            for report in reports:
                pair = tuple(int(x) for x in report.get("pair", ()))
                if pair and pair not in accepted_pairs:
                    report["accepted"] = False
                    if not str(report.get("reject_reason", "")):
                        report["reject_reason"] = "not_in_complete_pair_cover"
    return accepted, reports


def _trace_coherence_tol(cavity: HoleCavity) -> float:
    return max(
        _node_merge_tol(cavity),
        2.5 * float(cavity.mean_edge_length),
        0.06 * float(cavity.bbox_diag),
    )


def _consensus_nodes_from_traces(
    traces: Sequence[Tuple[np.ndarray, int, Tuple[int, int]]],
    vertex_labels: Mapping[int, Sequence[int]],
    cavity: HoleCavity,
) -> List[Tuple[np.ndarray, Set[Tuple[int, int]]]]:
    """
    角点追踪点：共享 patch 标签且相互一致 → 合并为汇交节点。

    返回 (位置, 关联 carrier 对)；后者由追踪来源确定，避免均值偏移后几何失配。
    """
    if not traces:
        return []
    tol = _trace_coherence_tol(cavity)
    n = len(traces)
    parent = list(range(n))

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def _union(i: int, j: int) -> None:
        ri, rj = _find(i), _find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        pos_i, ci, _ = traces[i]
        labels_i = {int(x) for x in vertex_labels.get(int(ci), ())}
        for j in range(i + 1, n):
            pos_j, cj, _ = traces[j]
            labels_j = {int(x) for x in vertex_labels.get(int(cj), ())}
            if not labels_i & labels_j:
                continue
            if float(np.linalg.norm(pos_i - pos_j)) <= tol:
                _union(i, j)

    groups: Dict[int, List[Tuple[np.ndarray, Tuple[int, int]]]] = {}
    for i in range(n):
        root = _find(i)
        groups.setdefault(root, []).append(
            (
                np.asarray(traces[i][0], dtype=np.float64).reshape(3),
                tuple(sorted(traces[i][2])),
            )
        )
    out: List[Tuple[np.ndarray, Set[Tuple[int, int]]]] = []
    for items in groups.values():
        pos = np.mean(np.vstack([item[0] for item in items]), axis=0)
        pairs = {tuple(sorted(pair)) for _, pair in items}
        out.append((pos, pairs))
    return out


def _pairwise_intersection_candidates(
    carriers: Mapping[Tuple[int, int], AnalyticCurve],
    cavity: HoleCavity,
    vertices: np.ndarray,
) -> List[np.ndarray]:
    hc = cavity.center
    max_extent = _max_extent(cavity)
    raw: List[np.ndarray] = []
    items = list(carriers.items())
    for i, (pair_a, curve_a) in enumerate(items):
        for pair_b, curve_b in items[i + 1 :]:
            if len(set(pair_a) & set(pair_b)) != 1:
                continue
            pt = intersect_analytic_curves(curve_a, curve_b, guide_point=hc)
            if pt is None:
                continue
            p = np.asarray(pt, dtype=np.float64).reshape(3)
            if float(np.linalg.norm(p - hc)) > max_extent:
                continue
            incident = _incident_pairs_for_point(p, carriers, cavity)
            # A true multi-carrier junction can lie just outside the projected
            # loop polygon for concave/non-planar openings; keep it if the
            # analytic carriers themselves certify a local corner.
            if not cavity.contains(vertices, p) and len(incident) < 3:
                continue
            raw.append(p)
    return raw


def _merge_trace_node_candidates(
    nodes: Sequence[Tuple[np.ndarray, Set[Tuple[int, int]]]],
    cavity: HoleCavity,
) -> List[Tuple[np.ndarray, Set[Tuple[int, int]]]]:
    if not nodes:
        return []
    merge_tol = _node_merge_tol(cavity)
    clusters: List[List[Tuple[np.ndarray, Set[Tuple[int, int]]]]] = []
    for pos, pairs in nodes:
        placed = False
        p_arr = np.asarray(pos, dtype=np.float64).reshape(3)
        for cluster in clusters:
            center = np.mean(np.vstack([item[0] for item in cluster]), axis=0)
            if float(np.linalg.norm(p_arr - center)) <= merge_tol:
                cluster.append((p_arr, set(pairs)))
                placed = True
                break
        if not placed:
            clusters.append([(p_arr, set(pairs))])
    out: List[Tuple[np.ndarray, Set[Tuple[int, int]]]] = []
    for cluster in clusters:
        pos = np.mean(np.vstack([item[0] for item in cluster]), axis=0)
        pairs: Set[Tuple[int, int]] = set()
        for _, item_pairs in cluster:
            pairs.update(item_pairs)
        out.append((pos, pairs))
    return out


def _merge_node_candidates(
    candidates: Sequence[np.ndarray],
    cavity: HoleCavity,
) -> List[np.ndarray]:
    if not candidates:
        return []
    merge_tol = _node_merge_tol(cavity)
    hc = cavity.center
    clusters: List[List[np.ndarray]] = []
    for p in candidates:
        placed = False
        p_arr = np.asarray(p, dtype=np.float64).reshape(3)
        for cluster in clusters:
            center = np.mean(np.vstack(cluster), axis=0)
            if float(np.linalg.norm(p_arr - center)) <= merge_tol:
                cluster.append(p_arr)
                placed = True
                break
        if not placed:
            clusters.append([p_arr])
    out: List[np.ndarray] = []
    for cluster in clusters:
        dists = [float(np.linalg.norm(pt - hc)) for pt in cluster]
        out.append(cluster[int(np.argmin(dists))])
    return out


def _discover_internal_nodes(
    carriers: Mapping[Tuple[int, int], AnalyticCurve],
    cavity: HoleCavity,
    vertices: np.ndarray,
    loop: Sequence[int],
    vertex_labels: Mapping[int, Sequence[int]],
    mesh_corners: Set[int],
    *,
    arc_corner_hints: Optional[Mapping[Tuple[int, int], Sequence[int]]] = None,
    protected_boundary_pairs: Optional[Set[Tuple[int, int]]] = None,
) -> List[Tuple[np.ndarray, Set[Tuple[int, int]]]]:
    """
    内部节点 = carrier 腔体内联立交点（过滤离群）。

    Boundary-paired carrier 已在前面直接闭合为 bounded feature interval。
    孤立角点只能作为射线起点；其终点必须由其他 carrier 联交证明，
    不能由 inward trace 或多个 trace 的均值凭空生成。
    """
    candidates: List[np.ndarray] = []
    for p in _pairwise_intersection_candidates(carriers, cavity, vertices):
        candidates.append(p)

    merged_pts = _merge_node_candidates(candidates, cavity)
    return [
        (pt, _incident_pairs_for_point(pt, carriers, cavity)) for pt in merged_pts
    ]


def _incident_pairs_for_point(
    point: np.ndarray,
    carriers: Mapping[Tuple[int, int], AnalyticCurve],
    cavity: HoleCavity,
) -> Set[Tuple[int, int]]:
    tol = _line_tol(cavity)
    guide = cavity.center
    out: Set[Tuple[int, int]] = set()
    for pair, ac in carriers.items():
        if _distance_point_to_carrier(point, ac, guide) <= tol:
            out.add(tuple(sorted(pair)))
    return out


def _closest_point_to_carrier_lines(
    pairs: Iterable[Tuple[int, int]],
    carriers: Mapping[Tuple[int, int], AnalyticCurve],
    cavity: HoleCavity,
    prior: np.ndarray,
) -> Optional[np.ndarray]:
    """多 carrier 最小二乘汇交点；trace 只作为弱先验而非节点定义。"""
    lines: List[Tuple[np.ndarray, np.ndarray]] = []
    for pair in sorted({tuple(sorted(p)) for p in pairs}):
        ac = carriers.get(pair)
        if ac is None:
            continue
        line = _analytic_curve_as_line(ac, cavity.center)
        if line is None:
            continue
        lp, ld = line
        lines.append((np.asarray(lp, dtype=np.float64), _safe_unit(ld)))
    if not lines:
        return None

    eye = np.eye(3, dtype=np.float64)
    a_mat = np.zeros((3, 3), dtype=np.float64)
    b_vec = np.zeros(3, dtype=np.float64)
    for lp, ld in lines:
        proj = eye - np.outer(ld, ld)
        a_mat += proj
        b_vec += proj @ lp

    prior_pt = np.asarray(prior, dtype=np.float64).reshape(3)
    prior_weight = max(1e-6, 0.03 * float(len(lines)))
    a_mat += prior_weight * eye
    b_vec += prior_weight * prior_pt
    try:
        point = np.linalg.solve(a_mat, b_vec)
    except np.linalg.LinAlgError:
        point = np.linalg.lstsq(a_mat, b_vec, rcond=None)[0]
    if not np.all(np.isfinite(point)):
        return None
    max_move = max(2.0 * float(cavity.bbox_diag), 1e-9)
    move = float(np.linalg.norm(point - prior_pt))
    if move > max_move:
        point = prior_pt + (point - prior_pt) * (max_move / move)
    return np.asarray(point, dtype=np.float64)


def _refine_internal_nodes_with_shared_carriers(
    nodes: Sequence[ArrangementNode],
    carriers: Mapping[Tuple[int, int], AnalyticCurve],
    cavity: HoleCavity,
    vertices: np.ndarray,
) -> List[Dict[str, object]]:
    """
    两个内部节点共享两个 surface labels 时，其 shared carrier 是五线拓扑的
    bridge 约束。将该 carrier 加入两端节点的 incident set，并用多 carrier
    最小二乘重求虚拟点位置。
    """
    diagnostics: List[Dict[str, object]] = []
    internal = [n for n in nodes if int(n.vertex_id) < 0]
    for i, node_a in enumerate(internal):
        for node_b in internal[i + 1 :]:
            labels_a = _node_patch_labels_from_pairs(node_a)
            labels_b = _node_patch_labels_from_pairs(node_b)
            shared = sorted(labels_a & labels_b)
            record: Dict[str, object] = {
                "node_ids": (int(node_a.node_id), int(node_b.node_id)),
                "shared_labels": [int(x) for x in shared],
                "accepted": False,
            }
            if len(shared) != 2:
                record["reject_reason"] = "not_two_shared_labels"
                diagnostics.append(record)
                continue
            pair = tuple(sorted((int(shared[0]), int(shared[1]))))
            record["pair"] = pair
            if pair not in carriers:
                record["reject_reason"] = "missing_carrier"
                diagnostics.append(record)
                continue
            mid = 0.5 * (
                np.asarray(node_a.position, dtype=np.float64)
                + np.asarray(node_b.position, dtype=np.float64)
            )
            if not cavity.contains(vertices, mid):
                record["reject_reason"] = "midpoint_outside_cavity"
                diagnostics.append(record)
                continue

            node_a.incident_pairs.add(pair)
            node_b.incident_pairs.add(pair)
            before_a = np.asarray(node_a.position, dtype=np.float64).copy()
            before_b = np.asarray(node_b.position, dtype=np.float64).copy()
            refined_a = _closest_point_to_carrier_lines(
                node_a.incident_pairs,
                carriers,
                cavity,
                before_a,
            )
            refined_b = _closest_point_to_carrier_lines(
                node_b.incident_pairs,
                carriers,
                cavity,
                before_b,
            )
            if refined_a is not None:
                node_a.position = refined_a
            if refined_b is not None:
                node_b.position = refined_b
            record["accepted"] = True
            record["move"] = (
                float(np.linalg.norm(np.asarray(node_a.position) - before_a)),
                float(np.linalg.norm(np.asarray(node_b.position) - before_b)),
            )
            diagnostics.append(record)
    return diagnostics


def _complete_single_junction_incident_pairs(
    nodes: Sequence[ArrangementNode],
    carriers: Mapping[Tuple[int, int], AnalyticCurve],
) -> List[Dict[str, object]]:
    """
    Conservative triple-junction closure.

    If a single virtual node is already supported by three surface labels, every
    pair among those labels is a candidate carrier incident to the same junction.
    Cell validation still decides whether the completed graph is acceptable.
    """
    internal = [n for n in nodes if int(n.vertex_id) < 0]
    diagnostics: List[Dict[str, object]] = []
    if len(internal) != 1:
        return diagnostics
    node = internal[0]
    labels = sorted(_node_patch_labels_from_pairs(node))
    if len(labels) != 3:
        return diagnostics
    added: List[Tuple[int, int]] = []
    for pair in combinations(labels, 2):
        key = tuple(sorted((int(pair[0]), int(pair[1]))))
        if key not in carriers or key in node.incident_pairs:
            continue
        node.incident_pairs.add(key)
        added.append(key)
    if added:
        diagnostics.append(
            {
                "node_id": int(node.node_id),
                "labels": [int(x) for x in labels],
                "added_pairs": [(int(a), int(b)) for a, b in added],
            }
        )
    return diagnostics


def _build_arrangement_nodes(
    carriers: Mapping[Tuple[int, int], AnalyticCurve],
    cavity: HoleCavity,
    vertices: np.ndarray,
    loop: Sequence[int],
    vertex_labels: Mapping[int, Sequence[int]],
    *,
    arc_corner_hints: Optional[Mapping[Tuple[int, int], Sequence[int]]] = None,
    protected_boundary_pairs: Optional[Set[Tuple[int, int]]] = None,
) -> List[ArrangementNode]:
    mesh_corners = _mesh_corner_vertices(loop, vertex_labels, arc_corner_hints=arc_corner_hints)
    internal_pts = _discover_internal_nodes(
        carriers,
        cavity,
        vertices,
        loop,
        vertex_labels,
        mesh_corners,
        arc_corner_hints=arc_corner_hints,
        protected_boundary_pairs=protected_boundary_pairs,
    )
    nodes: List[ArrangementNode] = []
    for vi in sorted(mesh_corners):
        pos = np.asarray(vertices[int(vi)], dtype=np.float64).reshape(3)
        nodes.append(
            ArrangementNode(
                node_id=len(nodes),
                position=pos,
                vertex_id=int(vi),
                incident_pairs=_incident_pairs_for_point(pos, carriers, cavity),
            )
        )
    for pt, trace_pairs in internal_pts:
        incident = set(trace_pairs) if trace_pairs else _incident_pairs_for_point(
            pt, carriers, cavity
        )
        nodes.append(
            ArrangementNode(
                node_id=len(nodes),
                position=np.asarray(pt, dtype=np.float64).reshape(3),
                vertex_id=-1,
                incident_pairs=incident,
            )
        )
    return nodes


def _corner_on_pair(
    vertex_id: int,
    pair: Tuple[int, int],
    vertex_labels: Mapping[int, Sequence[int]],
    *,
    arc_corner_hints: Optional[Mapping[Tuple[int, int], Sequence[int]]] = None,
) -> bool:
    key = tuple(sorted(pair))
    if arc_corner_hints is not None and int(vertex_id) in {
        int(v) for v in arc_corner_hints.get(key, ())
    }:
        return True
    labels = {int(x) for x in vertex_labels.get(int(vertex_id), ())}
    return int(pair[0]) in labels and int(pair[1]) in labels


def _nodes_on_carrier(
    pair: Tuple[int, int],
    analytic: AnalyticCurve,
    nodes: Sequence[ArrangementNode],
    cavity: HoleCavity,
    vertex_labels: Mapping[int, Sequence[int]],
    *,
    arc_corner_hints: Optional[Mapping[Tuple[int, int], Sequence[int]]] = None,
) -> List[ArrangementNode]:
    tol = _line_tol(cavity)
    guide = cavity.center
    key = tuple(sorted(pair))
    on_line: List[ArrangementNode] = []
    for node in nodes:
        if key in node.incident_pairs:
            on_line.append(node)
            continue
        if int(node.vertex_id) >= 0 and _corner_on_pair(
            int(node.vertex_id),
            pair,
            vertex_labels,
            arc_corner_hints=arc_corner_hints,
        ):
            on_line.append(node)
            continue
        if _distance_point_to_carrier(node.position, analytic, guide) > tol:
            continue
        if node.vertex_id >= 0:
            if not _corner_on_pair(
                int(node.vertex_id),
                pair,
                vertex_labels,
                arc_corner_hints=arc_corner_hints,
            ):
                # 几何落在 carrier 上的 mesh 角点仍纳入（圆柱面等角点标签可能不全）
                pass
        on_line.append(node)
    return on_line


def _trace_virtual_on_carrier(
    corner_id: int,
    analytic: AnalyticCurve,
    cavity: HoleCavity,
    vertices: np.ndarray,
    loop: Sequence[int],
) -> np.ndarray:
    corner_xyz = np.asarray(vertices[int(corner_id)], dtype=np.float64).reshape(3)
    return _hole_inward_junction_on_analytic(
        analytic,
        corner_xyz,
        cavity.center,
        vertices=vertices,
        loop=loop,
    )


def _segment_between_nodes(
    node_a: ArrangementNode,
    node_b: ArrangementNode,
    analytic: AnalyticCurve,
    cavity: HoleCavity,
    vertices: np.ndarray,
    patch_surface_fits: Mapping[int, SurfaceFit],
    *,
    guide_point: Optional[np.ndarray] = None,
    require_mesh_midpoint_in_cavity: bool = True,
) -> Optional[BoundedCurveSegment]:
    pair = tuple(sorted(analytic.patch_pair))
    if pair[0] not in patch_surface_fits or pair[1] not in patch_surface_fits:
        return None
    pa = np.asarray(node_a.position, dtype=np.float64).reshape(3)
    pb = np.asarray(node_b.position, dtype=np.float64).reshape(3)
    if float(np.linalg.norm(pb - pa)) <= 1e-12:
        return None
    va = int(node_a.vertex_id)
    vb = int(node_b.vertex_id)
    mid = 0.5 * (pa + pb)
    # mesh↔virtual 段由拓扑确定；环投影对边界角点不可靠（hole_test2）。
    if require_mesh_midpoint_in_cavity and va >= 0 and vb >= 0:
        if not cavity.contains(vertices, mid):
            return None
    ref = 0.75 * cavity.mean_edge_length
    length = float(np.linalg.norm(pb - pa))
    target_n = feature_curve_sample_count(length, max(ref, 1e-15))
    ic = recover_curve_between_points(
        patch_surface_fits[pair[0]],
        patch_surface_fits[pair[1]],
        pa,
        pb,
        cavity.center if guide_point is None else np.asarray(guide_point, dtype=np.float64),
        n_samples=target_n,
        min_samples=0,
        endpoint_vertex_indices=(va, vb),
        intersection_sampling_reference_step=ref,
    )
    pts = np.asarray(ic.curve_points, dtype=np.float64)
    if pts.shape[0] < 2:
        return None
    kind = "topology_virtual_bridge"
    if va >= 0 and vb >= 0:
        kind = "topology_mesh_mesh"
    elif va >= 0 or vb >= 0:
        kind = "topology_mesh_virtual"
    return BoundedCurveSegment(
        analytic=analytic,
        t_start=0.0,
        t_end=1.0,
        curve_points=pts,
        boundary_vertex_indices=(va, vb),
        clip_confidence=kind,
        start_xyz=pts[0].copy(),
        end_xyz=pts[-1].copy(),
    )


def _node_patch_labels_from_pairs(node: ArrangementNode) -> Set[int]:
    return {int(label) for pair in node.incident_pairs for label in pair}


def _best_internal_for_mesh_corner(
    mesh_node: ArrangementNode,
    internal_nodes: Sequence[ArrangementNode],
    vertex_labels: Mapping[int, Sequence[int]],
    cavity: HoleCavity,
    analytic: AnalyticCurve,
    vertices: np.ndarray,
    loop: Sequence[int],
) -> Optional[ArrangementNode]:
    """mesh 角点在 carrier 上无几何内部节点时，连到该 carrier 的拓扑汇交。"""
    pair_key = tuple(sorted(analytic.patch_pair))
    pair_candidates = [n for n in internal_nodes if pair_key in n.incident_pairs]
    if pair_candidates:
        mesh_pos = np.asarray(mesh_node.position, dtype=np.float64).reshape(3)
        return min(
            pair_candidates,
            key=lambda n: float(
                np.linalg.norm(np.asarray(n.position, dtype=np.float64) - mesh_pos)
            ),
        )
    corner_labels = {int(x) for x in vertex_labels.get(int(mesh_node.vertex_id), ())}
    label_candidates: List[ArrangementNode] = []
    for node in internal_nodes:
        if corner_labels & _node_patch_labels_from_pairs(node):
            label_candidates.append(node)
    if label_candidates:
        mesh_pos = np.asarray(mesh_node.position, dtype=np.float64).reshape(3)
        return min(
            label_candidates,
            key=lambda n: float(
                np.linalg.norm(np.asarray(n.position, dtype=np.float64) - mesh_pos)
            ),
        )
    return None


def _virtual_node_for_corner_on_carrier(
    corner_id: int,
    analytic: AnalyticCurve,
    cavity: HoleCavity,
    vertices: np.ndarray,
    loop: Sequence[int],
) -> ArrangementNode:
    pos = _trace_virtual_on_carrier(corner_id, analytic, cavity, vertices, loop)
    return ArrangementNode(
        node_id=-1,
        position=np.asarray(pos, dtype=np.float64).reshape(3),
        vertex_id=-1,
        incident_pairs={tuple(sorted(analytic.patch_pair))},
    )


def _sorted_carrier_nodes_toward_center(
    carrier_nodes: Sequence[ArrangementNode],
    analytic: AnalyticCurve,
    cavity: HoleCavity,
) -> List[ArrangementNode]:
    guide = cavity.center
    return sorted(
        carrier_nodes,
        key=lambda n: _param_on_carrier(n.position, analytic, guide),
    )


def _build_topology_segments(
    carriers: Mapping[Tuple[int, int], AnalyticCurve],
    nodes: Sequence[ArrangementNode],
    cavity: HoleCavity,
    vertices: np.ndarray,
    loop: Sequence[int],
    vertex_labels: Mapping[int, Sequence[int]],
    patch_surface_fits: Mapping[int, SurfaceFit],
    *,
    arc_corner_hints: Optional[Mapping[Tuple[int, int], Sequence[int]]] = None,
    paired_boundary_vertices: Optional[Mapping[Tuple[int, int], Tuple[int, int]]] = None,
) -> Tuple[List[BoundedCurveSegment], List[Dict[str, object]]]:
    """沿每条 carrier：mesh→最近内部节点；内部节点间仅保留无 mesh 分隔的相邻段。"""
    segments: List[BoundedCurveSegment] = []
    bridge_proofs: List[Dict[str, object]] = []
    seen: Set[Tuple[Tuple[int, int], int, int]] = set()
    protected_pairs = {
        tuple(sorted(pair)): tuple(int(v) for v in verts)
        for pair, verts in (paired_boundary_vertices or {}).items()
    }

    for pair, analytic in carriers.items():
        pair = tuple(sorted(pair))
        on_line = _nodes_on_carrier(
            pair,
            analytic,
            nodes,
            cavity,
            vertex_labels,
            arc_corner_hints=arc_corner_hints,
        )
        mesh_on_line = [n for n in on_line if int(n.vertex_id) >= 0]
        internal_on_line = [n for n in on_line if int(n.vertex_id) < 0]
        all_internal = [n for n in nodes if int(n.vertex_id) < 0]

        if pair in protected_pairs:
            va, vb = protected_pairs[pair]
            node_by_vertex = {
                int(n.vertex_id): n for n in mesh_on_line if int(n.vertex_id) >= 0
            }
            a = node_by_vertex.get(int(va))
            b = node_by_vertex.get(int(vb))
            if a is not None and b is not None:
                key = (pair, min(int(va), int(vb)), max(int(va), int(vb)))
                if key not in seen:
                    seg = _segment_between_nodes(
                        a,
                        b,
                        analytic,
                        cavity,
                        vertices,
                        patch_surface_fits,
                        require_mesh_midpoint_in_cavity=False,
                    )
                    if seg is not None:
                        seen.add(key)
                        segments.append(seg)
                continue

        if len(carriers) == 1 and len(mesh_on_line) == 2:
            a, b = _sorted_carrier_nodes_toward_center(
                mesh_on_line,
                analytic,
                cavity,
            )
            key = (
                pair,
                min(int(a.vertex_id), int(b.vertex_id)),
                max(int(a.vertex_id), int(b.vertex_id)),
            )
            if key not in seen:
                seg = _segment_between_nodes(
                    a,
                    b,
                    analytic,
                    cavity,
                    vertices,
                    patch_surface_fits,
                )
                if seg is not None:
                    seen.add(key)
                    segments.append(seg)
            continue

        if mesh_on_line and not internal_on_line:
            for mesh_node in mesh_on_line:
                target = _best_internal_for_mesh_corner(
                    mesh_node,
                    all_internal,
                    vertex_labels,
                    cavity,
                    analytic,
                    vertices,
                    loop,
                )
                if target is None:
                    continue
                seg = _segment_between_nodes(
                    mesh_node,
                    target,
                    analytic,
                    cavity,
                    vertices,
                    patch_surface_fits,
                )
                if seg is not None:
                    target.incident_pairs.add(tuple(sorted(pair)))
                    key = (pair, int(mesh_node.vertex_id), int(target.vertex_id))
                    if key not in seen:
                        seen.add(key)
                        segments.append(seg)
            continue

        for mesh_node in mesh_on_line:
            mesh_param = _param_on_carrier(mesh_node.position, analytic, cavity.center)
            best: Optional[ArrangementNode] = None
            best_gap = float("inf")
            for internal in internal_on_line:
                gap = abs(
                    _param_on_carrier(internal.position, analytic, cavity.center)
                    - mesh_param
                )
                if gap < best_gap:
                    best_gap = gap
                    best = internal
            if best is None:
                target = _best_internal_for_mesh_corner(
                    mesh_node,
                    all_internal,
                    vertex_labels,
                    cavity,
                    analytic,
                    vertices,
                    loop,
                )
                if target is None:
                    continue
                key = (pair, int(mesh_node.vertex_id), int(target.vertex_id))
                if key in seen:
                    continue
                seg = _segment_between_nodes(
                    mesh_node,
                    target,
                    analytic,
                    cavity,
                    vertices,
                    patch_surface_fits,
                )
                if seg is not None:
                    target.incident_pairs.add(tuple(sorted(pair)))
                    seen.add(key)
                    segments.append(seg)
                continue
            key = (pair, int(mesh_node.vertex_id), int(best.node_id))
            if key in seen:
                continue
            seg = _segment_between_nodes(
                mesh_node,
                best,
                analytic,
                cavity,
                vertices,
                patch_surface_fits,
            )
            if seg is not None:
                best.incident_pairs.add(tuple(sorted(pair)))
                seen.add(key)
                segments.append(seg)

        if len(internal_on_line) >= 2:
            ordered_internal = _sorted_carrier_nodes_toward_center(
                internal_on_line,
                analytic,
                cavity,
            )
            for i in range(len(ordered_internal) - 1):
                a = ordered_internal[i]
                b = ordered_internal[i + 1]
                pa = _param_on_carrier(a.position, analytic, cavity.center)
                pb = _param_on_carrier(b.position, analytic, cavity.center)
                blocked = False
                for mesh_node in mesh_on_line:
                    tm = _param_on_carrier(mesh_node.position, analytic, cavity.center)
                    if min(pa, pb) < tm < max(pa, pb):
                        blocked = True
                        break
                if blocked:
                    continue
                key = (
                    pair,
                    min(int(a.node_id), int(b.node_id)),
                    max(int(a.node_id), int(b.node_id)),
                )
                if key in seen:
                    continue
                seg = _segment_between_nodes(
                    a,
                    b,
                    analytic,
                    cavity,
                    vertices,
                    patch_surface_fits,
                )
                if seg is not None:
                    seen.add(key)
                    segments.append(seg)

    internal_nodes = [n for n in nodes if int(n.vertex_id) < 0]
    for i, node_a in enumerate(internal_nodes):
        labels_a = _node_patch_labels_from_pairs(node_a)
        for node_b in internal_nodes[i + 1 :]:
            labels_b = _node_patch_labels_from_pairs(node_b)
            shared_labels = sorted(labels_a & labels_b)
            proof: Dict[str, object] = {
                "node_ids": (int(node_a.node_id), int(node_b.node_id)),
                "shared_labels": [int(x) for x in shared_labels],
                "accepted": False,
            }
            if len(shared_labels) != 2:
                proof["reject_reason"] = "not_two_shared_labels"
                bridge_proofs.append(proof)
                continue
            pair = tuple(sorted((int(shared_labels[0]), int(shared_labels[1]))))
            proof["pair"] = pair
            analytic = carriers.get(pair)
            if analytic is None:
                proof["reject_reason"] = "missing_carrier"
                bridge_proofs.append(proof)
                continue
            key = (pair, min(int(node_a.node_id), int(node_b.node_id)), max(int(node_a.node_id), int(node_b.node_id)))
            if key in seen:
                proof["reject_reason"] = "already_present"
                bridge_proofs.append(proof)
                continue
            mid = 0.5 * (
                np.asarray(node_a.position, dtype=np.float64).reshape(3)
                + np.asarray(node_b.position, dtype=np.float64).reshape(3)
            )
            if not cavity.contains(vertices, mid):
                proof["reject_reason"] = "midpoint_outside_cavity"
                bridge_proofs.append(proof)
                continue
            seg = _segment_between_nodes(
                node_a,
                node_b,
                analytic,
                cavity,
                vertices,
                patch_surface_fits,
            )
            if seg is None:
                proof["reject_reason"] = "segment_recovery_failed"
                bridge_proofs.append(proof)
                continue
            edge_id = len(segments)
            seen.add(key)
            segments.append(seg)
            proof["accepted"] = True
            proof["optional_edge"] = True
            proof["edge_id"] = int(edge_id)
            proof["clip_confidence"] = str(seg.clip_confidence)
            bridge_proofs.append(proof)

    return segments, bridge_proofs


def _arrangement_node_key(node: ArrangementNode) -> str:
    if int(node.vertex_id) >= 0:
        return f"m:{int(node.vertex_id)}"
    return f"v:{int(node.node_id)}"


def _nearest_arrangement_node_key(
    point: np.ndarray,
    nodes: Sequence[ArrangementNode],
    cavity: HoleCavity,
    *,
    virtual_only: bool = False,
) -> Optional[str]:
    p = np.asarray(point, dtype=np.float64).reshape(3)
    tol = max(_node_merge_tol(cavity), 2.5 * float(cavity.mean_edge_length))
    best_key: Optional[str] = None
    best_dist = float(tol)
    for node in nodes:
        if virtual_only and int(node.vertex_id) >= 0:
            continue
        d = float(np.linalg.norm(p - np.asarray(node.position, dtype=np.float64)))
        if d <= best_dist:
            best_dist = d
            best_key = _arrangement_node_key(node)
    return best_key


def _segment_endpoint_keys(
    seg: BoundedCurveSegment,
    nodes: Sequence[ArrangementNode],
    cavity: HoleCavity,
) -> Optional[Tuple[str, str]]:
    pts = np.asarray(seg.curve_points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return None
    e0, e1 = (int(seg.boundary_vertex_indices[0]), int(seg.boundary_vertex_indices[1]))
    k0 = (
        f"m:{e0}"
        if e0 >= 0
        else _nearest_arrangement_node_key(pts[0], nodes, cavity, virtual_only=True)
    )
    k1 = (
        f"m:{e1}"
        if e1 >= 0
        else _nearest_arrangement_node_key(pts[-1], nodes, cavity, virtual_only=True)
    )
    if k0 is None or k1 is None or k0 == k1:
        return None
    return str(k0), str(k1)


def _candidate_edge_records(
    segments: Sequence[BoundedCurveSegment],
    nodes: Sequence[ArrangementNode],
    cavity: HoleCavity,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for idx, seg in enumerate(segments):
        keys = _segment_endpoint_keys(seg, nodes, cavity)
        if keys is None:
            continue
        pair = tuple(sorted(seg.analytic.patch_pair))
        records.append(
            {
                "edge_id": int(idx),
                "kind": str(seg.clip_confidence),
                "pair": (int(pair[0]), int(pair[1])),
                "endpoints": keys,
                "points": np.asarray(seg.curve_points, dtype=np.float64).tolist(),
            }
        )
    return records


def _candidate_node_records(nodes: Sequence[ArrangementNode]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for node in nodes:
        out.append(
            {
                "key": _arrangement_node_key(node),
                "vertex_id": int(node.vertex_id),
                "node_id": int(node.node_id),
                "position": np.asarray(node.position, dtype=np.float64).tolist(),
                "incident_pairs": [
                    (int(a), int(b)) for a, b in sorted(node.incident_pairs)
                ],
            }
        )
    return out


def _label_boundary_edges(
    label: int,
    vertices: np.ndarray,
    loop: Sequence[int],
    vertex_labels: Mapping[int, Sequence[int]],
    arc_corner_hints: Optional[Mapping[Tuple[int, int], Sequence[int]]] = None,
) -> List[Tuple[str, str, np.ndarray]]:
    verts = np.asarray(vertices, dtype=np.float64)
    loop_list = [int(v) for v in loop]

    def _arc_points_between(a: int, b: int) -> np.ndarray:
        if a not in loop_list or b not in loop_list:
            return verts[np.asarray([a, b], dtype=np.int64)]
        ia = loop_list.index(a)
        ib = loop_list.index(b)
        n = len(loop_list)
        path_fwd: List[int] = []
        cur = ia
        while True:
            path_fwd.append(loop_list[cur])
            if cur == ib:
                break
            cur = (cur + 1) % n
        path_bwd: List[int] = []
        cur = ia
        while True:
            path_bwd.append(loop_list[cur])
            if cur == ib:
                break
            cur = (cur - 1) % n

        def _support_score(path: Sequence[int]) -> Tuple[int, int]:
            labels_on_path = [
                int(label) in {int(x) for x in vertex_labels.get(int(v), ())}
                for v in path
            ]
            return int(sum(labels_on_path)), -int(len(path))

        chosen = path_fwd if _support_score(path_fwd) >= _support_score(path_bwd) else path_bwd
        return verts[np.asarray(chosen, dtype=np.int64)]

    if arc_corner_hints:
        hinted: List[int] = []
        loop_vertices = _loop_set(loop)
        for pair, hint_vertices in arc_corner_hints.items():
            if int(label) not in {int(pair[0]), int(pair[1])}:
                continue
            for vertex in hint_vertices:
                vi = int(vertex)
                if vi in loop_vertices and vi not in hinted:
                    hinted.append(vi)
        if len(hinted) == 2:
            a, b = int(hinted[0]), int(hinted[1])
            return [(f"m:{a}", f"m:{b}", _arc_points_between(a, b))]

    n = len(loop)
    if n < 2:
        return []
    present = [
        int(label) in {int(x) for x in vertex_labels.get(int(v), ())}
        for v in loop
    ]
    if not any(present):
        return []
    if all(present):
        return []
    starts: List[int] = []
    for i in range(n):
        if present[i] and not present[(i - 1) % n]:
            starts.append(i)
    edges: List[Tuple[str, str]] = []
    for start in starts:
        end = start
        while present[(end + 1) % n]:
            end = (end + 1) % n
            if end == start:
                break
        a = int(loop[start])
        b = int(loop[end])
        if a != b:
            edges.append((f"m:{a}", f"m:{b}", _arc_points_between(a, b)))
    return edges


def _label_boundary_edge_alternatives(
    label: int,
    vertices: np.ndarray,
    loop: Sequence[int],
    vertex_labels: Mapping[int, Sequence[int]],
    arc_corner_hints: Optional[Mapping[Tuple[int, int], Sequence[int]]] = None,
) -> List[List[Tuple[str, str, np.ndarray]]]:
    verts = np.asarray(vertices, dtype=np.float64)
    loop_list = [int(v) for v in loop]
    if arc_corner_hints:
        hinted: List[int] = []
        loop_vertices = _loop_set(loop)
        for pair, hint_vertices in arc_corner_hints.items():
            if int(label) not in {int(pair[0]), int(pair[1])}:
                continue
            for vertex in hint_vertices:
                vi = int(vertex)
                if vi in loop_vertices and vi not in hinted:
                    hinted.append(vi)
        if len(hinted) == 2 and hinted[0] in loop_list and hinted[1] in loop_list:
            a, b = int(hinted[0]), int(hinted[1])
            ia, ib = loop_list.index(a), loop_list.index(b)
            n = len(loop_list)
            paths: List[List[int]] = []
            cur = ia
            fwd: List[int] = []
            while True:
                fwd.append(loop_list[cur])
                if cur == ib:
                    break
                cur = (cur + 1) % n
            cur = ia
            bwd: List[int] = []
            while True:
                bwd.append(loop_list[cur])
                if cur == ib:
                    break
                cur = (cur - 1) % n
            paths.extend([fwd, bwd])
            out: List[List[Tuple[str, str, np.ndarray]]] = []
            seen: Set[Tuple[int, ...]] = set()
            for path in paths:
                key = tuple(path)
                if key in seen:
                    continue
                seen.add(key)
                out.append([(f"m:{a}", f"m:{b}", verts[np.asarray(path, dtype=np.int64)])])
            return out
    base = _label_boundary_edges(
        label,
        vertices,
        loop,
        vertex_labels,
        arc_corner_hints=arc_corner_hints,
    )
    if not base:
        return []
    if len(base) == 1:
        a_key, b_key, _pts = base[0]
        if a_key.startswith("m:") and b_key.startswith("m:"):
            a = int(a_key.split(":", 1)[1])
            b = int(b_key.split(":", 1)[1])
            if a in loop_list and b in loop_list:
                ia, ib = loop_list.index(a), loop_list.index(b)
                n = len(loop_list)
                paths: List[List[int]] = []
                cur = ia
                fwd: List[int] = []
                while True:
                    fwd.append(loop_list[cur])
                    if cur == ib:
                        break
                    cur = (cur + 1) % n
                cur = ia
                bwd: List[int] = []
                while True:
                    bwd.append(loop_list[cur])
                    if cur == ib:
                        break
                    cur = (cur - 1) % n
                paths.extend([fwd, bwd])
                out: List[List[Tuple[str, str, np.ndarray]]] = []
                seen: Set[Tuple[int, ...]] = set()
                for path in paths:
                    key = tuple(path)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(
                        [
                            (
                                f"m:{a}",
                                f"m:{b}",
                                verts[np.asarray(path, dtype=np.int64)],
                            )
                        ]
                    )
                if out:
                    return out
    return [base]


def _graph_components(adjacency: Mapping[str, Set[str]]) -> List[Set[str]]:
    unseen = set(adjacency)
    components: List[Set[str]] = []
    while unseen:
        root = unseen.pop()
        comp = {root}
        stack = [root]
        while stack:
            cur = stack.pop()
            for nxt in adjacency.get(cur, set()):
                if nxt in unseen:
                    unseen.remove(nxt)
                    comp.add(nxt)
                    stack.append(nxt)
        components.append(comp)
    return components


def _safe_unit(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if n <= 1e-15:
        return np.zeros(3, dtype=np.float64)
    return v / n


def _surface_projection_frame(
    fit: SurfaceFit,
    fallback_points: Sequence[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(fallback_points, dtype=np.float64).reshape(-1, 3)
    origin = np.mean(pts, axis=0) if pts.size else np.zeros(3, dtype=np.float64)
    params = fit.surface_params or {}
    normal: Optional[np.ndarray] = None
    stype = str(fit.surface_type)
    if stype == "plane" and "normal" in params:
        normal = _safe_unit(np.asarray(params["normal"], dtype=np.float64))
        if "point" in params:
            origin = np.asarray(params["point"], dtype=np.float64).reshape(3)
    elif stype in {"cylinder", "cone"} and "axis" in params:
        normal = _safe_unit(np.asarray(params["axis"], dtype=np.float64))
        if "point" in params:
            origin = np.asarray(params["point"], dtype=np.float64).reshape(3)
    if normal is None or float(np.linalg.norm(normal)) <= 1e-12:
        centered = pts - origin.reshape(1, 3)
        if centered.shape[0] >= 3:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            normal = _safe_unit(vh[-1])
        else:
            normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    axis_hint = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(axis_hint, normal))) > 0.9:
        axis_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u_axis = _safe_unit(np.cross(normal, axis_hint))
    if float(np.linalg.norm(u_axis)) <= 1e-12:
        u_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    v_axis = _safe_unit(np.cross(normal, u_axis))
    return origin, u_axis, v_axis


def _project_polyline_2d(
    points: np.ndarray,
    frame: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    origin, u_axis, v_axis = frame
    rel = pts - np.asarray(origin, dtype=np.float64).reshape(1, 3)
    return np.column_stack([rel @ u_axis, rel @ v_axis])


def _project_surface_polyline_2d(
    fit: SurfaceFit,
    points: np.ndarray,
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    params = fit.surface_params or {}
    stype = str(fit.surface_type)
    if stype in {"cylinder", "cone"} and "axis" in params and "point" in params:
        origin = np.asarray(params["point"], dtype=np.float64).reshape(3)
        axis = _safe_unit(np.asarray(params["axis"], dtype=np.float64).reshape(3))
        if float(np.linalg.norm(axis)) > 1e-12:
            hint = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(float(np.dot(hint, axis))) > 0.9:
                hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            u_axis = _safe_unit(np.cross(axis, hint))
            v_axis = _safe_unit(np.cross(axis, u_axis))
            rel = pts - origin.reshape(1, 3)
            z = rel @ axis
            radial = rel - z.reshape(-1, 1) * axis.reshape(1, 3)
            theta = np.unwrap(np.arctan2(radial @ v_axis, radial @ u_axis))
            radius = float(params.get("radius", 0.0) or 0.0)
            if radius <= 1e-12:
                norms = np.linalg.norm(radial, axis=1)
                radius = float(np.mean(norms)) if norms.size else 1.0
            return np.column_stack([radius * theta, z])
    frame = _surface_projection_frame(fit, pts)
    return _project_polyline_2d(pts, frame)


def _orient_polyline_for_edge(
    points: np.ndarray,
    a: str,
    b: str,
    node_positions: Mapping[str, np.ndarray],
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 2 or a not in node_positions or b not in node_positions:
        return pts
    pa = np.asarray(node_positions[a], dtype=np.float64).reshape(3)
    pb = np.asarray(node_positions[b], dtype=np.float64).reshape(3)
    direct = float(np.linalg.norm(pts[0] - pa) + np.linalg.norm(pts[-1] - pb))
    flipped = float(np.linalg.norm(pts[0] - pb) + np.linalg.norm(pts[-1] - pa))
    return pts if direct <= flipped else pts[::-1]


def _ordered_cycle_edges(
    adjacency: Mapping[str, Set[str]],
    edge_by_nodes: Mapping[Tuple[str, str], Tuple[str, str, str, np.ndarray]],
) -> Optional[List[Tuple[str, str, str, np.ndarray]]]:
    if not adjacency or any(len(neigh) != 2 for neigh in adjacency.values()):
        return None
    start = sorted(adjacency)[0]
    neighbors = sorted(adjacency[start])
    prev: Optional[str] = None
    cur = start
    out: List[Tuple[str, str, str, np.ndarray]] = []
    visited_edges: Set[Tuple[str, str]] = set()
    while True:
        next_candidates = sorted(n for n in adjacency[cur] if n != prev)
        if not next_candidates:
            return None
        nxt = next_candidates[0]
        ekey = tuple(sorted((cur, nxt)))
        if ekey in visited_edges:
            if nxt == start and len(visited_edges) == len(edge_by_nodes):
                break
            return None
        record = edge_by_nodes.get(ekey)
        if record is None:
            return None
        _ra, _rb, kind, pts = record
        out.append((cur, nxt, kind, pts))
        visited_edges.add(ekey)
        prev, cur = cur, nxt
        if cur == start:
            break
        if len(visited_edges) > len(edge_by_nodes):
            return None
    if len(visited_edges) != len(edge_by_nodes):
        return None
    return out


def _signed_area_2d(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _orient2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _segments_cross_strict_2d(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    *,
    tol: float,
) -> bool:
    if min(
        float(np.linalg.norm(a - c)),
        float(np.linalg.norm(a - d)),
        float(np.linalg.norm(b - c)),
        float(np.linalg.norm(b - d)),
    ) <= tol:
        return False
    o1 = _orient2d(a, b, c)
    o2 = _orient2d(a, b, d)
    o3 = _orient2d(c, d, a)
    o4 = _orient2d(c, d, b)
    return (o1 * o2 < -tol * tol) and (o3 * o4 < -tol * tol)


def _polyline_self_intersections_2d(points: np.ndarray, *, tol: float) -> int:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 4:
        return 0
    count = 0
    nseg = pts.shape[0] - 1
    for i in range(nseg):
        for j in range(i + 1, nseg):
            if abs(i - j) <= 1:
                continue
            if i == 0 and j == nseg - 1:
                continue
            if _segments_cross_strict_2d(
                pts[i],
                pts[i + 1],
                pts[j],
                pts[j + 1],
                tol=tol,
            ):
                count += 1
    return count


def _rdp_simplify_2d(points: np.ndarray, tol: float) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] <= 2:
        return pts

    def _point_line_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-30:
            return float(np.linalg.norm(p - a))
        t = max(0.0, min(1.0, float(np.dot(p - a, ab) / denom)))
        proj = a + t * ab
        return float(np.linalg.norm(p - proj))

    a = pts[0]
    b = pts[-1]
    if pts.shape[0] <= 2:
        return pts
    dists = [_point_line_distance(pts[i], a, b) for i in range(1, pts.shape[0] - 1)]
    if not dists:
        return pts
    max_idx = int(np.argmax(dists)) + 1
    if float(dists[max_idx - 1]) <= float(tol):
        return np.vstack([a, b])
    left = _rdp_simplify_2d(pts[: max_idx + 1], tol)
    right = _rdp_simplify_2d(pts[max_idx:], tol)
    return np.vstack([left[:-1], right])


def _validate_surface_cells(
    labels: Sequence[int],
    vertices: np.ndarray,
    loop: Sequence[int],
    vertex_labels: Mapping[int, Sequence[int]],
    candidate_edges: Sequence[Mapping[str, object]],
    candidate_nodes: Sequence[Mapping[str, object]],
    patch_surface_fits: Mapping[int, SurfaceFit],
    *,
    arc_corner_hints: Optional[Mapping[Tuple[int, int], Sequence[int]]] = None,
) -> Tuple[bool, List[Dict[str, object]]]:
    validations: List[Dict[str, object]] = []
    all_valid = True
    node_positions: Dict[str, np.ndarray] = {
        str(node["key"]): np.asarray(node["position"], dtype=np.float64).reshape(3)
        for node in candidate_nodes
        if "key" in node and "position" in node
    }

    def _evaluate_label_edges(
        label: int,
        graph_edges: List[Tuple[str, str, str, np.ndarray]],
    ) -> Dict[str, object]:
        adjacency: Dict[str, Set[str]] = {}
        edge_by_nodes: Dict[Tuple[str, str], Tuple[str, str, str, np.ndarray]] = {}
        degree_count: Dict[str, int] = {}
        for a, b, kind, pts in graph_edges:
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)
            degree_count[a] = int(degree_count.get(a, 0)) + 1
            degree_count[b] = int(degree_count.get(b, 0)) + 1
            edge_by_nodes[tuple(sorted((a, b)))] = (a, b, kind, pts)
        degree = {node: int(degree_count.get(node, 0)) for node in adjacency}
        components = _graph_components(adjacency) if adjacency else []
        dangling = sorted(node for node, deg in degree.items() if int(deg) != 2)
        boundary_count = sum(1 for _a, _b, kind, _pts in graph_edges if kind == "boundary")
        carrier_count = sum(1 for _a, _b, kind, _pts in graph_edges if kind == "carrier")
        cycle_edges = _ordered_cycle_edges(adjacency, edge_by_nodes)
        if (
            cycle_edges is None
            and len(graph_edges) == 2
            and boundary_count == 1
            and carrier_count == 1
            and tuple(sorted((graph_edges[0][0], graph_edges[0][1])))
            == tuple(sorted((graph_edges[1][0], graph_edges[1][1])))
        ):
            boundary_edge = next(edge for edge in graph_edges if edge[2] == "boundary")
            carrier_edge = next(edge for edge in graph_edges if edge[2] == "carrier")
            a, b, kind_b, pts_b = boundary_edge
            _ca, _cb, kind_c, pts_c = carrier_edge
            cycle_edges = [(a, b, kind_b, pts_b), (b, a, kind_c, pts_c)]
        loop_points_3d: List[np.ndarray] = []
        loop_segment_records: List[Dict[str, object]] = []
        if cycle_edges is not None:
            for a, b, _kind, pts in cycle_edges:
                oriented = _orient_polyline_for_edge(pts, a, b, node_positions)
                edge_record = {
                    "endpoints": (str(a), str(b)),
                    "kind": str(_kind),
                }
                for _seg_idx in range(max(0, int(oriented.shape[0]) - 1)):
                    loop_segment_records.append(dict(edge_record))
                if not loop_points_3d:
                    loop_points_3d.extend([p.copy() for p in oriented])
                else:
                    loop_points_3d.extend([p.copy() for p in oriented[1:]])
            close_tol = max(
                _node_merge_tol(
                    build_hole_cavity(
                        vertices,
                        loop,
                        np.mean(vertices[np.asarray(loop, dtype=np.int64)], axis=0),
                    )
                ),
                1e-9,
            )
            if loop_points_3d and float(
                np.linalg.norm(loop_points_3d[0] - loop_points_3d[-1])
            ) > close_tol:
                loop_points_3d.append(loop_points_3d[0].copy())
                loop_segment_records.append(
                    {"endpoints": ("closure", "closure"), "kind": "closure"}
                )
        area_abs = 0.0
        self_crossings = 0
        simplified_self_crossings = 0
        effective_self_crossings = 0
        simplification_sweep: List[Dict[str, object]] = []
        boundary_shortcut_proof: Dict[str, object] = {"accepted": False}
        self_crossing_pairs: List[Dict[str, object]] = []
        if loop_points_3d:
            loop2d = _project_surface_polyline_2d(
                patch_surface_fits[int(label)],
                np.asarray(loop_points_3d),
            )
            area_abs = abs(_signed_area_2d(loop2d))
            diag2 = max(float(np.linalg.norm(np.ptp(loop2d, axis=0))), 1e-12)
            self_crossings = _polyline_self_intersections_2d(
                loop2d,
                tol=max(1e-5 * diag2, 1e-10),
            )
            simplified = _rdp_simplify_2d(loop2d, tol=max(0.015 * diag2, 1e-10))
            simplified_self_crossings = _polyline_self_intersections_2d(
                simplified,
                tol=max(1e-5 * diag2, 1e-10),
            )
            for factor in (0.015, 0.03, 0.06, 0.1):
                simp = _rdp_simplify_2d(loop2d, tol=max(float(factor) * diag2, 1e-10))
                simplification_sweep.append(
                    {
                        "factor": float(factor),
                        "n_points": int(simp.shape[0]),
                        "self_intersections": int(
                            _polyline_self_intersections_2d(
                                simp,
                                tol=max(1e-5 * diag2, 1e-10),
                            )
                        ),
                    }
                )
            nseg = loop2d.shape[0] - 1
            tol = max(1e-5 * diag2, 1e-10)
            for i in range(nseg):
                for j in range(i + 1, nseg):
                    if abs(i - j) <= 1:
                        continue
                    if i == 0 and j == nseg - 1:
                        continue
                    if _segments_cross_strict_2d(
                        loop2d[i],
                        loop2d[i + 1],
                        loop2d[j],
                        loop2d[j + 1],
                        tol=tol,
                    ):
                        self_crossing_pairs.append(
                            {
                                "segments": (int(i), int(j)),
                                "edges": (
                                    loop_segment_records[i]
                                    if i < len(loop_segment_records)
                                    else {},
                                    loop_segment_records[j]
                                    if j < len(loop_segment_records)
                                    else {},
                                ),
                            }
                        )
            boundary_carrier_only = bool(self_crossing_pairs) and all(
                {
                    str(pair.get("edges", ({}, {}))[0].get("kind", "")),
                    str(pair.get("edges", ({}, {}))[1].get("kind", "")),
                }
                == {"boundary", "carrier"}
                for pair in self_crossing_pairs
            )
            effective_self_crossings = (
                int(simplified_self_crossings)
                if boundary_carrier_only
                else int(self_crossings)
            )
            if boundary_carrier_only and effective_self_crossings > 0:
                for item in simplification_sweep:
                    if (
                        float(item.get("factor", 1.0)) <= 0.1
                        and int(item.get("self_intersections", 999999)) == 0
                    ):
                        boundary_shortcut_proof = {
                            "accepted": False,
                            "diagnostic_only": True,
                            "factor": float(item.get("factor", 0.0)),
                            "n_points": int(item.get("n_points", 0)),
                            "reason": "boundary_carrier_crossing_removed_by_coarse_parameter_shortcut",
                        }
                        break
        area_tol = max(1e-12, 1e-8 * float(len(loop) or 1))
        failure_reasons: List[str] = []
        if not graph_edges:
            failure_reasons.append("no_edges")
        if boundary_count <= 0:
            failure_reasons.append("missing_boundary")
        if carrier_count <= 0:
            failure_reasons.append("missing_carrier")
        if dangling:
            failure_reasons.append("dangling_nodes")
        if len(components) != 1:
            failure_reasons.append("disconnected_cell")
        if cycle_edges is None:
            failure_reasons.append("cycle_not_reconstructed")
        if loop_points_3d and area_abs <= area_tol:
            failure_reasons.append("degenerate_area")
        if effective_self_crossings > 0:
            failure_reasons.append("parameter_self_intersection")
        label_valid = (
            bool(graph_edges)
            and boundary_count > 0
            and carrier_count > 0
            and not dangling
            and len(components) == 1
            and cycle_edges is not None
            and area_abs > area_tol
            and effective_self_crossings == 0
        )
        cycle_node_keys: List[str] = []
        if cycle_edges is not None:
            seen: Set[str] = set()
            for a, b, _kind, _pts in cycle_edges:
                for key in (str(a), str(b)):
                    if key not in seen:
                        seen.add(key)
                        cycle_node_keys.append(key)
        proof_report = {
            "cycle_node_keys": cycle_node_keys,
            "n_cycle_edges": int(len(cycle_edges) if cycle_edges is not None else 0),
            "boundary_alternative": None,
            "carrier_edge_endpoints": [
                {"endpoints": (a, b), "kind": kind}
                for a, b, kind, _pts in graph_edges
                if kind == "carrier"
            ],
            "failure_reasons": list(failure_reasons),
        }
        return {
            "label": int(label),
            "valid": bool(label_valid),
            "n_boundary_edges": int(boundary_count),
            "n_carrier_edges": int(carrier_count),
            "n_components": int(len(components)),
            "dangling_nodes": dangling,
            "disconnected_cells": max(0, int(len(components)) - 1),
            "parameter_loop_area": float(area_abs),
            "parameter_self_intersections": int(self_crossings),
            "parameter_simplified_self_intersections": int(simplified_self_crossings),
            "parameter_effective_self_intersections": int(effective_self_crossings),
            "parameter_simplification_sweep": simplification_sweep,
            "boundary_shortcut_proof": boundary_shortcut_proof,
            "parameter_self_crossing_pairs": self_crossing_pairs,
            "cycle_reconstructed": bool(cycle_edges is not None),
            "failure_reasons": failure_reasons,
            "proof_report": proof_report,
            "edges": [
                {"endpoints": (a, b), "kind": kind}
                for a, b, kind, _pts in graph_edges
            ],
        }

    for label in sorted(int(x) for x in labels):
        carrier_edges: List[Tuple[str, str, str, np.ndarray]] = []
        for edge in candidate_edges:
            pair = tuple(int(x) for x in edge.get("pair", ()))
            if len(pair) == 2 and int(label) in pair:
                endpoints = tuple(str(x) for x in edge.get("endpoints", ()))
                pts = np.asarray(edge.get("points", []), dtype=np.float64)
                if len(endpoints) == 2 and pts.ndim == 2 and pts.shape[0] >= 2:
                    carrier_edges.append((endpoints[0], endpoints[1], "carrier", pts))

        boundary_alternatives = _label_boundary_edge_alternatives(
            label,
            vertices,
            loop,
            vertex_labels,
            arc_corner_hints=arc_corner_hints,
        )
        if not boundary_alternatives:
            boundary_alternatives = [[]]
        evaluated: List[Dict[str, object]] = []
        for alt_idx, boundary_edges in enumerate(boundary_alternatives):
            result = _evaluate_label_edges(
                label,
                [
                    *[(a, b, "boundary", pts) for a, b, pts in boundary_edges],
                    *carrier_edges,
                ],
            )
            result["boundary_alternative"] = int(alt_idx)
            evaluated.append(result)
        valid_results = [item for item in evaluated if bool(item.get("valid", False))]
        if valid_results:
            chosen = max(valid_results, key=lambda item: float(item.get("parameter_loop_area", 0.0)))
        else:
            chosen = min(
                evaluated,
                key=lambda item: (
                    int(item.get("parameter_self_intersections", 999999)),
                    len(item.get("dangling_nodes", [])),
                    -float(item.get("parameter_loop_area", 0.0)),
                ),
            )
        if not bool(chosen.get("valid", False)):
            all_valid = False
        proof = dict(chosen.get("proof_report", {}) or {})
        proof["boundary_alternative"] = int(chosen.get("boundary_alternative", 0))
        chosen["proof_report"] = proof
        chosen["boundary_alternative_count"] = int(len(boundary_alternatives))
        chosen["boundary_alternative_evaluations"] = [
            {
                "alternative": int(item.get("boundary_alternative", idx)),
                "valid": bool(item.get("valid", False)),
                "self_intersections": int(
                    item.get("parameter_self_intersections", 0)
                ),
                "dangling_count": len(item.get("dangling_nodes", []) or []),
                "area": float(item.get("parameter_loop_area", 0.0)),
            }
            for idx, item in enumerate(evaluated)
        ]
        validations.append(chosen)
    return bool(all_valid), validations


def _validate_segment_set(
    labels: Sequence[int],
    vertices: np.ndarray,
    loop: Sequence[int],
    vertex_labels: Mapping[int, Sequence[int]],
    segments: Sequence[BoundedCurveSegment],
    nodes: Sequence[ArrangementNode],
    cavity: HoleCavity,
    candidate_nodes: Sequence[Mapping[str, object]],
    patch_surface_fits: Mapping[int, SurfaceFit],
    *,
    arc_corner_hints: Optional[Mapping[Tuple[int, int], Sequence[int]]] = None,
) -> Tuple[bool, List[Dict[str, object]], List[Dict[str, object]]]:
    candidate_edges = _candidate_edge_records(segments, nodes, cavity)
    cells_valid, surface_validation = _validate_surface_cells(
        labels,
        vertices,
        loop,
        vertex_labels,
        candidate_edges,
        candidate_nodes,
        patch_surface_fits,
        arc_corner_hints=arc_corner_hints,
    )
    return bool(cells_valid), surface_validation, candidate_edges


def _surface_cell_score(
    valid: bool,
    surface_validation: Sequence[Mapping[str, object]],
    *,
    n_edges: int,
) -> Tuple[int, int, int, int, int, int, int, int, int]:
    """Lower is better: reject dense candidate graphs unless they prove valid cells."""
    invalid_labels = 0
    missing_carrier = 0
    missing_boundary = 0
    missing_cycle = 0
    self_crossings = 0
    dangling = 0
    disconnected = 0
    for item in surface_validation:
        if not bool(item.get("valid", False)):
            invalid_labels += 1
        if int(item.get("n_carrier_edges", 0)) <= 0:
            missing_carrier += 1
        if int(item.get("n_boundary_edges", 0)) <= 0:
            missing_boundary += 1
        if not bool(item.get("cycle_reconstructed", False)):
            missing_cycle += 1
        self_crossings += int(item.get("parameter_self_intersections", 0))
        dangling += len(item.get("dangling_nodes", []) or [])
        disconnected += int(item.get("disconnected_cells", 0))
    return (
        0 if bool(valid) else 1,
        int(invalid_labels),
        int(missing_carrier),
        int(missing_boundary),
        int(missing_cycle),
        int(self_crossings),
        int(dangling),
        int(disconnected),
        int(n_edges),
    )


def _select_greedy_cell_complex_subset(
    labels: Sequence[int],
    vertices: np.ndarray,
    loop: Sequence[int],
    vertex_labels: Mapping[int, Sequence[int]],
    segments: Sequence[BoundedCurveSegment],
    nodes: Sequence[ArrangementNode],
    cavity: HoleCavity,
    candidate_nodes: Sequence[Mapping[str, object]],
    patch_surface_fits: Mapping[int, SurfaceFit],
    initial_valid: bool,
    initial_validation: Sequence[Mapping[str, object]],
    *,
    arc_corner_hints: Optional[Mapping[Tuple[int, int], Sequence[int]]] = None,
) -> Tuple[
    List[BoundedCurveSegment],
    bool,
    List[Dict[str, object]],
    List[Dict[str, object]],
    Dict[str, object],
]:
    """
    Experimental accepted-cell-complex selection.

    The candidate arrangement may contain analytic intersections that are geometrically
    possible but not needed by any cavity-restricted trimming cell.  Greedily remove
    carrier segments only when the parameter-domain cell proof improves.
    """
    if len(segments) > 18:
        valid, validation, edges = _validate_segment_set(
            labels,
            vertices,
            loop,
            vertex_labels,
            segments,
            nodes,
            cavity,
            candidate_nodes,
            patch_surface_fits,
            arc_corner_hints=arc_corner_hints,
        )
        return (
            list(segments),
            bool(valid),
            validation,
            edges,
            {
                "selection_mode": "minimal_cell_complex_greedy_skipped",
                "search_attempts": 1,
                "search_skipped_reason": "too_many_candidate_edges",
                "n_candidate_edges": int(len(segments)),
            },
        )

    selected_ids: Set[int] = set(range(len(segments)))
    current_segments = [segments[i] for i in sorted(selected_ids)]
    current_valid = bool(initial_valid)
    current_validation = [dict(x) for x in initial_validation]
    current_score = _surface_cell_score(
        current_valid,
        current_validation,
        n_edges=len(current_segments),
    )
    attempts = 1
    removed: List[int] = []
    max_attempts = min(32, 2 * max(1, len(segments)))

    while attempts < max_attempts and len(selected_ids) > 1:
        best_choice: Optional[
            Tuple[
                Tuple[int, int, int, int, int, int, int, int, int],
                int,
                bool,
                List[Dict[str, object]],
                List[Dict[str, object]],
            ]
        ] = None
        for edge_id in sorted(selected_ids):
            trial_ids = selected_ids - {int(edge_id)}
            trial_segments = [segments[i] for i in sorted(trial_ids)]
            attempts += 1
            valid, validation, edges = _validate_segment_set(
                labels,
                vertices,
                loop,
                vertex_labels,
                trial_segments,
                nodes,
                cavity,
                candidate_nodes,
                patch_surface_fits,
                arc_corner_hints=arc_corner_hints,
            )
            score = _surface_cell_score(
                bool(valid),
                validation,
                n_edges=len(trial_segments),
            )
            if score < current_score and (
                best_choice is None or score < best_choice[0]
            ):
                best_choice = (
                    score,
                    int(edge_id),
                    bool(valid),
                    validation,
                    edges,
                )
            if attempts >= max_attempts:
                break
        if best_choice is None:
            break
        current_score, edge_id, current_valid, current_validation, current_edges = best_choice
        selected_ids.remove(int(edge_id))
        removed.append(int(edge_id))
        current_segments = [segments[i] for i in sorted(selected_ids)]
        if current_valid:
            diagnostics = {
                "selection_mode": "minimal_cell_complex_greedy",
                "search_attempts": int(attempts),
                "removed_edge_ids": sorted(int(i) for i in removed),
                "selected_edge_ids": sorted(int(i) for i in selected_ids),
                "final_score": list(current_score),
            }
            return (
                current_segments,
                True,
                current_validation,
                current_edges,
                diagnostics,
            )

    final_valid, final_validation, final_edges = _validate_segment_set(
        labels,
        vertices,
        loop,
        vertex_labels,
        current_segments,
        nodes,
        cavity,
        candidate_nodes,
        patch_surface_fits,
        arc_corner_hints=arc_corner_hints,
    )
    final_score = _surface_cell_score(
        bool(final_valid),
        final_validation,
        n_edges=len(current_segments),
    )
    diagnostics = {
        "selection_mode": (
            "minimal_cell_complex_greedy"
            if bool(final_valid)
            else "minimal_cell_complex_greedy_incomplete"
        ),
        "search_attempts": int(attempts),
        "removed_edge_ids": sorted(int(i) for i in removed),
        "selected_edge_ids": sorted(int(i) for i in selected_ids),
        "final_score": list(final_score),
    }
    return current_segments, bool(final_valid), final_validation, final_edges, diagnostics


def _optional_edge_ids_from_proofs(
    bridge_proofs: Sequence[Mapping[str, object]],
    n_segments: int,
) -> Set[int]:
    out: Set[int] = set()
    for proof in bridge_proofs:
        if not bool(proof.get("accepted", False)):
            continue
        if not bool(proof.get("optional_edge", False)):
            continue
        if "edge_id" not in proof:
            continue
        edge_id = int(proof["edge_id"])
        if 0 <= edge_id < n_segments:
            out.add(edge_id)
    return out


def _select_minimal_legal_cell_complex(
    labels: Sequence[int],
    vertices: np.ndarray,
    loop: Sequence[int],
    vertex_labels: Mapping[int, Sequence[int]],
    segments: Sequence[BoundedCurveSegment],
    nodes: Sequence[ArrangementNode],
    cavity: HoleCavity,
    candidate_nodes: Sequence[Mapping[str, object]],
    patch_surface_fits: Mapping[int, SurfaceFit],
    bridge_proofs: Sequence[Mapping[str, object]],
    *,
    arc_corner_hints: Optional[Mapping[Tuple[int, int], Sequence[int]]] = None,
) -> Tuple[
    List[BoundedCurveSegment],
    bool,
    List[Dict[str, object]],
    List[Dict[str, object]],
    Dict[str, object],
]:
    """
    Select a minimal edge subset that passes parameter-domain cell proof.

    Required segments are non-optional topology edges; optional virtual bridges
    are searched exhaustively (small n) or removed greedily.
    """
    all_segments = list(segments)
    n = len(all_segments)
    optional_ids = sorted(_optional_edge_ids_from_proofs(bridge_proofs, n))
    optional_set = set(optional_ids)
    required_ids = {i for i in range(n) if i not in optional_set}

    def _evaluate(
        selected_ids: Set[int],
    ) -> Tuple[bool, List[Dict[str, object]], List[Dict[str, object]], Tuple[int, ...]]:
        selected_segments = [all_segments[i] for i in sorted(selected_ids)]
        valid, validation, edges = _validate_segment_set(
            labels,
            vertices,
            loop,
            vertex_labels,
            selected_segments,
            nodes,
            cavity,
            candidate_nodes,
            patch_surface_fits,
            arc_corner_hints=arc_corner_hints,
        )
        score = _surface_cell_score(
            bool(valid),
            validation,
            n_edges=len(selected_segments),
        )
        return bool(valid), validation, edges, score

    diagnostics: Dict[str, object] = {
        "selection_mode": "minimal_legal_cell_complex",
        "optional_edge_ids": [int(i) for i in optional_ids],
        "required_edge_ids": sorted(int(i) for i in required_ids),
        "search_attempts": 0,
    }

    best_valid: Optional[
        Tuple[Set[int], List[Dict[str, object]], List[Dict[str, object]], Tuple[int, ...]]
    ] = None
    attempts = 0
    max_mask_search = 14

    if len(optional_ids) <= max_mask_search:
        diagnostics["search_strategy"] = "optional_exhaustive"
        for mask in range(1 << len(optional_ids)):
            selected_optional = {
                int(optional_ids[i])
                for i in range(len(optional_ids))
                if (int(mask) >> i) & 1
            }
            trial_ids = set(required_ids) | selected_optional
            attempts += 1
            valid, validation, edges, score = _evaluate(trial_ids)
            if not valid:
                continue
            if best_valid is None:
                best_valid = (trial_ids, validation, edges, score)
                continue
            best_ids, _, _, best_score = best_valid
            if score < best_score or (
                score == best_score and len(trial_ids) < len(best_ids)
            ):
                best_valid = (trial_ids, validation, edges, score)
    else:
        diagnostics["search_strategy"] = "greedy_only"

    diagnostics["search_attempts"] = int(attempts)

    if best_valid is not None:
        best_ids, best_validation, best_edges, best_score = best_valid
        selected_segments = [all_segments[i] for i in sorted(best_ids)]
        diagnostics["selected_edge_ids"] = sorted(int(i) for i in best_ids)
        diagnostics["removed_edge_ids"] = sorted(int(i) for i in range(n) if i not in best_ids)
        diagnostics["final_score"] = list(best_score)
        diagnostics["cells_valid"] = True
        return selected_segments, True, best_validation, best_edges, diagnostics

    (
        greedy_segments,
        greedy_valid,
        greedy_validation,
        greedy_edges,
        greedy_diag,
    ) = _select_greedy_cell_complex_subset(
        labels,
        vertices,
        loop,
        vertex_labels,
        all_segments,
        nodes,
        cavity,
        candidate_nodes,
        patch_surface_fits,
        False,
        [],
        arc_corner_hints=arc_corner_hints,
    )
    diagnostics.update(greedy_diag)
    diagnostics["search_attempts"] = int(diagnostics.get("search_attempts", 0)) + int(
        greedy_diag.get("search_attempts", 0)
    )
    diagnostics["selection_mode"] = "minimal_legal_cell_complex_greedy_fallback"
    diagnostics["cells_valid"] = bool(greedy_valid)
    return (
        greedy_segments,
        bool(greedy_valid),
        greedy_validation,
        greedy_edges,
        diagnostics,
    )


def _arrangement_node_lookup(
    nodes: Sequence[ArrangementNode],
) -> Dict[str, ArrangementNode]:
    return {_arrangement_node_key(node): node for node in nodes}


def _segment_incidence_by_node_key(
    segments: Sequence[BoundedCurveSegment],
    nodes: Sequence[ArrangementNode],
    cavity: HoleCavity,
) -> Dict[str, List[Tuple[int, str, Tuple[int, int]]]]:
    incidence: Dict[str, List[Tuple[int, str, Tuple[int, int]]]] = {}
    for idx, seg in enumerate(segments):
        keys = _segment_endpoint_keys(seg, nodes, cavity)
        if keys is None:
            continue
        k0, k1 = keys
        pair = tuple(sorted(int(x) for x in seg.analytic.patch_pair))
        incidence.setdefault(str(k0), []).append((int(idx), str(k1), pair))
        incidence.setdefault(str(k1), []).append((int(idx), str(k0), pair))
    return incidence


def _distinct_incident_pairs(
    incidence: Mapping[str, Sequence[Tuple[int, str, Tuple[int, int]]]],
    node_key: str,
) -> Set[Tuple[int, int]]:
    return {tuple(pair) for _idx, _other, pair in incidence.get(str(node_key), ())}


def _prune_unsupported_virtual_junctions(
    segments: Sequence[BoundedCurveSegment],
    arrangement_nodes: Sequence[ArrangementNode],
    carriers: Mapping[Tuple[int, int], AnalyticCurve],
    cavity: HoleCavity,
    vertices: np.ndarray,
    patch_surface_fits: Mapping[int, SurfaceFit],
) -> Tuple[List[BoundedCurveSegment], List[ArrangementNode], Dict[str, object]]:
    """
    交线裁剪后同步清理虚拟汇交：保留段图上 incident carrier pair < 2 的点
    只是同一条 carrier 上的分段断点，应塌缩进相邻真实汇交。
    """
    current_segments = list(segments)
    current_nodes = list(arrangement_nodes)
    removed_keys: List[str] = []
    merged_records: List[Dict[str, object]] = []
    max_passes = max(8, 2 * len(current_nodes) + 4)

    for _pass in range(max_passes):
        lookup = _arrangement_node_lookup(current_nodes)
        incidence = _segment_incidence_by_node_key(
            current_segments,
            current_nodes,
            cavity,
        )
        collapsed_key: Optional[str] = None

        virtual_keys = sorted(
            key
            for key, node in lookup.items()
            if key.startswith("v:")
            and int(node.vertex_id) < 0
        )
        for node_key in virtual_keys:
            incident = list(incidence.get(node_key, ()))
            distinct_pairs = _distinct_incident_pairs(incidence, node_key)
            if len(distinct_pairs) >= 2:
                continue

            if len(incident) == 2 and len(distinct_pairs) == 1:
                (_i0, other0, pair0), (_i1, other1, pair1) = incident
                if pair0 != pair1:
                    continue
                if str(other0) == str(other1):
                    continue
                node_a = lookup.get(str(other0))
                node_b = lookup.get(str(other1))
                analytic = carriers.get(tuple(pair0))
                if node_a is None or node_b is None or analytic is None:
                    continue
                merged = _segment_between_nodes(
                    node_a,
                    node_b,
                    analytic,
                    cavity,
                    vertices,
                    patch_surface_fits,
                )
                if merged is None:
                    continue
                seg_indices = sorted({int(_i0), int(_i1)}, reverse=True)
                for seg_idx in seg_indices:
                    current_segments.pop(int(seg_idx))
                current_segments.append(merged)
                current_nodes = [
                    node
                    for node in current_nodes
                    if _arrangement_node_key(node) != str(node_key)
                ]
                removed_keys.append(str(node_key))
                merged_records.append(
                    {
                        "removed_key": str(node_key),
                        "merged_pair": [int(pair0[0]), int(pair0[1])],
                        "endpoints": [str(other0), str(other1)],
                        "reason": "single_carrier_collinear_split",
                    }
                )
                collapsed_key = str(node_key)
                break

            if len(incident) <= 1:
                seg_indices = sorted({int(idx) for idx, _other, _pair in incident}, reverse=True)
                for seg_idx in seg_indices:
                    current_segments.pop(int(seg_idx))
                current_nodes = [
                    node
                    for node in current_nodes
                    if _arrangement_node_key(node) != str(node_key)
                ]
                removed_keys.append(str(node_key))
                merged_records.append(
                    {
                        "removed_key": str(node_key),
                        "reason": "unsupported_dangling_virtual",
                        "n_incident_segments": int(len(incident)),
                    }
                )
                collapsed_key = str(node_key)
                break

        if collapsed_key is None:
            break

    _virtual_junction_nodes(
        [node for node in current_nodes if int(node.vertex_id) < 0]
    )
    diagnostics: Dict[str, object] = {
        "attempted": bool(segments),
        "accepted": bool(removed_keys),
        "passes": int(len(merged_records)),
        "removed_node_keys": removed_keys,
        "merged_records": merged_records,
        "n_segments_before": int(len(segments)),
        "n_segments_after": int(len(current_segments)),
        "n_virtual_nodes_after": int(
            sum(1 for node in current_nodes if int(node.vertex_id) < 0)
        ),
    }
    return current_segments, current_nodes, diagnostics


def _select_validated_segments(
    labels: Sequence[int],
    vertices: np.ndarray,
    loop: Sequence[int],
    vertex_labels: Mapping[int, Sequence[int]],
    segments: Sequence[BoundedCurveSegment],
    nodes: Sequence[ArrangementNode],
    cavity: HoleCavity,
    candidate_nodes: Sequence[Mapping[str, object]],
    patch_surface_fits: Mapping[int, SurfaceFit],
    bridge_proofs: Sequence[Mapping[str, object]],
    *,
    arc_corner_hints: Optional[Mapping[Tuple[int, int], Sequence[int]]] = None,
) -> Tuple[List[BoundedCurveSegment], bool, List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    optional_proofs: List[Mapping[str, object]] = [dict(item) for item in bridge_proofs]
    known_optional = _optional_edge_ids_from_proofs(optional_proofs, len(segments))
    for edge_id, seg in enumerate(segments):
        if int(edge_id) in known_optional:
            continue
        e0, e1 = (
            int(seg.boundary_vertex_indices[0]),
            int(seg.boundary_vertex_indices[1]),
        )
        if e0 < 0 and e1 < 0:
            optional_proofs.append(
                {
                    "accepted": True,
                    "optional_edge": True,
                    "edge_id": int(edge_id),
                    "pair": tuple(sorted(int(x) for x in seg.analytic.patch_pair)),
                    "reason": "virtual_virtual_candidate_prune",
                }
            )
    return _select_minimal_legal_cell_complex(
        labels,
        vertices,
        loop,
        vertex_labels,
        segments,
        nodes,
        cavity,
        candidate_nodes,
        patch_surface_fits,
        optional_proofs,
        arc_corner_hints=arc_corner_hints,
    )


def _virtual_node_relocation_candidates(
    node: ArrangementNode,
    carriers: Mapping[Tuple[int, int], AnalyticCurve],
    cavity: HoleCavity,
    vertices: np.ndarray,
) -> List[np.ndarray]:
    pairs = sorted({tuple(sorted(pair)) for pair in node.incident_pairs})
    if len(pairs) < 2:
        if len(pairs) != 1:
            return []
        ac = carriers.get(pairs[0])
        if ac is None:
            return []
        line = _analytic_curve_as_line(ac, node.position)
        if line is None:
            return []
        lp, ld = line
        ld = _safe_unit(ld)
        if float(np.linalg.norm(ld)) <= 1e-12:
            return []
        pos = np.asarray(node.position, dtype=np.float64).reshape(3)
        t0 = float(np.dot(pos - np.asarray(lp, dtype=np.float64).reshape(3), ld))
        step = max(float(cavity.mean_edge_length), 0.04 * float(cavity.bbox_diag), 1e-9)
        out: List[np.ndarray] = []
        for scale in (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0):
            p = np.asarray(lp, dtype=np.float64).reshape(3) + (t0 + scale * step) * ld
            if float(np.linalg.norm(p - cavity.center)) > _max_extent(cavity):
                continue
            if not cavity.contains(vertices, p):
                continue
            out.append(np.asarray(p, dtype=np.float64))
        return out
    pos = np.asarray(node.position, dtype=np.float64).reshape(3)
    center = np.asarray(cavity.center, dtype=np.float64).reshape(3)
    loop_pts = vertices[np.asarray(cavity.loop, dtype=np.int64)]
    rel = loop_pts - np.mean(loop_pts, axis=0).reshape(1, 3)
    try:
        _, _, vh = np.linalg.svd(rel, full_matrices=False)
        axes = [vh[0], vh[1]]
    except np.linalg.LinAlgError:
        axes = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
    step = max(float(cavity.mean_edge_length), 0.04 * float(cavity.bbox_diag), 1e-9)
    guides: List[np.ndarray] = [
        pos,
        center,
        0.5 * (pos + center),
        center + 0.65 * (pos - center),
        center + 1.35 * (pos - center),
    ]
    for axis in axes:
        a = _safe_unit(axis)
        if float(np.linalg.norm(a)) <= 1e-12:
            continue
        guides.extend([pos + step * a, pos - step * a, center + step * a, center - step * a])

    out: List[np.ndarray] = []
    merge_tol = max(0.25 * _node_merge_tol(cavity), 1e-9)
    for pair_a, pair_b in combinations(pairs, 2):
        ac_a = carriers.get(pair_a)
        ac_b = carriers.get(pair_b)
        if ac_a is None or ac_b is None:
            continue
        for guide in guides:
            cand = intersect_analytic_curves(ac_a, ac_b, guide_point=guide)
            if cand is None:
                continue
            p = np.asarray(cand, dtype=np.float64).reshape(3)
            if not np.all(np.isfinite(p)):
                continue
            if float(np.linalg.norm(p - center)) > _max_extent(cavity):
                continue
            if not cavity.contains(vertices, p):
                continue
            if any(float(np.linalg.norm(p - q)) <= merge_tol for q in out):
                continue
            out.append(p)
    return out


def _refine_virtual_nodes_for_cell_validation(
    labels: Sequence[int],
    vertices: np.ndarray,
    loop: Sequence[int],
    vertex_labels: Mapping[int, Sequence[int]],
    carriers: Mapping[Tuple[int, int], AnalyticCurve],
    nodes: Sequence[ArrangementNode],
    cavity: HoleCavity,
    patch_surface_fits: Mapping[int, SurfaceFit],
    current_segments: Sequence[BoundedCurveSegment],
    current_valid: bool,
    current_validation: Sequence[Mapping[str, object]],
    *,
    arc_corner_hints: Optional[Mapping[Tuple[int, int], Sequence[int]]] = None,
    paired_boundary_vertices: Optional[Mapping[Tuple[int, int], Tuple[int, int]]] = None,
) -> Tuple[
    List[BoundedCurveSegment],
    List[Dict[str, object]],
    bool,
    List[Dict[str, object]],
    List[Dict[str, object]],
    Dict[str, object],
]:
    """Local experiment: re-solve virtual junction branch if it improves cell proof."""
    internal = [n for n in nodes if int(n.vertex_id) < 0]
    base_score = _surface_cell_score(
        bool(current_valid),
        current_validation,
        n_edges=len(current_segments),
    )
    best_score = base_score
    best_payload: Optional[
        Tuple[
            List[BoundedCurveSegment],
            List[Dict[str, object]],
            bool,
            List[Dict[str, object]],
            List[Dict[str, object]],
            int,
            np.ndarray,
        ]
    ] = None
    attempts = 0
    max_attempts = 36
    original_positions = {
        int(node.node_id): np.asarray(node.position, dtype=np.float64).copy()
        for node in internal
    }

    for node in internal:
        candidates = _virtual_node_relocation_candidates(node, carriers, cavity, vertices)
        before = np.asarray(node.position, dtype=np.float64).copy()
        for cand in candidates:
            if attempts >= max_attempts:
                break
            if float(np.linalg.norm(cand - before)) <= max(1e-9, 0.05 * _node_merge_tol(cavity)):
                continue
            attempts += 1
            node.position = np.asarray(cand, dtype=np.float64).reshape(3)
            trial_nodes = _candidate_node_records(nodes)
            trial_segments, trial_bridge_proofs = _build_topology_segments(
                carriers,
                nodes,
                cavity,
                vertices,
                loop,
                vertex_labels,
                patch_surface_fits,
                arc_corner_hints=arc_corner_hints,
                paired_boundary_vertices=paired_boundary_vertices,
            )
            valid, validation, edges = _validate_segment_set(
                labels,
                vertices,
                loop,
                vertex_labels,
                trial_segments,
                nodes,
                cavity,
                trial_nodes,
                patch_surface_fits,
                arc_corner_hints=arc_corner_hints,
            )
            score = _surface_cell_score(
                bool(valid),
                validation,
                n_edges=len(trial_segments),
            )
            if score < best_score:
                best_score = score
                best_payload = (
                    trial_segments,
                    trial_bridge_proofs,
                    bool(valid),
                    validation,
                    edges,
                    int(node.node_id),
                    np.asarray(cand, dtype=np.float64).copy(),
                )
                if bool(valid):
                    break
        node.position = before
        if best_payload is not None and bool(best_payload[2]):
            break

    for node in internal:
        node.position = original_positions[int(node.node_id)].copy()

    diagnostics: Dict[str, object] = {
        "attempted": int(attempts),
        "initial_score": list(base_score),
        "accepted": False,
    }
    if best_payload is None:
        diagnostics["reject_reason"] = "no_improving_virtual_branch"
        return (
            list(current_segments),
            [],
            bool(current_valid),
            [dict(x) for x in current_validation],
            _candidate_edge_records(current_segments, nodes, cavity),
            diagnostics,
        )

    segments, bridge_proofs, valid, validation, edges, node_id, position = best_payload
    for node in internal:
        if int(node.node_id) == int(node_id):
            node.position = np.asarray(position, dtype=np.float64).reshape(3)
            break
    diagnostics.update(
        {
            "accepted": True,
            "node_id": int(node_id),
            "final_score": list(best_score),
            "valid_after_refine": bool(valid),
        }
    )
    return segments, bridge_proofs, bool(valid), validation, edges, diagnostics


def _carrier_branch_guide_candidates(
    points: np.ndarray,
    cavity: HoleCavity,
    vertices: np.ndarray,
) -> List[np.ndarray]:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 2:
        return [np.asarray(cavity.center, dtype=np.float64)]
    a = pts[0]
    b = pts[-1]
    mid = 0.5 * (a + b)
    loop_pts = vertices[np.asarray(cavity.loop, dtype=np.int64)]
    rel = loop_pts - np.mean(loop_pts, axis=0).reshape(1, 3)
    try:
        _, _, vh = np.linalg.svd(rel, full_matrices=False)
        normal = _safe_unit(vh[-1])
    except np.linalg.LinAlgError:
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    chord = _safe_unit(b - a)
    side = _safe_unit(np.cross(normal, chord))
    step = max(float(cavity.mean_edge_length), 0.08 * float(cavity.bbox_diag), 1e-9)
    center = np.asarray(cavity.center, dtype=np.float64).reshape(3)
    guides = [center, mid, 2.0 * mid - center]
    if float(np.linalg.norm(side)) > 1e-12:
        guides.extend(
            [
                mid + step * side,
                mid - step * side,
                mid + 2.0 * step * side,
                mid - 2.0 * step * side,
            ]
        )
    return [np.asarray(g, dtype=np.float64).reshape(3) for g in guides]


def _refine_mesh_mesh_carrier_branches(
    labels: Sequence[int],
    vertices: np.ndarray,
    loop: Sequence[int],
    vertex_labels: Mapping[int, Sequence[int]],
    nodes: Sequence[ArrangementNode],
    cavity: HoleCavity,
    patch_surface_fits: Mapping[int, SurfaceFit],
    current_segments: Sequence[BoundedCurveSegment],
    current_valid: bool,
    current_validation: Sequence[Mapping[str, object]],
    *,
    arc_corner_hints: Optional[Mapping[Tuple[int, int], Sequence[int]]] = None,
) -> Tuple[
    List[BoundedCurveSegment],
    bool,
    List[Dict[str, object]],
    List[Dict[str, object]],
    Dict[str, object],
]:
    base_score = _surface_cell_score(
        bool(current_valid),
        current_validation,
        n_edges=len(current_segments),
    )
    mesh_mesh_indices = [
        idx
        for idx, seg in enumerate(current_segments)
        if int(seg.boundary_vertex_indices[0]) >= 0
        and int(seg.boundary_vertex_indices[1]) >= 0
    ]
    if not mesh_mesh_indices:
        return (
            list(current_segments),
            bool(current_valid),
            [dict(x) for x in current_validation],
            _candidate_edge_records(current_segments, nodes, cavity),
            {
                "attempted": 0,
                "accepted": False,
                "initial_score": list(base_score),
                "reject_reason": "no_mesh_mesh_carrier_edges",
            },
        )
    best_score = base_score
    best_payload: Optional[
        Tuple[List[BoundedCurveSegment], bool, List[Dict[str, object]], List[Dict[str, object]], int]
    ] = None
    attempts = 0
    max_attempts = 12
    node_by_vertex = {int(n.vertex_id): n for n in nodes if int(n.vertex_id) >= 0}
    for seg_idx in mesh_mesh_indices:
        seg = current_segments[int(seg_idx)]
        e0, e1 = (int(seg.boundary_vertex_indices[0]), int(seg.boundary_vertex_indices[1]))
        if e0 < 0 or e1 < 0 or e0 not in node_by_vertex or e1 not in node_by_vertex:
            continue
        for guide in _carrier_branch_guide_candidates(seg.curve_points, cavity, vertices):
            if attempts >= max_attempts:
                break
            attempts += 1
            rebuilt = _segment_between_nodes(
                node_by_vertex[e0],
                node_by_vertex[e1],
                seg.analytic,
                cavity,
                vertices,
                patch_surface_fits,
                guide_point=guide,
            )
            if rebuilt is None:
                continue
            trial_segments = list(current_segments)
            trial_segments[int(seg_idx)] = rebuilt
            valid, validation, edges = _validate_segment_set(
                labels,
                vertices,
                loop,
                vertex_labels,
                trial_segments,
                nodes,
                cavity,
                _candidate_node_records(nodes),
                patch_surface_fits,
                arc_corner_hints=arc_corner_hints,
            )
            score = _surface_cell_score(bool(valid), validation, n_edges=len(trial_segments))
            if score < best_score:
                best_score = score
                best_payload = (trial_segments, bool(valid), validation, edges, int(seg_idx))
                if bool(valid):
                    break
        if best_payload is not None and bool(best_payload[1]):
            break
    diagnostics: Dict[str, object] = {
        "attempted": int(attempts),
        "accepted": False,
        "initial_score": list(base_score),
    }
    if best_payload is None:
        diagnostics["reject_reason"] = "no_improving_carrier_branch"
        return (
            list(current_segments),
            bool(current_valid),
            [dict(x) for x in current_validation],
            _candidate_edge_records(current_segments, nodes, cavity),
            diagnostics,
        )
    segments, valid, validation, edges, seg_idx = best_payload
    diagnostics.update(
        {
            "accepted": True,
            "segment_index": int(seg_idx),
            "valid_after_refine": bool(valid),
            "final_score": list(best_score),
        }
    )
    return segments, bool(valid), validation, edges, diagnostics


VIRTUAL_JUNCTION_SOURCE_BASE = 920_000


def canonical_junction_source(node_id: int) -> int:
    """Arrangement 虚拟汇交 source（与 L3 boundary_sources 一致）。"""
    return -(VIRTUAL_JUNCTION_SOURCE_BASE + int(node_id))


def _feature_curve_guide_point(
    start: np.ndarray,
    end: np.ndarray,
    fallback_center: Optional[np.ndarray] = None,
    *,
    endpoint_vertex_indices: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    if fallback_center is not None and endpoint_vertex_indices is not None:
        v0, v1 = (
            int(endpoint_vertex_indices[0]),
            int(endpoint_vertex_indices[1]),
        )
        if (v0 >= 0 and v1 < 0) or (v1 >= 0 and v0 < 0) or (v0 < 0 and v1 < 0):
            return np.asarray(fallback_center, dtype=np.float64).reshape(3)
    mid = 0.5 * (start + end)
    if fallback_center is None:
        return mid
    center = np.asarray(fallback_center, dtype=np.float64)
    return 0.85 * mid + 0.15 * center


def _curve_midpoint_guide(
    curve_points: np.ndarray,
    fallback_center: np.ndarray,
) -> np.ndarray:
    """Use the existing recovered carrier shape as the branch guide when rebuilding endpoints."""
    pts = np.asarray(curve_points, dtype=np.float64)
    if pts.ndim == 2 and pts.shape[0] >= 3:
        return np.asarray(pts[int(pts.shape[0]) // 2], dtype=np.float64).reshape(3)
    return np.asarray(fallback_center, dtype=np.float64).reshape(3)


def _virtual_junction_nodes(nodes: Sequence[ArrangementNode]) -> List[ArrangementNode]:
    virtual = [n for n in nodes if int(n.vertex_id) < 0]
    for idx, node in enumerate(virtual):
        node.node_id = int(idx)
    return virtual


def _rebuild_curves_with_cavity_junction_nodes(
    curves: Sequence[IntersectionCurve],
    nodes: Sequence[ArrangementNode],
    cavity: HoleCavity,
    patch_surface_fits: Mapping[int, SurfaceFit],
    hole_center: np.ndarray,
    *,
    junction_point: Optional[np.ndarray] = None,
) -> List[IntersectionCurve]:
    virtual_nodes = _virtual_junction_nodes(list(nodes))
    if not virtual_nodes:
        return list(curves)
    loop_step = float(cavity.mean_edge_length)
    match_tol = max(
        float(cavity.junction_cluster_tol),
        2.5 * float(cavity.mean_edge_length),
        0.06 * float(cavity.bbox_diag),
    )
    cluster_centers = [
        np.asarray(node.position, dtype=np.float64).reshape(3) for node in virtual_nodes
    ]
    cluster_sources = [
        canonical_junction_source(int(node.node_id)) for node in virtual_nodes
    ]

    def _nearest_virtual_id(point: np.ndarray, pair: Tuple[int, int]) -> Optional[int]:
        best_id: Optional[int] = None
        best_d = float(match_tol)
        p = np.asarray(point, dtype=np.float64).reshape(3)
        pair_key = tuple(sorted((int(pair[0]), int(pair[1]))))
        for node in virtual_nodes:
            if pair_key not in {tuple(sorted(x)) for x in node.incident_pairs}:
                continue
            d = float(np.linalg.norm(p - node.position))
            if d <= best_d:
                best_d = d
                best_id = int(node.node_id)
        return best_id

    rebuilt: List[IntersectionCurve] = []
    for curve in curves:
        pts = np.asarray(curve.curve_points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] < 2:
            rebuilt.append(curve)
            continue
        e0, e1 = (
            int(curve.endpoint_vertex_indices[0]),
            int(curve.endpoint_vertex_indices[1]),
        )
        pair = tuple(sorted((int(curve.patch_pair[0]), int(curve.patch_pair[1]))))
        if pair[0] not in patch_surface_fits or pair[1] not in patch_surface_fits:
            rebuilt.append(curve)
            continue

        has_v0 = e0 < 0
        has_v1 = e1 < 0
        if not has_v0 and not has_v1:
            rebuilt.append(curve)
            continue

        if has_v0 and has_v1:
            c0 = _nearest_virtual_id(pts[0], pair)
            c1 = _nearest_virtual_id(pts[-1], pair)
            if c0 is None or c1 is None:
                continue
            if int(c0) == int(c1):
                continue
            p0 = np.asarray(cluster_centers[int(c0)], dtype=np.float64)
            p1 = np.asarray(cluster_centers[int(c1)], dtype=np.float64)
            local_guide = _curve_midpoint_guide(pts, hole_center)
            guide = _feature_curve_guide_point(
                p0,
                p1,
                local_guide,
                endpoint_vertex_indices=(
                    int(cluster_sources[int(c0)]),
                    int(cluster_sources[int(c1)]),
                ),
            )
            rebuilt.append(
                recover_curve_between_points(
                    patch_surface_fits[pair[0]],
                    patch_surface_fits[pair[1]],
                    p0,
                    p1,
                    guide,
                    endpoint_vertex_indices=(
                        int(cluster_sources[int(c0)]),
                        int(cluster_sources[int(c1)]),
                    ),
                    intersection_sampling_reference_step=loop_step,
                )
            )
            continue

        if has_v1:
            ci = _nearest_virtual_id(pts[-1], pair)
            if ci is None:
                continue
            junction = np.asarray(cluster_centers[int(ci)], dtype=np.float64)
            fixed_pt = np.asarray(pts[0], dtype=np.float64)
            local_guide = _curve_midpoint_guide(pts, hole_center)
            guide = _feature_curve_guide_point(
                fixed_pt,
                junction,
                local_guide,
                endpoint_vertex_indices=(int(e0), int(cluster_sources[int(ci)])),
            )
            rebuilt.append(
                recover_curve_between_points(
                    patch_surface_fits[pair[0]],
                    patch_surface_fits[pair[1]],
                    fixed_pt,
                    junction,
                    guide,
                    endpoint_vertex_indices=(int(e0), int(cluster_sources[int(ci)])),
                    intersection_sampling_reference_step=loop_step,
                )
            )
            continue

        ci = _nearest_virtual_id(pts[0], pair)
        if ci is None:
            continue
        junction = np.asarray(cluster_centers[int(ci)], dtype=np.float64)
        fixed_pt = np.asarray(pts[-1], dtype=np.float64)
        local_guide = _curve_midpoint_guide(pts, hole_center)
        guide = _feature_curve_guide_point(
            junction,
            fixed_pt,
            local_guide,
            endpoint_vertex_indices=(int(cluster_sources[int(ci)]), int(e1)),
        )
        rebuilt.append(
            recover_curve_between_points(
                patch_surface_fits[pair[0]],
                patch_surface_fits[pair[1]],
                junction,
                fixed_pt,
                guide,
                endpoint_vertex_indices=(int(cluster_sources[int(ci)]), int(e1)),
                intersection_sampling_reference_step=loop_step,
            )
        )
    return rebuilt


def finalize_cavity_arrangement_layout(
    cavity_result: CavityArrangementResult,
    patch_surface_fits: Mapping[int, SurfaceFit],
    *,
    hole_center: np.ndarray,
) -> CavityArrangementResult:
    """L2 单一 layout 出口：定稿虚拟 source，L3 禁止再聚类。"""
    curves = _rebuild_curves_with_cavity_junction_nodes(
        cavity_result.curves,
        cavity_result.junction_nodes,
        cavity_result.cavity,
        patch_surface_fits,
        hole_center,
        junction_point=cavity_result.junction_point,
    )
    virtual_nodes = _virtual_junction_nodes(list(cavity_result.junction_nodes))
    diagnostics = dict(cavity_result.diagnostics)
    diagnostics["layout_finalized"] = True
    diagnostics["layout_source"] = "cavity_topology_arrangement"
    diagnostics["virtual_sources"] = [
        int(canonical_junction_source(int(n.node_id))) for n in virtual_nodes
    ]
    return CavityArrangementResult(
        cavity=cavity_result.cavity,
        curves=curves,
        bounded_segments=list(cavity_result.bounded_segments),
        analytic_curves=list(cavity_result.analytic_curves),
        junction_nodes=virtual_nodes,
        junction_point=cavity_result.junction_point,
        junction_confidence=str(cavity_result.junction_confidence),
        diagnostics=diagnostics,
    )


def recover_cavity_restricted_curves(
    patch_surface_fits: Mapping[int, SurfaceFit],
    patch_pairs: Iterable[Tuple[int, int]],
    vertices: np.ndarray,
    loop: Sequence[int],
    *,
    hole_center: np.ndarray,
    loop_mean_edge: Optional[float] = None,
    vertex_labels: Optional[Mapping[int, Sequence[int]]] = None,
    arc_corner_hints: Optional[Mapping[Tuple[int, int], Sequence[int]]] = None,
) -> CavityArrangementResult:
    """
    孔腔约束 arrangement 恢复（L2 唯一几何入口）。

    由 carrier 联立交点 + 沿 carrier 连边构造 Γ_ij，无 per-pair 聚类。
    """
    cavity = build_hole_cavity(
        vertices,
        loop,
        hole_center,
        loop_mean_edge=loop_mean_edge,
    )
    if vertex_labels is None:
        vertex_labels = {}

    labels = _active_labels(patch_surface_fits, patch_pairs)
    carriers_raw = _build_carrier_map(patch_surface_fits, labels)
    carriers, carrier_certificates = _gate_certified_carriers(
        carriers_raw,
        cavity,
        vertices,
        filter_carriers=False,
    )
    mesh_corners = _mesh_corner_vertices(
        loop,
        vertex_labels,
        arc_corner_hints=arc_corner_hints,
    )
    paired_boundary_vertices, boundary_pair_certificates = (
        _boundary_pair_certificate_records(
            carriers,
            mesh_corners,
            cavity,
            vertices,
            vertex_labels,
            arc_corner_hints=arc_corner_hints,
        )
    )
    protected_boundary_pairs = set(paired_boundary_vertices)
    complete_boundary_pair_cover = (
        len(mesh_corners) >= 6
        and len(mesh_corners) % 2 == 0
        and len(paired_boundary_vertices) * 2 == len(mesh_corners)
        and {
            int(v)
            for anchors in paired_boundary_vertices.values()
            for v in anchors
        }
        == {int(v) for v in mesh_corners}
    )
    if complete_boundary_pair_cover:
        carriers = {
            tuple(pair): carriers[tuple(pair)]
            for pair in sorted(paired_boundary_vertices)
            if tuple(pair) in carriers
        }
    arrangement_nodes_raw = _build_arrangement_nodes(
        carriers,
        cavity,
        vertices,
        loop,
        vertex_labels,
        arc_corner_hints=arc_corner_hints,
        protected_boundary_pairs=protected_boundary_pairs,
    )
    arrangement_nodes = list(arrangement_nodes_raw)
    junction_certificates: List[Dict[str, object]] = []
    virtual_nodes = _virtual_junction_nodes(
        [n for n in arrangement_nodes if int(n.vertex_id) < 0]
    )
    junction_refinement = _refine_internal_nodes_with_shared_carriers(
        arrangement_nodes,
        carriers,
        cavity,
        vertices,
    )
    incident_completion = _complete_single_junction_incident_pairs(
        arrangement_nodes,
        carriers,
    )
    bounded_segments, bridge_proofs = _build_topology_segments(
        carriers,
        arrangement_nodes,
        cavity,
        vertices,
        loop,
        vertex_labels,
        patch_surface_fits,
        arc_corner_hints=arc_corner_hints,
        paired_boundary_vertices=paired_boundary_vertices,
    )
    post_build_completion = _complete_single_junction_incident_pairs(
        arrangement_nodes,
        carriers,
    )
    if post_build_completion:
        incident_completion.extend(post_build_completion)
        bounded_segments, bridge_proofs = _build_topology_segments(
            carriers,
            arrangement_nodes,
            cavity,
            vertices,
            loop,
            vertex_labels,
            patch_surface_fits,
            arc_corner_hints=arc_corner_hints,
            paired_boundary_vertices=paired_boundary_vertices,
        )
    arrangement_nodes, junction_certificates = _gate_certified_junctions(
        arrangement_nodes,
        carriers,
        cavity,
        vertices,
        patch_surface_fits,
        filter_nodes=False,
    )
    virtual_nodes = _virtual_junction_nodes(
        [n for n in arrangement_nodes if int(n.vertex_id) < 0]
    )
    analytic_curves = list(carriers.values())
    candidate_nodes = _candidate_node_records(arrangement_nodes)
    (
        bounded_segments,
        cells_valid,
        surface_cell_validation,
        candidate_edges,
        selection_diagnostics,
    ) = _select_validated_segments(
        labels,
        vertices,
        loop,
        vertex_labels,
        bounded_segments,
        arrangement_nodes,
        cavity,
        candidate_nodes,
        patch_surface_fits,
        bridge_proofs,
        arc_corner_hints=arc_corner_hints,
    )
    virtual_refinement_diagnostics: Dict[str, object] = {"attempted": 0, "accepted": False}
    if not bool(cells_valid):
        (
            refined_segments,
            refined_bridge_proofs,
            refined_valid,
            refined_validation,
            refined_edges,
            virtual_refinement_diagnostics,
        ) = _refine_virtual_nodes_for_cell_validation(
            labels,
            vertices,
            loop,
            vertex_labels,
            carriers,
            arrangement_nodes,
            cavity,
            patch_surface_fits,
            bounded_segments,
            cells_valid,
            surface_cell_validation,
            arc_corner_hints=arc_corner_hints,
            paired_boundary_vertices=paired_boundary_vertices,
        )
        if bool(virtual_refinement_diagnostics.get("accepted", False)):
            bounded_segments = refined_segments
            bridge_proofs = refined_bridge_proofs
            cells_valid = bool(refined_valid)
            surface_cell_validation = refined_validation
            candidate_edges = refined_edges
            candidate_nodes = _candidate_node_records(arrangement_nodes)
    carrier_branch_diagnostics: Dict[str, object] = {"attempted": 0, "accepted": False}
    if not bool(cells_valid):
        (
            branch_segments,
            branch_valid,
            branch_validation,
            branch_edges,
            carrier_branch_diagnostics,
        ) = _refine_mesh_mesh_carrier_branches(
            labels,
            vertices,
            loop,
            vertex_labels,
            arrangement_nodes,
            cavity,
            patch_surface_fits,
            bounded_segments,
            cells_valid,
            surface_cell_validation,
            arc_corner_hints=arc_corner_hints,
        )
        if bool(carrier_branch_diagnostics.get("accepted", False)):
            bounded_segments = branch_segments
            cells_valid = bool(branch_valid)
            surface_cell_validation = branch_validation
            candidate_edges = branch_edges

    (
        bounded_segments,
        arrangement_nodes,
        virtual_junction_prune_diagnostics,
    ) = _prune_unsupported_virtual_junctions(
        bounded_segments,
        arrangement_nodes,
        carriers,
        cavity,
        vertices,
        patch_surface_fits,
    )
    virtual_nodes = _virtual_junction_nodes(
        [n for n in arrangement_nodes if int(n.vertex_id) < 0]
    )
    candidate_nodes = _candidate_node_records(arrangement_nodes)
    cells_valid, surface_cell_validation, candidate_edges = _validate_segment_set(
        labels,
        vertices,
        loop,
        vertex_labels,
        bounded_segments,
        arrangement_nodes,
        cavity,
        candidate_nodes,
        patch_surface_fits,
        arc_corner_hints=arc_corner_hints,
    )
    arrangement_nodes, junction_certificates = _gate_certified_junctions(
        arrangement_nodes,
        carriers,
        cavity,
        vertices,
        patch_surface_fits,
        filter_nodes=False,
    )
    virtual_nodes = _virtual_junction_nodes(
        [n for n in arrangement_nodes if int(n.vertex_id) < 0]
    )

    optional_ids = _optional_edge_ids_from_proofs(bridge_proofs, len(bounded_segments))
    edge_certificates = [
        {
            "edge_id": int(cert.edge_id),
            "pair": [int(cert.pair[0]), int(cert.pair[1])],
            "certified": bool(cert.certified),
            "midpoint_in_cavity": bool(cert.midpoint_in_cavity),
            "max_endpoint_carrier_dist": float(cert.max_endpoint_carrier_dist),
            "optional_edge": bool(cert.optional_edge),
            "reject_reason": str(cert.reject_reason),
        }
        for cert in (
            _certify_edge_segment(
                idx,
                seg,
                cavity,
                vertices,
                optional_edge=int(idx) in optional_ids,
            )
            for idx, seg in enumerate(bounded_segments)
        )
    ]
    cell_proof_report = [
        dict(item.get("proof_report", {}) or {})
        for item in surface_cell_validation
    ]
    virtual_junction_certificate_valid = all(
        bool(item.get("certified", False))
        for item in junction_certificates
        if int(item.get("vertex_id", -1)) < 0
    )
    edge_certificate_valid = all(
        bool(item.get("certified", False)) or bool(item.get("optional_edge", False))
        for item in edge_certificates
    )
    certificate_gate_valid = bool(virtual_junction_certificate_valid) and bool(
        edge_certificate_valid
    )

    curves: List[IntersectionCurve] = []
    for seg in bounded_segments:
        curves.append(
            bounded_segment_to_intersection_curve(
                seg,
                intersection_sampling_reference_step=cavity.mean_edge_length,
                vertices=vertices,
            )
        )

    junction_point: Optional[np.ndarray] = None
    junction_confidence = "none"
    if len(virtual_nodes) == 1:
        junction_point = np.asarray(virtual_nodes[0].position, dtype=np.float64).reshape(3)
        junction_confidence = "high"
    elif len(virtual_nodes) >= 2:
        junction_point = np.mean(
            np.vstack([n.position for n in virtual_nodes]),
            axis=0,
        )
        junction_confidence = "medium"

    pair_keys = {tuple(sorted(seg.analytic.patch_pair)) for seg in bounded_segments}
    diagnostics: Dict[str, object] = {
        "recovery_mode": "cavity_topology_arrangement",
        "cavity_center": cavity.center.tolist(),
        "junction_cluster_tol": float(cavity.junction_cluster_tol),
        "n_carriers": len(carriers),
        "n_carriers_raw": len(carriers_raw),
        "carrier_certificates": carrier_certificates,
        "boundary_pair_certificates": boundary_pair_certificates,
        "protected_boundary_pairs": [
            [int(pair[0]), int(pair[1])]
            for pair in sorted(paired_boundary_vertices)
        ],
        "complete_boundary_pair_cover": bool(complete_boundary_pair_cover),
        "junction_certificates": junction_certificates,
        "edge_certificates": edge_certificates,
        "cell_proof_report": cell_proof_report,
        "n_arrangement_nodes": len(arrangement_nodes),
        "n_arrangement_nodes_raw": len(arrangement_nodes_raw),
        "n_internal_nodes": len(virtual_nodes),
        "n_segments": len(bounded_segments),
        "n_junction_nodes": len(virtual_nodes),
        "n_bridge_segments": sum(
            1
            for seg in bounded_segments
            if int(seg.boundary_vertex_indices[0]) < 0
            and int(seg.boundary_vertex_indices[1]) < 0
        ),
        "internal_bridge_proofs": bridge_proofs,
        "junction_refinement": junction_refinement,
        "incident_completion": incident_completion,
        "candidate_nodes": candidate_nodes,
        "candidate_edges": candidate_edges,
        "surface_cell_validation": surface_cell_validation,
        "candidate_selection": selection_diagnostics,
        "virtual_cell_refinement": virtual_refinement_diagnostics,
        "carrier_branch_refinement": carrier_branch_diagnostics,
        "virtual_junction_prune": virtual_junction_prune_diagnostics,
        "certificate_gate_valid": bool(certificate_gate_valid),
        "virtual_junction_certificate_valid": bool(virtual_junction_certificate_valid),
        "edge_certificate_valid": bool(edge_certificate_valid),
        "cell_validation_valid": bool(cells_valid),
        "arrangement_valid": (
            bool(cells_valid)
            and bool(certificate_gate_valid)
            and all(
                bool(proof.get("accepted", False))
                for proof in bridge_proofs
                if len(proof.get("shared_labels", [])) == 2
                and "pair" in proof
                and str(proof.get("reject_reason", "")) != "already_present"
            )
        ),
        "segment_pairs": sorted(pair_keys),
    }

    return CavityArrangementResult(
        cavity=cavity,
        curves=curves,
        bounded_segments=bounded_segments,
        analytic_curves=analytic_curves,
        junction_nodes=virtual_nodes,
        junction_point=junction_point,
        junction_confidence=junction_confidence,
        diagnostics=diagnostics,
    )
