#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""孔洞分析核心数据类型（补洞管线消费）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

from .surface_fitting import SurfaceFit
from .surface_intersections import AnalyticCurve, BoundedCurveSegment, IntersectionCurve
from .surface_parameterization import SurfaceParameterization

if TYPE_CHECKING:
    from .feature_graph import FeatureArrangement, FeatureGraph

HoleType = str

SINGLE_PATCH: HoleType = "single_patch"
MULTI_PATCH: HoleType = "multi_patch"

# 以下仅用于 diagnostics.template_hint，不参与补洞决策。
SHARP_EDGE_CROSSING: HoleType = "sharp_edge_crossing"
TRIPLE_SURFACE_JUNCTION: HoleType = "triple_surface_junction"
CLOSED_PAIR_CURVE_MULTI_PATCH: HoleType = "closed_pair_curve_multi_patch"
S2C_MULTI_CORNER_PAIR: HoleType = "s2c_multi_corner_pair"
MULTI_PATCH_GENERIC: HoleType = "multi_patch_generic"

PUBLIC_HOLE_TYPES = frozenset({SINGLE_PATCH, MULTI_PATCH})

# L3 剖分障碍（与 stage 无关的统一诊断）
PARTITION_OBSTACLE_O1 = "O1_cycle_open"
PARTITION_OBSTACLE_O2 = "O2_missing_intersection"
PARTITION_OBSTACLE_O3 = "O3_coverage_gap"
PARTITION_OBSTACLE_O4 = "O4_non_wedge"


@dataclass(frozen=True)
class HoleScale:
    """孔环局部尺度：全管线唯一几何归一化源。"""

    loop_perimeter: float
    mean_edge_length: float
    bbox_diag: float

    @property
    def node_merge_tol(self) -> float:
        return max(1e-9, 1e-8 * float(self.bbox_diag))

@dataclass(frozen=True)
class PartitionObstacle:
    """L3 剖分失败时可报告的最小障碍单元。"""

    kind: str
    label: Optional[int]
    detail: str
    layer: str = "L3"

@dataclass
class BoundaryArc:
    """孔边上属于同一局部面域的一段连续弧段。"""

    patch_label: int
    edge_indices: List[int]
    vertex_indices: List[int]
    source_face_patch_label: Optional[int] = None


@dataclass
class PreparedSubhole:
    """分区补洞用的闭合子孔边界。"""

    patch_label: int
    boundary_vertex_indices: List[int]
    boundary_points: np.ndarray
    open_boundary_edges: np.ndarray
    closed_boundary_points: np.ndarray
    closed_boundary_edges: np.ndarray
    boundary_points_2d: np.ndarray
    parameterization_kind: str
    boundary_sources: List[int]
    closure_kind: str
    feature_point_vertex_indices: Tuple[int, int]
    reference_normal: Optional[np.ndarray] = None
    parameterization: Optional[SurfaceParameterization] = None


@dataclass(frozen=True)
class FillGateResult:
    """
    S7 窄腰验收：补洞管线（S8）应优先读此对象，而非解析 diagnostics 字符串。

    - ``accepted``：``prepared_subholes`` 是否通过 ``_accept_prepared_subholes``
    - ``expected_labels`` / ``got_labels``：L1 回归核心字段
    """

    expected_labels: FrozenSet[int]
    got_labels: FrozenSet[int]
    accepted: bool
    pipeline_stage: str
    reject_reason: str = ""


@dataclass(frozen=True)
class FillOwnershipSnapshot:
    """
    L2 一次性定稿：聚类 label → 补洞 label 的裁决结果。

    下游 L3/L4 只读此快照，不得再回溯修改 active/support 或重剪交线。
    """

    cluster_labels: FrozenSet[int]
    active_fill_labels: FrozenSet[int]
    support_labels: FrozenSet[int]
    kept_curve_pairs: Tuple[Tuple[int, int], ...]
    removed_curve_pairs: Tuple[Tuple[int, int], ...]
    demoted_feature_points: FrozenSet[int]
    active_feature_points: FrozenSet[int]


@dataclass
class FillPatchClassification:
    """活跃补洞面 / 支撑条带 / 退化路径分类。"""

    active_fill_labels: Set[int]
    support_labels: Set[int]
    degenerate_label_paths: Dict[int, List[List[int]]]
    inactive_feature_points: Set[int]
    active_feature_points: Set[int]
    suppressed_pairs: Set[Tuple[int, int]]
    diagnostics: Dict[str, object]
    ownership_snapshot: Optional[FillOwnershipSnapshot] = None


FillStrategy = str

FILL_STRATEGY_WHOLE_LOOP: FillStrategy = "whole_loop"
FILL_STRATEGY_OPENING_CARRIER: FillStrategy = "opening_carrier"
FILL_STRATEGY_CURVE_ARC_PARTITION: FillStrategy = "curve_arc_partition"


@dataclass(frozen=True)
class FillPlan:
    """
    L2/L3 定稿后的补洞路由契约（供 L4 与批处理只读消费）。

    - ``boundary_patch_count`` (K): 孔环边界弧 label 数
    - ``active_fill_labels`` (M): 需要生成 fill 的 label 集
    - ``fill_strategy``: whole_loop | opening_carrier | curve_arc_partition
    """

    boundary_patch_count: int
    active_fill_labels: FrozenSet[int]
    support_labels: FrozenSet[int]
    fill_strategy: FillStrategy
    skipped_intersection_recovery: bool = False


def infer_fill_strategy(
    boundary_patch_count: int,
    active_fill_labels: Set[int] | FrozenSet[int],
) -> FillStrategy:
    """由 K 与 |M| 推断 L3/L4 补洞策略（单一裁决入口）。"""
    if int(boundary_patch_count) <= 1:
        return FILL_STRATEGY_WHOLE_LOOP
    if len(active_fill_labels) <= 1:
        return FILL_STRATEGY_OPENING_CARRIER
    return FILL_STRATEGY_CURVE_ARC_PARTITION


def build_fill_plan(
    boundary_patch_count: int,
    fill_classification: FillPatchClassification,
    *,
    skipped_intersection_recovery: bool = False,
) -> FillPlan:
    active = frozenset(int(x) for x in fill_classification.active_fill_labels)
    support = frozenset(int(x) for x in fill_classification.support_labels)
    strategy = infer_fill_strategy(int(boundary_patch_count), active)
    return FillPlan(
        boundary_patch_count=int(boundary_patch_count),
        active_fill_labels=active,
        support_labels=support,
        fill_strategy=strategy,
        skipped_intersection_recovery=bool(skipped_intersection_recovery),
    )


@dataclass
class AnalysisDiagnostics:
    """
    调试、批处理统计与论文 ablation 用信息。
    补洞管线不依赖本对象。
    """

    template_hint: str = ""
    analysis_confidence: float = 1.0
    feature_graph: Optional["FeatureGraph"] = None
    recovery_mode: str = ""
    recovery_diagnostics: Dict[str, object] = field(default_factory=dict)
    neighborhood_face_indices: List[int] = field(default_factory=list)
    surface_patch_labels: Dict[int, int] = field(default_factory=dict)
    patch_face_indices: Dict[int, List[int]] = field(default_factory=dict)
    feature_point_candidates: List[int] = field(default_factory=list)
    feature_edge_candidates: List[Tuple[int, int]] = field(default_factory=list)
    boundary_half_edges: List[int] = field(default_factory=list)
    bounded_segments: List[BoundedCurveSegment] = field(default_factory=list)
    analytic_curves: List[AnalyticCurve] = field(default_factory=list)
    feature_arrangement: Optional["FeatureArrangement"] = None


@dataclass
class HoleAnalysis:
    """孔洞局部结构分析核心结果。"""

    boundary_vertices: List[int]
    patch_surface_fits: Dict[int, SurfaceFit]
    boundary_edge_patch_labels: List[int]
    boundary_vertex_patch_labels: Dict[int, List[int]]
    boundary_arcs: List[BoundaryArc]
    intersection_curves: List[IntersectionCurve]
    junction_point: Optional[np.ndarray]
    junction_confidence: str
    prepared_subholes: List[PreparedSubhole]
    hole_type: HoleType
    fill_plan: Optional[FillPlan] = None
    fill_classification: Optional[FillPatchClassification] = None
    fill_gate: Optional[FillGateResult] = None
    hole_scale: Optional[HoleScale] = None
    partition_obstacles: List[PartitionObstacle] = field(default_factory=list)
    diagnostics: Optional[AnalysisDiagnostics] = None

    @property
    def template_hint(self) -> str:
        if self.diagnostics is None:
            return ""
        return str(self.diagnostics.template_hint)

    @property
    def analysis_confidence(self) -> float:
        if self.diagnostics is None:
            return 1.0
        return float(self.diagnostics.analysis_confidence)

    @property
    def feature_graph(self):
        if self.diagnostics is None:
            return None
        return self.diagnostics.feature_graph

    @property
    def feature_point_candidates(self) -> List[int]:
        if self.diagnostics is None:
            return []
        return list(self.diagnostics.feature_point_candidates)

    @property
    def feature_edge_candidates(self) -> List[Tuple[int, int]]:
        if self.diagnostics is None:
            return []
        return list(self.diagnostics.feature_edge_candidates)

    @property
    def boundary_half_edges(self) -> List[int]:
        if self.diagnostics is None:
            return []
        return list(self.diagnostics.boundary_half_edges)

    @property
    def neighborhood_face_indices(self) -> List[int]:
        if self.diagnostics is None:
            return []
        return list(self.diagnostics.neighborhood_face_indices)

    @property
    def surface_patch_labels(self) -> Dict[int, int]:
        if self.diagnostics is None:
            return {}
        return dict(self.diagnostics.surface_patch_labels)

    @property
    def patch_face_indices(self) -> Dict[int, List[int]]:
        if self.diagnostics is None:
            return {}
        return dict(self.diagnostics.patch_face_indices)
