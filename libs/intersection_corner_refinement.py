#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
两解析面交线：在孔边弧上按各点 d(v)（到交线距离）划分候选，再沿弧做「有隙 / 扫描」以更新 sharp edge 的端点。

``d``：平面-平面为点到两平面交直线的欧氏距；其他类型为 p 到 ``project_point_to_surface_pair`` 的残差长。
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .surface_fitting import SurfaceFit, project_point_to_surface_pair
from .surface_intersections import _plane_plane_line


def point_distance_to_pair_intersection(
    p: np.ndarray, fit_a: SurfaceFit, fit_b: SurfaceFit
) -> float:
    """标量 d(v)。"""
    pt = np.asarray(p, dtype=np.float64).reshape(3)
    if fit_a.surface_type == "plane" and fit_b.surface_type == "plane":
        line = _plane_plane_line(fit_a, fit_b)
        if line is None:
            return float("inf")
        lp = np.asarray(line[0], dtype=np.float64)
        ld = np.asarray(line[1], dtype=np.float64)
        ln = float(np.linalg.norm(ld))
        if ln < 1e-15:
            return float("inf")
        ld = ld / ln
        t = float(np.dot(pt - lp, ld))
        proj = lp + t * ld
        return float(np.linalg.norm(pt - proj))
    q = project_point_to_surface_pair(fit_a, fit_b, pt, iterations=6)
    return float(np.linalg.norm(pt - q))


def _d_along_arc(
    vertices: np.ndarray,
    seq: Sequence[int],
    fit_a: SurfaceFit,
    fit_b: SurfaceFit,
) -> np.ndarray:
    return np.array(
        [
            point_distance_to_pair_intersection(vertices[int(vi)], fit_a, fit_b)
            for vi in seq
        ],
        dtype=np.float64,
    )


def _candidate_indices(
    d: np.ndarray, *, d_ratio: float, min_candidates: int
) -> List[int]:
    n = int(d.shape[0])
    d_min = float(np.min(d))
    d_max = float(np.max(d))
    span = max(d_max - d_min, 1e-15)
    tau = d_min + float(d_ratio) * span
    C = [i for i in range(n) if d[i] <= tau + 1e-12]
    if len(C) < int(min_candidates) and n >= 1:
        m = int(np.argmin(d))
        C = sorted(set(C + [m]))
    if not C:
        C = [0, n - 1] if n > 1 else [0]
    return sorted(C)


def refine_sharp_edge_arc_endpoints(
    vertices: np.ndarray,
    seq_vertex_ids: Sequence[int],
    fit_a: SurfaceFit,
    fit_b: SurfaceFit,
    *,
    d_ratio: float = 0.38,
    min_candidates: int = 2,
) -> Tuple[int, int, str]:
    """
    在孔边**开弧**的顶点列 ``seq``（与 BoundaryArc 顺序一致）上，返回 ``recover`` 用的一对顶点索引
    ``(i_start, i_end)`` 与原因标签 ``tag``。

    逻辑概要：
    - 用 ``d_ratio`` 从 ``d`` 得候选下标列 ``C``（全弧偏近交线则判 ``single_patch_chain`` 仍用弧端）；
    - 在 **C 的升序** 中找**第一对相邻下标**在弧上**不**相差 1 → 以该对为新角点（``refined_first_gap``）；
    - 否则用 ``W = sorted({0} ∪ C)`` 在弧上自 0 起找**首隙**（``refined_scan``）；
    - 若 C 为连续块但未满整弧 → 用块两端（``refined_ends_tight``）；
    - 否则回退为弧拓扑端点 ``(seq[0], seq[-1])``。
    """
    seq = [int(x) for x in seq_vertex_ids]
    n = len(seq)
    if n < 2:
        return (int(seq[0]), int(seq[0]), "degenerate")
    d = _d_along_arc(vertices, seq, fit_a, fit_b)
    C = _candidate_indices(
        d, d_ratio=float(d_ratio), min_candidates=max(2, int(min_candidates))
    )

    # 整段弧都进候选且即 0..n-1：沿弧全部贴近交线
    if len(C) == n and C == list(range(n)):
        return int(seq[0]), int(seq[-1]), "single_patch_chain"

    # 候选内第一处**下标不连续**（在弧上无公共边 = 不「依次相连」）
    for k in range(len(C) - 1):
        if C[k + 1] - C[k] > 1:
            return int(seq[C[k]]), int(seq[C[k + 1]]), "refined_first_gap"

    # 自弧首 0 与候选并：在 W 中找**首处**间隙
    W = sorted(set([0] + C))
    for k in range(len(W) - 1):
        if W[k + 1] - W[k] > 1:
            return int(seq[W[k]]), int(seq[W[k + 1]]), "refined_scan"

    # 候选为连续块、但未盖住全部 —— 用该块在弧上端点
    if len(C) >= 2 and (C[0] > 0 or C[-1] < n - 1):
        return int(seq[C[0]]), int(seq[C[-1]]), "refined_ends_tight"

    return int(seq[0]), int(seq[-1]), "fallback_topology"
