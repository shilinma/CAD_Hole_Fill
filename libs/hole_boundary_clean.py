#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
孔洞边界清理（Boundary cleaning / tooth face removal）。

三角网格中，若某三角形有至少两条边落在网格开边界上，则该面为「齿状面」
（tooth face），会向内伸出使孔洞边界呈锯齿状。迭代删除这些面直至不再出现。

契约：输出 mesh 视为全新带孔输入；**仅删面，不重排/合并顶点**。
孔洞环顶点 id 与坐标保持不变，由下游 ``HoleDetector`` 重新标定孔环顺序。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np


def _edge_key(u: int, v: int) -> Tuple[int, int]:
    u, v = int(u), int(v)
    return (u, v) if u < v else (v, u)


def _build_edge_to_faces(faces: np.ndarray) -> Dict[Tuple[int, int], List[int]]:
    out: Dict[Tuple[int, int], List[int]] = {}
    for fi in range(int(faces.shape[0])):
        a, b, c = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
        for u, v in ((a, b), (b, c), (c, a)):
            k = _edge_key(u, v)
            out.setdefault(k, []).append(fi)
    return out


def _boundary_edge_keys(edge2faces: Dict[Tuple[int, int], List[int]]) -> Set[Tuple[int, int]]:
    return {k for k, fl in edge2faces.items() if len(fl) == 1}


def _face_boundary_edge_count(
    face: np.ndarray, boundary_keys: Set[Tuple[int, int]]
) -> int:
    a, b, c = int(face[0]), int(face[1]), int(face[2])
    n = 0
    for u, v in ((a, b), (b, c), (c, a)):
        if _edge_key(u, v) in boundary_keys:
            n += 1
    return n


def _faces_incident_to_boundary_edges(
    boundary_keys: Set[Tuple[int, int]],
    edge2faces: Dict[Tuple[int, int], List[int]],
) -> Set[int]:
    incident: Set[int] = set()
    for ek in boundary_keys:
        for fi in edge2faces.get(ek, []):
            incident.add(int(fi))
    return incident


@dataclass
class HoleBoundaryCleanResult:
    """清理统计。"""

    vertices: np.ndarray
    faces: np.ndarray
    iterations: int
    faces_removed_total: int


def clean_hole_boundary_tooth_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    max_iterations: int = 10_000,
) -> HoleBoundaryCleanResult:
    """
    迭代删除齿状三角形，直到不存在「至少两条边在开边界上」的三角形。

    不 compact、不合并顶点：避免 119→118 等 id 语义漂移；孔环由下游重新检测。
    """
    v = np.asarray(vertices, dtype=np.float64, order="C")
    f = np.asarray(faces, dtype=np.int64, order="C")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("faces 需要为 (M, 3) 的三角形索引")

    total_removed = 0
    it = 0
    while it < max_iterations:
        it += 1
        edge2faces = _build_edge_to_faces(f)
        bkeys = _boundary_edge_keys(edge2faces)
        if not bkeys:
            break

        candidates = _faces_incident_to_boundary_edges(bkeys, edge2faces)
        tooth: List[int] = []
        for fi in candidates:
            bc = _face_boundary_edge_count(f[int(fi)], bkeys)
            if bc >= 2:
                tooth.append(int(fi))

        if not tooth:
            break

        tooth_unique = sorted(set(tooth), reverse=True)
        total_removed += len(tooth_unique)
        keep = np.ones(int(f.shape[0]), dtype=bool)
        keep[tooth_unique] = False
        f = f[keep]

    return HoleBoundaryCleanResult(
        vertices=v,
        faces=f,
        iterations=it,
        faces_removed_total=total_removed,
    )
