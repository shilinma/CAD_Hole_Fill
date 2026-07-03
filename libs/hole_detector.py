#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
孔洞检测（CAD_Hole_Fill）：基于半边数据结构找出模型上所有孔洞，并用颜色标注边界点。

半边结构：
- HalfEdge: origin_vertex, next, twin, face
- 边界半边：twin=-1 或 face=-1
- 沿边界 next 形成闭环
"""

import numpy as np
import trimesh
from typing import List, Tuple, Dict, Set, Optional, Callable
from dataclasses import dataclass, field

# ==================== 半边数据结构 ====================

@dataclass
class HalfEdge:
    """半边：从 origin 指向某顶点，next 为同面内下一条，twin 为对向半边，prev 为边界链上的前一条（链接后填充）"""
    origin: int
    next: int = -1
    prev: int = -1
    twin: int = -1
    face: int = -1


@dataclass
class HalfEdgeMesh:
    """半边网格"""
    vertices: np.ndarray
    half_edges: List[HalfEdge] = field(default_factory=list)
    faces: List[List[int]] = field(default_factory=list)

    def num_vertices(self) -> int:
        return len(self.vertices)

    def num_half_edges(self) -> int:
        return len(self.half_edges)

    def num_faces(self) -> int:
        return len(self.faces)


def build_half_edge_mesh(vertices: np.ndarray, faces: np.ndarray) -> HalfEdgeMesh:
    """
    从三角网格构建半边结构。
    每条边拆成两条半边，twin 互为对向；边界边 twin=-1。
    """
    mesh = HalfEdgeMesh(vertices=vertices.copy())
    he_list: List[HalfEdge] = []
    face_to_he: List[int] = []
    edge_to_he: Dict[Tuple[int, int], int] = {}

    for fi, f in enumerate(faces):
        v0, v1, v2 = int(f[0]), int(f[1]), int(f[2])
        he0 = HalfEdge(origin=v0, face=fi)
        he1 = HalfEdge(origin=v1, face=fi)
        he2 = HalfEdge(origin=v2, face=fi)
        idx0 = len(he_list)
        idx1 = idx0 + 1
        idx2 = idx0 + 2
        he0.next = idx1
        he1.next = idx2
        he2.next = idx0
        he_list.extend([he0, he1, he2])
        face_to_he.append(idx0)

        for (a, b), he_idx in [((v0, v1), idx0), ((v1, v2), idx1), ((v2, v0), idx2)]:
            key = (min(a, b), max(a, b))
            if key not in edge_to_he:
                edge_to_he[key] = he_idx
            else:
                other = edge_to_he[key]
                he_list[he_idx].twin = other
                he_list[other].twin = he_idx

    mesh.half_edges = he_list
    mesh.faces = faces.tolist()
    return mesh

# ==================== 非流形顶点：局部投影与辅助线段 ====================

def _mesh_diagonal(vertices: np.ndarray) -> float:
    bmin = vertices.min(axis=0)
    bmax = vertices.max(axis=0)
    return float(np.linalg.norm(bmax - bmin))


def _vertex_normals_area_weighted(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """按三角形面积加权的顶点法向（用于非流形顶点处局部坐标 z 轴）。"""
    n_v = len(vertices)
    acc = np.zeros((n_v, 3), dtype=np.float64)
    for fi in range(len(faces)):
        i0, i1, i2 = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
        p0, p1, p2 = vertices[i0], vertices[i1], vertices[i2]
        cr = np.cross(p1 - p0, p2 - p0)
        acc[i0] += cr
        acc[i1] += cr
        acc[i2] += cr
    ln = np.linalg.norm(acc, axis=1, keepdims=True)
    ln = np.where(ln < 1e-15, 1.0, ln)
    return acc / ln


def _local_frame_R(n: np.ndarray) -> np.ndarray:
    """
    以 n 为 P_z，构造正交基；R 的行向量为 [P_x, P_y, P_z]^T，
    对向量 p，局部坐标为 R @ p（与论文中 V0xy 平面投影一致）。
    """
    Pz = n / (np.linalg.norm(n) + 1e-15)
    ex = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    ey = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if abs(np.dot(Pz, ex)) > 0.9:
        Py = np.cross(Pz, ey)
    else:
        Py = np.cross(Pz, ex)
    if np.linalg.norm(Py) < 1e-12:
        Py = np.cross(Pz, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    Py = Py / (np.linalg.norm(Py) + 1e-15)
    Px = np.cross(Py, Pz)
    Px = Px / (np.linalg.norm(Px) + 1e-15)
    return np.stack([Px, Py, Pz], axis=0)


def _project_to_v0xy(R: np.ndarray, origin: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """3D 点 -> V0xy 平面 2D 坐标（取 R@(p-origin) 的前两维）。"""
    rel = pts - origin.reshape(1, 3)
    loc = (R @ rel.T).T
    return loc[:, :2].astype(np.float64, copy=False)


def _orient_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment_2d(a: np.ndarray, b: np.ndarray, p: np.ndarray, eps: float) -> bool:
    if abs(_orient_2d(a, b, p)) > eps * (np.linalg.norm(b - a) + 1.0):
        return False
    return (
        min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
    )


def _segments_intersect_proper_or_touch(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, eps: float
) -> bool:
    """线段 ab 与 cd 相交（含端点接触、共线重叠；与三角形边重合视为相交）。"""
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
    if s1 == 0 and _on_segment_2d(a, b, c, eps):
        return True
    if s2 == 0 and _on_segment_2d(a, b, d, eps):
        return True
    if s3 == 0 and _on_segment_2d(c, d, a, eps):
        return True
    if s4 == 0 and _on_segment_2d(c, d, b, eps):
        return True
    return False


def _triangle_area_2d(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
    return 0.5 * abs(_orient_2d(p, q, r))


def _segment_triangle_intersect_2d(
    a: np.ndarray,
    b: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    eps: float,
) -> bool:
    """
    辅助线段 ab 与三角形 pqr 在 2D 中是否“相交”（含与边重合）。
    退化三角形（面积过小）忽略。
    """
    if _triangle_area_2d(p, q, r) < eps * eps * 0.5:
        return False
    edges = ((p, q), (q, r), (r, p))
    for e0, e1 in edges:
        if _segments_intersect_proper_or_touch(a, b, e0, e1, eps):
            return True
    # 端点在三角形内部（开线段穿过内部）
    o1 = _orient_2d(p, q, r)
    if abs(o1) < eps * eps:
        return False

    def inside_tri(t: np.ndarray) -> bool:
        oa = _orient_2d(p, q, t)
        ob = _orient_2d(q, r, t)
        oc = _orient_2d(r, p, t)
        sa, sb, sc = np.sign(oa), np.sign(ob), np.sign(oc)
        if abs(oa) < eps and _on_segment_2d(p, q, t, eps):
            return False
        if abs(ob) < eps and _on_segment_2d(q, r, t, eps):
            return False
        if abs(oc) < eps and _on_segment_2d(r, p, t, eps):
            return False
        return (sa >= 0 and sb >= 0 and sc >= 0) or (sa <= 0 and sb <= 0 and sc <= 0)

    # 收缩端点避免与端点邻接三角形误报；仍检测与边的共线重叠（上已覆盖）
    ab = b - a
    lab = float(np.linalg.norm(ab))
    if lab < eps:
        return False
    tdir = ab / lab
    shrink = min(eps * 0.5, lab * 1e-6)
    if lab <= 2 * shrink:
        return inside_tri((a + b) * 0.5)
    aa = a + tdir * shrink
    bb = b - tdir * shrink
    mid = 0.5 * (aa + bb)
    if inside_tri(mid):
        return True
    return False


def _point_in_triangle_barycentric(
    x: np.ndarray, p: np.ndarray, q: np.ndarray, r: np.ndarray, eps: float
) -> bool:
    """x 在三角形 pqr 内（含边），重心坐标非负。"""
    v0 = q - p
    v1 = r - p
    v2 = x - p
    d00 = float(np.dot(v0, v0))
    d01 = float(np.dot(v0, v1))
    d11 = float(np.dot(v1, v1))
    d20 = float(np.dot(v2, v0))
    d21 = float(np.dot(v2, v1))
    denom = d00 * d11 - d01 * d01
    if abs(denom) < eps * eps * eps:
        return False
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return u >= -eps and v >= -eps and w >= -eps


def _segment_triangle_intersect_3d(
    a: np.ndarray,
    b: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    eps: float,
) -> bool:
    """
    3D 线段 ab 与三角形 pqr 是否相交（含与边重合、顶点接触）。
    先求线段与三角形所在平面的交点，共面时投影到 2D。
    """
    e1 = q - p
    e2 = r - p
    n = np.cross(e1, e2)
    ln = float(np.linalg.norm(n))
    if ln < eps * eps:
        return False
    n = n / ln
    ab = b - a
    denom = float(np.dot(n, ab))
    if abs(denom) < eps * 1e-9:
        if abs(float(np.dot(n, a - p))) > eps * 1e-6:
            return False
        ax = abs(n[0])
        ay = abs(n[1])
        az = abs(n[2])
        if ax >= ay and ax >= az:
            a2, b2 = np.array([a[1], a[2]]), np.array([b[1], b[2]])
            p2, q2, r2 = np.array([p[1], p[2]]), np.array([q[1], q[2]]), np.array([r[1], r[2]])
        elif ay >= ax and ay >= az:
            a2, b2 = np.array([a[0], a[2]]), np.array([b[0], b[2]])
            p2, q2, r2 = np.array([p[0], p[2]]), np.array([q[0], q[2]]), np.array([r[0], r[2]])
        else:
            a2, b2 = np.array([a[0], a[1]]), np.array([b[0], b[1]])
            p2, q2, r2 = np.array([p[0], p[1]]), np.array([q[0], q[1]]), np.array([r[0], r[1]])
        return _segment_triangle_intersect_2d(a2, b2, p2, q2, r2, eps)

    t = float(np.dot(n, p - a) / denom)
    if t < -eps or t > 1.0 + eps:
        return False
    x = a + t * ab
    return _point_in_triangle_barycentric(x, p, q, r, eps)


def _auxiliary_segment_hits_mesh_3d(
    a: np.ndarray,
    b: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    skip_face_indices: Optional[Set[int]],
    eps: float,
) -> bool:
    """3D 辅助线段是否与网格三角形相交（可跳过与边界边重合的面以减少端点误报）。"""
    skip_face_indices = skip_face_indices or set()
    # 端点微收缩，避免仅顶点接触被当成“穿过”
    ab = b - a
    lab = float(np.linalg.norm(ab))
    if lab < eps:
        return False
    tdir = ab / lab
    shrink = max(eps * 2.0, 1e-9 * lab)
    if lab <= 2 * shrink:
        aa, bb = a, b
    else:
        aa = a + tdir * shrink
        bb = b - tdir * shrink
    for fi in range(len(faces)):
        if fi in skip_face_indices:
            continue
        i0, i1, i2 = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
        p, q, r = vertices[i0], vertices[i1], vertices[i2]
        if _segment_triangle_intersect_3d(aa, bb, p, q, r, eps):
            return True
    return False


def _boundary_face_indices_incident_to_edge(faces: np.ndarray, u: int, v: int) -> Set[int]:
    """包含无向边 (u,v) 的三角形面索引（边界边仅一个面）。"""
    out: Set[int] = set()
    for fi in range(len(faces)):
        a, b, c = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
        s = {a, b, c}
        if u in s and v in s:
            out.add(fi)
    return out


def _compute_boundary_tip_before_link(mesh: HalfEdgeMesh) -> np.ndarray:
    """
    在覆盖 next 之前，记录每条边界半边对应的几何边终点（面内 next 的 origin）。
    """
    tips = np.full(len(mesh.half_edges), -1, dtype=np.int64)
    for i, he in enumerate(mesh.half_edges):
        if he.twin == -1:
            tips[i] = mesh.half_edges[he.next].origin
    return tips


def _boundary_undirected_edges(mesh: HalfEdgeMesh, boundary_tip: np.ndarray) -> Set[Tuple[int, int]]:
    """边界无向边集合，用于分叉顶点处的拓扑配对。"""
    edges: Set[Tuple[int, int]] = set()
    for i, he in enumerate(mesh.half_edges):
        if he.twin == -1:
            u, v = int(he.origin), int(boundary_tip[i])
            if u > v:
                u, v = v, u
            edges.add((u, v))
    return edges


def _uf_find_without_vertex(
    n_vertices: int, boundary_edges: Set[Tuple[int, int]], exclude_v: int
):
    """
    在「删除」顶点 exclude_v 及其关联边界边后，对剩余边界边做并查集。
    同一简单孔在分叉点两侧的两个邻点会落在同一集合中。
    返回 find 函数 find(i) -> 根。
    """
    parent = np.arange(n_vertices, dtype=np.int64)

    def find(x: int) -> int:
        x = int(x)
        root = x
        while parent[root] != root:
            root = int(parent[root])
        while parent[x] != x:
            px = int(parent[x])
            parent[x] = root
            x = px
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for u, v in boundary_edges:
        if u == exclude_v or v == exclude_v:
            continue
        union(u, v)

    return find


def _sort_candidates_by_left_turn(
    v0: np.ndarray,
    v1: np.ndarray,
    cand_indices: List[int],
    tip: np.ndarray,
    verts_xy: np.ndarray,
) -> List[int]:
    """
    在 V0xy 投影上按“左转优先”（叉积）排序候选半边，利于 >180° 凹角与多分支时
    与曲面外侧一致的几何顺序；再交给相交检测最终判定。
    """
    d_in = v1 - v0
    lin = float(np.linalg.norm(d_in))
    if lin < 1e-15:
        return list(cand_indices)
    d_in = d_in / lin
    scored: List[Tuple[float, int]] = []
    for ci in cand_indices:
        v2 = verts_xy[tip[ci]]
        d_out = v2 - v1
        cr = d_in[0] * d_out[1] - d_in[1] * d_out[0]
        scored.append((-cr, ci))
    scored.sort(key=lambda x: (x[0], x[1]))
    return [c for _, c in scored]


def _build_out_boundary_map(mesh: HalfEdgeMesh) -> Dict[int, List[int]]:
    """按起点收集所有边界半边索引。"""
    from collections import defaultdict

    out_boundary: Dict[int, List[int]] = defaultdict(list)
    for i, he in enumerate(mesh.half_edges):
        if he.twin == -1:
            out_boundary[he.origin].append(i)
    return out_boundary


def _choose_boundary_candidate_by_topology(
    v0: int,
    junction: int,
    candidates: List[int],
    boundary_tip: np.ndarray,
    n_vertices: int,
    boundary_edges: Set[Tuple[int, int]],
    uf_cache: Dict[int, Callable[[int], int]],
) -> int:
    """
    在非流形分叉点优先做拓扑配对。
    若删除分叉点后，输入边起点 v0 与候选输出边终点仍在同一连通分量，
    则认为它们属于同一简单孔边界。
    """
    if junction not in uf_cache:
        uf_cache[junction] = _uf_find_without_vertex(n_vertices, boundary_edges, junction)

    uf_find = uf_cache[junction]
    comp_in = uf_find(v0)
    for cand in candidates:
        if uf_find(int(boundary_tip[cand])) == comp_in:
            return cand
    return -1


def _choose_boundary_candidate_by_geometry(
    v0: int,
    v1: int,
    candidates: List[int],
    boundary_tip: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    vnormals: np.ndarray,
    eps: float,
    eps_len: float,
) -> int:
    """
    当拓扑配对无法唯一确定时，退化到局部投影 + 辅助线段相交检测。
    若没有完全无交的候选，则保留左转优先排序中的第一个候选。
    """
    n = vnormals[v1]
    R = _local_frame_R(n)
    origin = vertices[v1]
    verts_xy = _project_to_v0xy(R, origin, vertices)
    p0 = verts_xy[v0]
    p1 = verts_xy[v1]

    if float(np.linalg.norm(p1 - p0)) < eps_len:
        return candidates[0]

    skip_faces_uv = _boundary_face_indices_incident_to_edge(faces, v0, v1)
    ordered = _sort_candidates_by_left_turn(p0, p1, candidates, boundary_tip, verts_xy)
    for cand in ordered:
        v2 = int(boundary_tip[cand])
        if v2 == v0:
            continue
        if float(np.linalg.norm(vertices[v2] - vertices[v0])) < eps_len:
            continue

        skip = set(skip_faces_uv)
        skip |= _boundary_face_indices_incident_to_edge(faces, v1, v2)
        if not _auxiliary_segment_hits_mesh_3d(
            vertices[v0], vertices[v2], vertices, faces, skip, eps
        ):
            return cand

    return ordered[0]


def _set_boundary_prev_links(mesh: HalfEdgeMesh) -> None:
    """根据已建立的边界 next 指针回填 prev。"""
    for i, he in enumerate(mesh.half_edges):
        if he.twin != -1:
            continue
        nxt = he.next
        if nxt >= 0 and mesh.half_edges[nxt].twin == -1:
            mesh.half_edges[nxt].prev = i


def _link_boundary_half_edges(mesh: HalfEdgeMesh, faces: np.ndarray) -> None:
    """
    为边界半边设置 next，使沿边界形成简单孔闭环。

    非流形边界顶点（多条出边候选）处：
    1) 拓扑配对（主）：在边界子图中去掉该顶点及其关联边后，当前半边起点 v0 与候选终点
       n_out 若在同一连通分量，则属于同一简单孔边界（沿孔绕行不经过该分叉点）。
    2) 辅助线段 + 3D/2D 相交（回退）：论文中的局部投影与无相交判定，用于拓扑无法唯一时。
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    boundary_tip = _compute_boundary_tip_before_link(mesh)
    boundary_edges = _boundary_undirected_edges(mesh, boundary_tip)
    n_v = len(vertices)
    diag = _mesh_diagonal(vertices)
    eps = max(1e-10, 1e-12 * diag)
    eps_len = max(1e-12, 1e-9 * diag)

    vnormals = _vertex_normals_area_weighted(vertices, faces)
    uf_at_junction: Dict[int, Callable[[int], int]] = {}
    out_boundary = _build_out_boundary_map(mesh)

    for i, he in enumerate(mesh.half_edges):
        if he.twin != -1:
            continue

        v0 = int(he.origin)
        v1 = int(boundary_tip[i])
        candidates = out_boundary.get(v1, [])

        if len(candidates) == 0:
            he.next = -1
            continue
        if len(candidates) == 1:
            he.next = candidates[0]
            continue

        chosen = _choose_boundary_candidate_by_topology(
            v0,
            v1,
            candidates,
            boundary_tip,
            n_v,
            boundary_edges,
            uf_at_junction,
        )
        if chosen < 0:
            chosen = _choose_boundary_candidate_by_geometry(
                v0,
                v1,
                candidates,
                boundary_tip,
                vertices,
                faces,
                vnormals,
                eps,
                eps_len,
            )
        he.next = chosen

    _set_boundary_prev_links(mesh)


def get_boundary_loops_half_edge(mesh: HalfEdgeMesh) -> List[List[int]]:
    """
    基于半边结构找出所有边界环（孔洞）。
    返回每个环的顶点索引列表（按环顺序）。
    """
    used = set()
    loops = []
    for i, he in enumerate(mesh.half_edges):
        if he.twin != -1 or i in used:
            continue
        loop = []
        cur = i
        while True:
            used.add(cur)
            h = mesh.half_edges[cur]
            loop.append(h.origin)
            nxt = h.next
            if nxt == -1:
                break
            cur = nxt
            if cur == i:
                break
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def detect_holes_half_edge(vertices: np.ndarray, faces: np.ndarray) -> Tuple[HalfEdgeMesh, List[List[int]]]:
    """
    使用半边结构检测所有孔洞。
    返回: (半边网格, 边界环列表)
    """
    mesh = build_half_edge_mesh(vertices, faces)
    _link_boundary_half_edges(mesh, np.asarray(faces, dtype=np.int64))
    loops = get_boundary_loops_half_edge(mesh)
    return mesh, loops

# ==================== 与 trimesh 集成 ====================

def _trace_boundary_loops_from_edges(boundary_edges: np.ndarray) -> List[List[int]]:
    """从无向边界边集合追踪边界环，供 trimesh 回退路径使用。"""
    from collections import defaultdict

    adj = defaultdict(list)
    for a, b in boundary_edges:
        adj[int(a)].append(int(b))

    loops = []
    used_edges = set()
    for a, b in boundary_edges:
        a, b = int(a), int(b)
        key = (min(a, b), max(a, b))
        if key in used_edges:
            continue

        loop = []
        cur, prev = a, None
        while True:
            loop.append(cur)
            next_cands = [x for x in adj[cur] if x != prev]
            if not next_cands:
                break
            nxt = next_cands[0]
            used_edges.add((min(cur, nxt), max(cur, nxt)))
            prev, cur = cur, nxt
            if cur == a:
                break

        if len(loop) >= 3:
            loops.append(loop)
    return loops


def _detect_holes_trimesh_fallback(mesh: trimesh.Trimesh) -> List[List[int]]:
    """半边路径失败后的简化回退实现。"""
    try:
        idx = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)
        boundary_edges = mesh.edges[idx]
    except Exception:
        return []
    return _trace_boundary_loops_from_edges(boundary_edges)


def detect_holes(mesh: trimesh.Trimesh) -> List[List[int]]:
    """
    检测网格上所有孔洞边界环。
    优先使用半边结构，失败时回退到 trimesh.group_rows。
    """
    V = np.array(mesh.vertices, dtype=np.float64)
    F = np.array(mesh.faces, dtype=np.int32)
    try:
        _, loops = detect_holes_half_edge(V, F)
        return loops
    except Exception:
        return _detect_holes_trimesh_fallback(mesh)


def get_all_boundary_vertices(loops: List[List[int]]) -> Set[int]:
    """所有边界环中的顶点集合"""
    return set(v for loop in loops for v in loop)


class HoleDetector:
    """
    孔洞检测器：封装半边结构检测与 trimesh 回退逻辑。

    用法::

        det = HoleDetector()
        loops = det.detect(mesh)  # List[List[int]]，每个环为有序顶点索引
        he_mesh, loops = det.detect_with_half_edge(V, F)
    """

    def detect(self, mesh: trimesh.Trimesh) -> List[List[int]]:
        return detect_holes(mesh)

    def detect_with_half_edge(
        self, vertices: np.ndarray, faces: np.ndarray
    ) -> Tuple[HalfEdgeMesh, List[List[int]]]:
        return detect_holes_half_edge(vertices, faces)

