import igraph as ig
import math
import sqlite3
import statistics
from collections import Counter
from typing import Dict, Any, List, Optional

try:
    import numpy as np
    import scipy.sparse as sp
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

class SSNAnalyzer:
    """
    SSN 网络特征描述模块，用于计算图的各种拓扑指标。
    """
    def __init__(self, graph: ig.Graph):
        self.graph = graph
        # 记录当前生效的权重属性名。如果图中已有 'weight'，则默认使用它。
        self.active_weight = 'weight' if 'weight' in self.graph.edge_attributes() else None

    def basic_stats(self) -> Dict[str, Any]:
        """
        计算基础统计指标，并包含更丰富的网络特征。
        """
        if self.graph.vcount() == 0:
            return {
                "nodes": 0, 
                "edges": 0,
                "density": 0.0,
                "is_connected": False,
                "components": 0,
                "avg_clustering": 0.0,
                "avg_degree": 0.0,
                "max_degree": 0,
                "min_degree": 0,
                "lcc_size": 0,
                "lcc_percentage": 0.0,
                "total_weight": 0.0
            }
            
        components = self.graph.connected_components()
        lcc = components.giant()
        lcc_size = lcc.vcount()
        
        degrees = self.graph.degree()
        
        stats = {
            "nodes": self.graph.vcount(),
            "edges": self.graph.ecount(),
            "density": self.graph.density(),
            "is_connected": self.graph.is_connected(),
            "components": len(components),
            "avg_clustering": self.graph.transitivity_avglocal_undirected(mode="zero"),
            "avg_degree": sum(degrees) / len(degrees) if degrees else 0,
            "max_degree": max(degrees) if degrees else 0,
            "min_degree": min(degrees) if degrees else 0,
            "lcc_size": lcc_size,
            "lcc_percentage": (lcc_size / self.graph.vcount()) * 100 if self.graph.vcount() > 0 else 0
        }

        # 添加权重相关的统计
        if self.active_weight and self.active_weight in self.graph.edge_attributes():
            weights = self.graph.es[self.active_weight]
            stats["total_weight"] = sum(weights)
            stats["avg_weight"] = sum(weights) / len(weights) if weights else 0
            stats["min_weight"] = min(weights) if weights else 0
            stats["max_weight"] = max(weights) if weights else 0
            stats["sd_weight"] = statistics.stdev(weights) if len(weights) >= 2 else 0.0
        
        return stats

    def get_connected_components(self) -> ig.VertexClustering:
        """
        获取连通分量。
        """
        return self.graph.connected_components()

    def local_clustering_coefficient(self) -> List[float]:
        """
        计算每个节点的局部聚集系数。
        """
        return self.graph.transitivity_local_undirected(mode="zero")

    def max_flow(self, source: str, target: str, capacity: Optional[str] = None) -> float:
        """
        计算两点间的最大流。
        :param source: 源节点名称
        :param target: 目标节点名称
        :param capacity: 边属性名称，作为容量。如果为 None，则尝试使用 self.active_weight。
        """
        s_idx = self.graph.vs.find(name=source).index
        t_idx = self.graph.vs.find(name=target).index
        
        cap_attr = capacity or self.active_weight
        cap = self.graph.es[cap_attr] if cap_attr and cap_attr in self.graph.es.attributes() else None
        return self.graph.maxflow_value(s_idx, t_idx, capacity=cap)

    def min_cut(self, source: str, target: str, capacity: Optional[str] = None) -> float:
        """
        计算两点间的最小切割。
        :param source: 源节点名称
        :param target: 目标节点名称
        :param capacity: 边属性名称，作为容量。如果为 None，则尝试使用 self.active_weight。
        """
        s_idx = self.graph.vs.find(name=source).index
        t_idx = self.graph.vs.find(name=target).index
        
        cap_attr = capacity or self.active_weight
        cap = self.graph.es[cap_attr] if cap_attr and cap_attr in self.graph.es.attributes() else None
        return self.graph.mincut_value(s_idx, t_idx, capacity=cap)

    def modularity(self, membership: List[int], weights: Optional[str] = None) -> float:
        """
        计算给定划分的模块度。
        :param membership: 节点所属社区的列表
        :param weights: 边权重属性名称。如果为 None，则尝试使用 self.active_weight。
        """
        w_attr = weights or self.active_weight
        w = self.graph.es[w_attr] if w_attr and w_attr in self.graph.es.attributes() else None
        return self.graph.modularity(membership, weights=w)

    def inter_cluster_edge_ratio(
        self,
        clustering: ig.VertexClustering,
        weights: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        计算跨 cluster 的边比例，用于评估聚类质量。
        比例越低，说明社区内部连接越紧密，聚类效果越好。

        :param clustering: igraph.VertexClustering 聚类结果
        :param weights: 边权重属性名（可选）。如果为 None，则尝试使用 self.active_weight。
        :return: 包含边数统计和比例的字典
        """
        membership = clustering.membership
        total_edges = self.graph.ecount()
        inter_edges = sum(
            1 for e in self.graph.es
            if membership[e.source] != membership[e.target]
        )
        intra_edges = total_edges - inter_edges

        result: Dict[str, Any] = {
            "num_clusters": len(set(membership)),
            "total_edges": total_edges,
            "intra_cluster_edges": intra_edges,
            "inter_cluster_edges": inter_edges,
            "inter_cluster_ratio": inter_edges / total_edges if total_edges > 0 else 0.0,
        }

        w_attr = weights or self.active_weight
        if w_attr and w_attr in self.graph.edge_attributes():
            total_weight = sum(self.graph.es[w_attr])
            inter_weight = sum(
                e[w_attr] for e in self.graph.es
                if membership[e.source] != membership[e.target]
            )
            result["inter_cluster_weight_ratio"] = inter_weight / total_weight if total_weight > 0 else 0.0

        return result

    def apply_jaccard_weighting(self, base_weight: str = 'weight', new_attr: str = 'jaccard_weight'):
        """
        对边的权重进行 Jaccard 系数加权。
        新权重 = 原始权重 * (两个节点邻居的交集 / 两个节点邻居的并集)
        
        :param base_weight: 原始权重的属性名。
        :param new_attr: 加权后权重的属性名。
        """
        if base_weight not in self.graph.edge_attributes():
            old_weights = np.ones(self.graph.ecount()) if _NUMPY_AVAILABLE else [1.0] * self.graph.ecount()
        else:
            old_weights = self.graph.es[base_weight]

        if _NUMPY_AVAILABLE:
            jaccard_weights = self._apply_jaccard_weighting_fast(old_weights)
        else:
            jaccard_weights = self._apply_jaccard_weighting_fallback(old_weights)

        self.graph.es[new_attr] = jaccard_weights
        self.active_weight = new_attr
        print(f"已完成 Jaccard 加权，新权重已保存至属性: {new_attr}，后续分析将默认使用此权重。")

    def _apply_jaccard_weighting_fast(self, old_weights) -> list:
        """
        使用 scipy 稀疏矩阵批量计算 Jaccard 系数，大幅提升大图性能。
        对于边 (u,v)，Jaccard = |N(u)-{v} ∩ N(v)-{u}| / |N(u)-{v} ∪ N(v)-{u}|。
        
        利用 A²[u,v] = |N(u) ∩ N(v)|（A 无自环，故不含端点自身），
        union = (deg_u - 1) + (deg_v - 1) - intersection，其中 -1 是排除对方端点。
        """
        n = self.graph.vcount()
        edges = self.graph.get_edgelist()
        if not edges:
            return []

        us = np.array([e[0] for e in edges], dtype=np.int32)
        vs = np.array([e[1] for e in edges], dtype=np.int32)

        # 构建无向邻接矩阵（对称，无自环）
        data = np.ones(len(edges) * 2, dtype=np.float32)
        row = np.concatenate([us, vs])
        col = np.concatenate([vs, us])
        A = sp.csr_matrix((data, (row, col)), shape=(n, n))

        degrees = np.array(self.graph.degree(), dtype=np.float64)
        old_w = np.array(old_weights, dtype=np.float64)

        # A²[u,v] = |N(u) ∩ N(v)|（公共邻居数，不含 u、v 自身，因为 A 无自环）
        A2 = A @ A
        intersection = np.asarray(A2[us, vs]).ravel().astype(np.float64)

        # 各自排除对方端点后的度数：deg(u)-1, deg(v)-1
        deg_u = degrees[us] - 1.0
        deg_v = degrees[vs] - 1.0
        union = deg_u + deg_v - intersection

        # 叶节点边（原始度数为1）：直接保留原始权重（j_coeff=1）
        leaf_mask = (degrees[us] == 1) | (degrees[vs] == 1)

        safe_union = np.where(union > 0, union, 1.0)
        j_coeff = np.where(union > 0, intersection / safe_union, 0.0)
        j_coeff[leaf_mask] = 1.0

        return (old_w * j_coeff).tolist()

    def _apply_jaccard_weighting_fallback(self, old_weights) -> list:
        """
        纯 Python 回退实现（numpy/scipy 不可用时使用）。
        """
        jaccard_weights = []
        adj_list = [set(self.graph.neighbors(v)) for v in range(self.graph.vcount())]
        degrees = self.graph.degree()

        for edge in self.graph.es:
            u, v = edge.tuple
            if degrees[u] == 1 or degrees[v] == 1:
                jaccard_weights.append(old_weights[edge.index])
                continue
            neighbors_u = adj_list[u] - {u, v}
            neighbors_v = adj_list[v] - {u, v}
            intersection = len(neighbors_u & neighbors_v)
            union = len(neighbors_u | neighbors_v)
            j_coeff = intersection / union if union > 0 else 0.0
            jaccard_weights.append(old_weights[edge.index] * j_coeff)

        return jaccard_weights


class PfamDomainAnalyzer:
    """
    基于 SQLite 数据库查询蛋白质 Pfam 结构域，并计算一组序列的结构域信息熵，
    用于评价 SSN 聚类结果中每个 cluster 的结构域一致程度。
    """

    def __init__(self, db_path: str, evalue_threshold: float = 1e-5):
        """
        :param db_path: hmmscan 结果 SQLite 数据库路径
        :param evalue_threshold: E-value 过滤阈值（默认 1e-5）
        """
        self.db_path = db_path
        self.evalue_threshold = evalue_threshold
        self._conn: Optional[sqlite3.Connection] = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
        return self._conn

    def close(self):
        """关闭数据库连接。"""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def query_domains(self, seq_ids: List[str]) -> Dict[str, List[str]]:
        """
        批量查询一组序列 ID 对应的 Pfam 结构域。

        :param seq_ids: 序列 ID 列表（对应数据库 target_name 字段）
        :return: {seq_id: [domain1, domain2, ...]} 的字典；无命中则值为空列表
        """
        if not seq_ids:
            return {}
        conn = self._get_conn()
        placeholders = ",".join("?" * len(seq_ids))
        sql = (
            f"SELECT target_name, query_name FROM hmmscan_tblout "
            f"WHERE target_name IN ({placeholders}) AND full_evalue <= ? "
            f"ORDER BY target_name, full_evalue"
        )
        cursor = conn.execute(sql, seq_ids + [self.evalue_threshold])
        result: Dict[str, List[str]] = {sid: [] for sid in seq_ids}
        for target_name, query_name in cursor:
            if target_name in result:
                result[target_name].append(query_name)
        return result

    def domain_entropy(self, seq_ids: List[str]) -> Dict[str, Any]:
        """
        计算一组序列所涉及 Pfam 结构域的信息熵，用于评价结构域一致性。
        熵越低表示该 cluster 内序列的结构域组成越一致。

        :param seq_ids: 序列 ID 列表
        :return: 包含熵值及统计信息的字典
        """
        domain_map = self.query_domains(seq_ids)
        total_seqs = len(seq_ids)
        seqs_with_hit = sum(1 for v in domain_map.values() if v)

        # 收集所有结构域（每条序列的每个结构域计一次，允许重复）
        all_domains: List[str] = []
        for domains in domain_map.values():
            all_domains.extend(domains)

        if not all_domains:
            return {
                "total_seqs": total_seqs,
                "seqs_with_hit": 0,
                "hit_ratio": 0.0,
                "unique_domains": 0,
                "domain_entropy": float("nan"),
                "top_domains": []
            }

        counts = Counter(all_domains)
        total = sum(counts.values())
        entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())

        top_domains = counts.most_common(5)

        return {
            "total_seqs": total_seqs,
            "seqs_with_hit": seqs_with_hit,
            "hit_ratio": seqs_with_hit / total_seqs if total_seqs > 0 else 0.0,
            "unique_domains": len(counts),
            "domain_entropy": entropy,
            "top_domains": top_domains
        }
