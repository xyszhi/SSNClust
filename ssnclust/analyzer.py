import igraph as ig
from typing import Dict, Any, List, Optional

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
        计算基础统计指标。
        """
        if self.graph.vcount() == 0:
            return {
                "nodes": 0, 
                "edges": 0,
                "density": 0.0,
                "is_connected": False,
                "components": 0,
                "avg_clustering": 0.0
            }
            
        return {
            "nodes": self.graph.vcount(),
            "edges": self.graph.ecount(),
            "density": self.graph.density(),
            "is_connected": self.graph.is_connected(),
            "components": len(self.graph.connected_components()),
            "avg_clustering": self.graph.transitivity_avglocal_undirected(mode="zero")
        }

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

    def apply_jaccard_weighting(self, base_weight: str = 'weight', new_attr: str = 'jaccard_weight'):
        """
        对边的权重进行 Jaccard 系数加权。
        新权重 = 原始权重 * (两个节点邻居的交集 / 两个节点邻居的并集)
        
        :param base_weight: 原始权重的属性名。
        :param new_attr: 加权后权重的属性名。
        """
        if base_weight not in self.graph.edge_attributes():
            # 如果没有指定权重，则默认原始权重为 1.0
            old_weights = [1.0] * self.graph.ecount()
        else:
            old_weights = self.graph.es[base_weight]

        jaccard_weights = []
        
        # 获取所有邻居集合以提高效率
        adj_list = [set(self.graph.neighbors(v)) for v in range(self.graph.vcount())]
        
        for edge in self.graph.es:
            u, v = edge.tuple
            neighbors_u = adj_list[u]
            neighbors_v = adj_list[v]
            
            intersection = len(neighbors_u.intersection(neighbors_v))
            union = len(neighbors_u.union(neighbors_v))
            
            j_coeff = intersection / union if union > 0 else 0.0
            jaccard_weights.append(old_weights[edge.index] * j_coeff)
            
        self.graph.es[new_attr] = jaccard_weights
        self.active_weight = new_attr
        print(f"已完成 Jaccard 加权，新权重已保存至属性: {new_attr}，后续分析将默认使用此权重。")
