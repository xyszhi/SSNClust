import igraph as ig
import markov_clustering as mcl
from typing import List, Optional, Union


class MCLClustering:
    """
    基于 markov_clustering 包的针对 SSN 的 Markov Clustering (MCL) 实现。
    """

    def __init__(self, graph: ig.Graph):
        self.graph = graph

    def cluster(
        self,
        inflation: float = 2.0,
        expansion: int = 2,
        iterations: int = 100,
        weights: Optional[Union[str, List[float]]] = None,
        **kwargs
    ) -> ig.VertexClustering:
        """
        执行 MCL 聚类。

        :param inflation: 膨胀系数 (Inflation parameter)，决定聚类的紧密度。
        :param expansion: 扩张系数 (Expansion parameter)。
        :param iterations: 最大迭代次数。
        :param weights: 边权重。
        :param kwargs: 传递给 markov_clustering.run_mcl 的其他参数。
        :return: igraph.VertexClustering 对象。
        """
        # 1. 获取邻接矩阵 (稀疏矩阵)
        if isinstance(weights, str):
            if weights in self.graph.edge_attributes():
                adj = self.graph.get_adjacency_sparse(attribute=weights)
            else:
                raise ValueError(f"图中不存在边属性: {weights}")
        elif isinstance(weights, (list, tuple)):
            temp_attr = "_temp_mcl_weight"
            self.graph.es[temp_attr] = weights
            adj = self.graph.get_adjacency_sparse(attribute=temp_attr)
            del self.graph.es[temp_attr]
        else:
            adj = self.graph.get_adjacency_sparse()

        # 2. 运行 MCL
        result_matrix = mcl.run_mcl(
            adj,
            inflation=inflation,
            expansion=expansion,
            iterations=iterations,
            **kwargs
        )
        
        # 3. 提取聚类结果
        clusters = mcl.get_clusters(result_matrix)
        
        # 4. 转换为 igraph 的 membership 格式
        # mcl.get_clusters 返回的是 (node_idx, ...) 的元组列表
        membership = [0] * self.graph.vcount()
        for cluster_id, nodes in enumerate(clusters):
            for node in nodes:
                membership[node] = cluster_id
                
        return ig.VertexClustering(self.graph, membership=membership)
