import os
import sys
import warnings
import igraph as ig
from scipy.sparse import SparseEfficiencyWarning

# markov_clustering 在导入时若检测到 networkx 未安装会向 stderr 打印警告，
# 此处临时抑制这些无关输出
with open(os.devnull, 'w') as _devnull:
    _old_stderr, sys.stderr = sys.stderr, _devnull
    import markov_clustering as mcl
    sys.stderr = _old_stderr
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

        # 2. 运行 MCL（屏蔽第三方库内部的稀疏矩阵结构变更警告）
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
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
        # 用 -1 标记未分配节点，避免孤立节点被错误归入 cluster 0
        membership = [-1] * self.graph.vcount()
        for cluster_id, nodes in enumerate(clusters):
            for node in nodes:
                membership[node] = cluster_id

        # 将未被 MCL 覆盖的孤立节点各自单独成簇
        next_id = len(clusters)
        for i, m in enumerate(membership):
            if m == -1:
                membership[i] = next_id
                next_id += 1

        return ig.VertexClustering(self.graph, membership=membership)
