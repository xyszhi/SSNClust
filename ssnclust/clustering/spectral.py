import igraph as ig
import numpy as np
from sklearn.cluster import SpectralClustering as SKLSpectralClustering
from typing import Optional, Union, List


class SSNSpectralClustering:
    """
    基于 scikit-learn 的谱聚类 (Spectral Clustering) 实现。
    适用于 SSN 网络的序列聚类。
    """

    def __init__(self, graph: ig.Graph):
        self.graph = graph

    def cluster(
        self,
        n_clusters: int = 8,
        weights: Optional[Union[str, List[float]]] = None,
        assign_labels: str = "kmeans",
        random_state: Optional[int] = None,
        **kwargs
    ) -> ig.VertexClustering:
        """
        执行谱聚类。

        :param n_clusters: 聚类数量。
        :param weights: 边权重属性名称或权重列表。
        :param assign_labels: 分配标签的策略 ('kmeans' 或 'discretize')。
        :param random_state: 随机种子。
        :param kwargs: 传递给 sklearn.cluster.SpectralClustering 的其他参数。
        :return: igraph.VertexClustering 对象。
        """
        # 获取邻接矩阵
        if isinstance(weights, str):
            if weights in self.graph.edge_attributes():
                adj = self.graph.get_adjacency(attribute=weights).data
            else:
                raise ValueError(f"图中不存在边属性: {weights}")
        elif isinstance(weights, (list, tuple)):
            # 如果是列表，需要构建带权重的邻接矩阵，igraph 的 get_adjacency 不直接支持列表权重
            # 手动构建
            adj = np.zeros((self.graph.vcount(), self.graph.vcount()))
            for i, edge in enumerate(self.graph.es):
                u, v = edge.tuple
                adj[u, v] = adj[v, u] = weights[i]
        else:
            # 无权重
            adj = self.graph.get_adjacency().data

        # 转换为 numpy 数组
        affinity_matrix = np.array(adj)

        # 执行 scikit-learn 的谱聚类
        # 注意：这里我们直接传入预计算的亲和矩阵 (affinity='precomputed')
        sc = SKLSpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            assign_labels=assign_labels,
            random_state=random_state,
            **kwargs
        )
        
        labels = sc.fit_predict(affinity_matrix)

        return ig.VertexClustering(self.graph, membership=labels.tolist())
