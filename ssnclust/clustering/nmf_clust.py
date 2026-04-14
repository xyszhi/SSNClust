import igraph as ig
import numpy as np
from sklearn.decomposition import NMF
from typing import List, Optional, Union


class NMFClustering:
    """
    基于非负矩阵分解 (NMF) 的针对 SSN 的聚类实现。
    NMF 将邻接矩阵 A 分解为 W * H，其中 H 矩阵通常表示节点与社区的关系强度。
    """

    def __init__(self, graph: ig.Graph):
        self.graph = graph

    def cluster(
        self,
        n_components: int = 8,
        init: str = 'nndsvd',
        random_state: Optional[int] = 42,
        weights: Optional[Union[str, List[float]]] = None,
        max_iter: int = 1000,
        **kwargs
    ) -> ig.VertexClustering:
        """
        执行 NMF 聚类。

        :param n_components: 分解的组件数（即预期的聚类数）。
        :param init: 初始化方法 ('random', 'nndsvd', 'nndsvdar', 'nndsvdca', 'custom')。
        :param random_state: 随机种子。
        :param weights: 边权重。
        :param max_iter: 最大迭代次数。
        :param kwargs: 传递给 sklearn.decomposition.NMF 的其他参数。
        :return: igraph.VertexClustering 对象。
        """
        # 1. 获取邻接矩阵 (密集矩阵，NMF 通常在非稀疏或适度稀疏的表示上运行)
        # 获取稀疏矩阵然后转换为密集矩阵
        if isinstance(weights, str):
            if weights in self.graph.edge_attributes():
                adj = self.graph.get_adjacency_sparse(attribute=weights)
            else:
                raise ValueError(f"图中不存在边属性: {weights}")
        elif isinstance(weights, (list, tuple)):
            temp_attr = "_temp_nmf_weight"
            self.graph.es[temp_attr] = weights
            adj = self.graph.get_adjacency_sparse(attribute=temp_attr)
            del self.graph.es[temp_attr]
        else:
            adj = self.graph.get_adjacency_sparse()

        # NMF 需要非负矩阵，邻接矩阵通常是非负的。
        # A 是 n x n 矩阵。我们要找到 W (n x k) 和 H (k x n)，使得 A ≈ W * H
        # 在社区检测中，通常 A ≈ W * W.T (对称分解)，或者直接对 A 进行标准分解。
        # 这里使用标准分解 A ≈ W * H，H 的列向量表示节点在各个组件（社区）上的权重。
        
        # n_components 不能超过节点数，否则 NMF 会报错
        n_components = min(n_components, self.graph.vcount())

        nmf_model = NMF(
            n_components=n_components,
            init=init,
            random_state=random_state,
            max_iter=max_iter,
            **kwargs
        )
        
        # W 是节点对主题/社区的贡献，H 是主题对节点的贡献。
        # 在对称邻接矩阵中，W 和 H.T 应该是类似的。
        # nndsvd 初始化不支持稀疏矩阵，统一转为密集矩阵
        W = nmf_model.fit_transform(adj.toarray())
        
        # 3. 提取聚类结果：每个节点所属的社区为其在 W 中权重最大的索引
        membership = np.argmax(W, axis=1).tolist()
        
        return ig.VertexClustering(self.graph, membership=membership)
