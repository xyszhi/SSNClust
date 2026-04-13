import igraph as ig
import leidenalg as la
from typing import List, Optional, Union, Type


class LeidenClustering:
    """
    基于 leidenalg 包的针对 SSN 的 Leiden 聚类方法实现。
    支持多种划分模型和加权。
    """

    def __init__(self, graph: ig.Graph):
        self.graph = graph

    def cluster(
        self,
        partition_type: Union[str, Type[la.VertexPartition]] = "Modularity",
        initial_membership: Optional[List[int]] = None,
        weights: Optional[Union[str, List[float]]] = None,
        n_iterations: int = 2,
        max_comm_size: int = 0,
        seed: Optional[int] = None,
        **kwargs
    ) -> ig.VertexClustering:
        """
        执行 Leiden 聚类。

        :param partition_type: 划分模型类型。可以是字符串或 leidenalg.VertexPartition 的子类。
            常见字符串选项: 'Modularity', 'RBConfiguration', 'RBER', 'CPM', 'Significance', 'Surprise'。
        :param initial_membership: 初始成员列表。
        :param weights: 边权重。可以是边属性名称或权重列表。
        :param n_iterations: 迭代次数。默认 2，-1 表示运行到收敛。
        :param max_comm_size: 社区最大规模限制（默认 0 表示无限制）。
        :param seed: 随机种子。
        :param kwargs: 传递给特定划分模型的参数（如 resolution_parameter）。
        :return: igraph.VertexClustering 对象。
        """
        # 处理划分类型
        pt_map = {
            "Modularity": la.ModularityVertexPartition,
            "RBConfiguration": la.RBConfigurationVertexPartition,
            "RBER": la.RBERVertexPartition,
            "CPM": la.CPMVertexPartition,
            "Significance": la.SignificanceVertexPartition,
            "Surprise": la.SurpriseVertexPartition,
        }

        if isinstance(partition_type, str):
            if partition_type not in pt_map:
                raise ValueError(f"不支持的划分类型: {partition_type}。可选: {list(pt_map.keys())}")
            actual_partition_class = pt_map[partition_type]
        else:
            actual_partition_class = partition_type

        # 处理权重
        edge_weights = None
        if isinstance(weights, str):
            if weights in self.graph.edge_attributes():
                edge_weights = self.graph.es[weights]
            else:
                raise ValueError(f"图中不存在边属性: {weights}")
        elif isinstance(weights, (list, tuple)):
            edge_weights = weights

        # 执行聚类
        partition = la.find_partition(
            self.graph,
            actual_partition_class,
            initial_membership=initial_membership,
            weights=edge_weights,
            n_iterations=n_iterations,
            max_comm_size=max_comm_size,
            seed=seed,
            **kwargs
        )

        return ig.VertexClustering(self.graph, membership=partition.membership)

    def cluster_modularity(self, weights: Optional[str] = None, **kwargs) -> ig.VertexClustering:
        """使用模块度优化的快捷方法"""
        return self.cluster(partition_type="Modularity", weights=weights, **kwargs)

    def cluster_cpm(self, resolution: float = 0.01, weights: Optional[str] = None, **kwargs) -> ig.VertexClustering:
        """使用 CPM (Constant Potts Model) 优化的快捷方法，适用于 SSN"""
        return self.cluster(partition_type="CPM", resolution_parameter=resolution, weights=weights, **kwargs)

    def cluster_rb(self, resolution: float = 1.0, weights: Optional[str] = None, **kwargs) -> ig.VertexClustering:
        """使用 RB (Reichardt & Bornholdt) 优化的快捷方法"""
        return self.cluster(partition_type="RBConfiguration", resolution_parameter=resolution, weights=weights, **kwargs)
