import igraph as ig
import numpy as np
try:
    import graph_tool.all as gt
    HAS_GRAPH_TOOL = True
except ImportError:
    HAS_GRAPH_TOOL = False
    from sknetwork.clustering import Louvain
from typing import List, Optional, Union


class SBMClustering:
    """
    基于 Stochastic Block Model (SBM) 思想的聚类实现。
    由于环境中缺少 graph-tool 且 igraph 原生不支持 SBM 推理，
    本模块通过使用 scikit-network 的 Louvain 算法并配置 Potts 模型来实现类似效果。
    Potts 模型的模块度定义更接近于标准 SBM 在特定情况下的优化目标。
    """

    def __init__(self, graph: ig.Graph):
        self.graph = graph

    def _to_gt_graph(self, weight_attr: Optional[str] = None):
        """将 igraph 对象转换为 graph-tool 对象"""
        g_gt = gt.Graph(directed=False)
        g_gt.add_vertex(self.graph.vcount())
        
        edges = self.graph.get_edgelist()
        g_gt.add_edge_list(edges)
        
        weights = None
        if weight_attr and weight_attr in self.graph.edge_attributes():
            weights = g_gt.new_edge_property("double")
            weights.a = self.graph.es[weight_attr]
            
        return g_gt, weights

    def cluster(
        self,
        use_gt: bool = True,
        sbm_type: str = 'standard',  # 'standard' 或 'nested'
        degree_corrected: bool = True,
        resolution: float = 1.0,
        modularity: str = 'potts',
        weights: Optional[Union[str, List[float]]] = None,
        random_state: Optional[int] = 42,
        **kwargs
    ) -> ig.VertexClustering:
        """
        执行 SBM 风格的聚类。
        优先使用 graph-tool 进行贝叶斯推断，若缺失则回退至 scikit-network。

        :param use_gt: 是否尝试使用 graph-tool (如果可用)。
        :param sbm_type: SBM 模型类型: 'standard' (标准), 'nested' (层次/嵌套)。
        :param degree_corrected: 是否启用度校正 (仅限 graph-tool)。
        :param resolution: 分辨率参数 (仅限 scikit-network)。
        :param modularity: 模块度类型 (仅限 scikit-network)。
        :param weights: 边权重。
        :param random_state: 随机种子。
        :param kwargs: 其他传递给底层算法的参数。
        :return: igraph.VertexClustering 对象。
        """
        # 场景 A: 使用 graph-tool (推荐)
        if HAS_GRAPH_TOOL and use_gt:
            # 同时支持字符串属性名和列表权重
            weight_attr = weights if isinstance(weights, str) else None
            if isinstance(weights, (list, tuple)):
                temp_attr = "_temp_sbm_weight"
                self.graph.es[temp_attr] = weights
                weight_attr = temp_attr
            g_gt, gt_weights = self._to_gt_graph(weight_attr)
            if isinstance(weights, (list, tuple)):
                del self.graph.es[temp_attr]

            # rec_types 指定边协变量类型，'real-exponential' 适合非负权重（如 fident、bits）
            rec_types = ['real-exponential'] if gt_weights else []

            if sbm_type == 'nested':
                # 嵌套 SBM (自动推断层级和聚类数)
                state = gt.minimize_nested_blockmodel_dl(
                    g_gt,
                    state_args=dict(recs=[gt_weights] if gt_weights else [],
                                    rec_types=rec_types)
                )
                # 获取最底层（最细颗粒度）的划分
                levels = state.get_bs()
                membership = list(levels[0])
            else:
                # 标准 SBM (自动推断聚类数)
                state = gt.minimize_blockmodel_dl(
                    g_gt,
                    deg_corr=degree_corrected,
                    state_args=dict(recs=[gt_weights] if gt_weights else [],
                                    rec_types=rec_types)
                )
                membership = list(state.get_blocks())
                
            return ig.VertexClustering(self.graph, membership=membership)

        # 场景 B: 回退至 scikit-network (Potts 模型模拟)
        else:
            if not HAS_GRAPH_TOOL and use_gt:
                print("警告: 未检测到 graph-tool，将回退至 scikit-network (Potts 模型)。")

            # 1. 获取邻接矩阵 (CSR 格式)
            if isinstance(weights, str):
                if weights in self.graph.edge_attributes():
                    adj = self.graph.get_adjacency_sparse(attribute=weights)
                else:
                    raise ValueError(f"图中不存在边属性: {weights}")
            elif isinstance(weights, (list, tuple)):
                temp_attr = "_temp_sbm_weight"
                self.graph.es[temp_attr] = weights
                adj = self.graph.get_adjacency_sparse(attribute=temp_attr)
                del self.graph.es[temp_attr]
            else:
                adj = self.graph.get_adjacency_sparse()

            # 2. 运行 scikit-network 的 Louvain
            from sknetwork.clustering import Louvain
            louvain = Louvain(
                resolution=resolution,
                modularity=modularity,
                random_state=random_state,
                **kwargs
            )
            
            # sknetwork 的 fit_predict 接收邻接矩阵
            labels = louvain.fit_predict(adj)
            
            # 3. 转换为 igraph 的 membership 格式
            membership = labels.tolist()
            
            return ig.VertexClustering(self.graph, membership=membership)
