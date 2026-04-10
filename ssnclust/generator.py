import igraph as ig
import math
from .utils import parse_m8_tsv
from typing import Optional, Dict, Any, List

class SSNGenerator:
    """
    SSN 生成器，将比对结果文件转换为序列相似性网络 (SSN)。
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.graph = ig.Graph(directed=False)

    def generate(self, 
                 evalue_threshold: float = 1e-5, 
                 identity_threshold: float = 0.0,
                 alnlen_threshold: int = 0,
                 weight_by: str = 'evalue') -> ig.Graph:
        """
        根据过滤条件生成 SSN。
        
        :param evalue_threshold: E-value 阈值，E-value <= 该值则保留。
        :param identity_threshold: Identity 阈值 (0-1)，fident >= 该值则保留。
        :param alnlen_threshold: 比对长度阈值，alnlen >= 该值则保留。
        :param weight_by: 权重基于哪个指标。'evalue' (计算 -log10), 'fident', 'bits' 或 None。
        """
        nodes = set()
        edges = []
        weights = []

        for row in parse_m8_tsv(self.file_path):
            query = row['query']
            target = row['target']
            evalue = row['evalue']
            identity = row['fident']
            alnlen = row.get('alnlen', 0)
            bits = row.get('bits', 0.0)

            # 自环通常在 SSN 中意义不大，或者根据需求保留
            if query == target:
                nodes.add(query)
                continue

            # 过滤
            if evalue > evalue_threshold:
                continue
            if identity < identity_threshold:
                continue
            if alnlen < alnlen_threshold:
                continue

            nodes.add(query)
            nodes.add(target)

            # 计算权重
            w = 0.0
            if weight_by == 'evalue':
                # 防止 log(0)
                safe_evalue = max(evalue, 1e-200)
                w = -math.log10(safe_evalue)
            elif weight_by == 'fident':
                w = identity
            elif weight_by == 'bits':
                w = bits

            edges.append((query, target))
            weights.append(w)

        # 构建图
        # 首先添加所有节点
        node_list = sorted(list(nodes))
        self.graph.add_vertices(node_list)
        self.graph.vs['name'] = node_list
        
        # 将节点名称映射到索引
        node_to_idx = {name: i for i, name in enumerate(node_list)}
        
        # 转换为索引列表
        edge_indices = [(node_to_idx[q], node_to_idx[t]) for q, t in edges]
        
        self.graph.add_edges(edge_indices)
        if weight_by:
            self.graph.es['weight'] = weights

        return self.graph
