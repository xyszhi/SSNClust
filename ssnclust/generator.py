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
                 weight_by: Optional[str] = 'evalue',
                 **extra_filters: Any) -> ig.Graph:
        """
        根据过滤条件生成 SSN。
        
        :param evalue_threshold: E-value 阈值，E-value <= 该值则保留。
        :param identity_threshold: Identity 阈值 (0-1)，fident >= 该值则保留。
        :param alnlen_threshold: 比对长度阈值，alnlen >= 该值则保留。
        :param weight_by: 权重基于哪个指标。'evalue' (计算 -log10), 'fident', 'bits' 或任何数值列名。
        :param extra_filters: 额外的过滤条件 (列名=阈值)，默认执行 '列值 >= 阈值'。
        """
        nodes = set()
        edges = []
        edge_attrs = {} # 用于存储所有提取的列作为边属性

        for row in parse_m8_tsv(self.file_path):
            query = row['query']
            target = row['target']
            
            # 自环处理
            if query == target:
                nodes.add(query)
                continue

            # 默认过滤
            if 'evalue' in row and row['evalue'] > evalue_threshold:
                continue
            if 'fident' in row and row['fident'] < identity_threshold:
                continue
            if 'alnlen' in row and row['alnlen'] < alnlen_threshold:
                continue
            
            # 额外的自定义过滤
            skip = False
            for col, threshold in extra_filters.items():
                if col in row:
                    if isinstance(threshold, (int, float)) and isinstance(row[col], (int, float)):
                        if row[col] < threshold:
                            skip = True
                            break
                    elif row[col] != threshold: # 如果不是数值，则进行相等判断
                        skip = True
                        break
            if skip:
                continue

            nodes.add(query)
            nodes.add(target)

            edges.append((query, target))
            
            # 收集该边的所有属性（除了 query 和 target）
            for k, v in row.items():
                if k in ('query', 'target'):
                    continue
                if k not in edge_attrs:
                    edge_attrs[k] = []
                edge_attrs[k].append(v)

        # 构建图
        node_list = sorted(list(nodes))
        self.graph.add_vertices(node_list)
        self.graph.vs['name'] = node_list
        
        node_to_idx = {name: i for i, name in enumerate(node_list)}
        edge_indices = [(node_to_idx[q], node_to_idx[t]) for q, t in edges]
        self.graph.add_edges(edge_indices)
        
        # 添加边属性
        for attr_name, values in edge_attrs.items():
            self.graph.es[attr_name] = values

        # 计算权重 (如果指定了且在 edge_attrs 中)
        if weight_by:
            if weight_by == 'evalue' and 'evalue' in self.graph.es.attributes():
                # 特殊处理 evalue: -log10
                weights = []
                for ev in self.graph.es['evalue']:
                    safe_evalue = max(ev, 1e-200)
                    weights.append(-math.log10(safe_evalue))
                self.graph.es['weight'] = weights
            elif weight_by in self.graph.es.attributes():
                # 直接使用该列作为权重
                self.graph.es['weight'] = self.graph.es[weight_by]

        return self.graph
