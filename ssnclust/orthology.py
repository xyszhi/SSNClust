"""
直系同源精修模块 (Orthology Refinement)

用于在已经生成的子网络 (SSN cluster) 内部，解决"序列数/基因组数 > 1"
（即簇内存在同一基因组贡献多条序列，旁系同源/近期重复）的问题，
从中挑选出满足"每个基因组恰好一条序列"原则的直系同源子簇。

核心算法（方案 C）：
1. 构建"基因组级精简图" (Genome-Reduced Graph)：对每一对存在边连接的基因组，
   只保留权重最高的一条边 (Best Reciprocal Edge, BRE)。
2. 在精简图上按加权度 (weighted degree) 从高到低贪心选择每个基因组的代表节点。
3. (可选) 对基因组数较少的簇执行局部交换搜索 (local swap search) 进一步优化。
4. 用选出的代表节点集合在原图上做连通性检验，按连通分量拆分为若干最终子簇。
5. 未被选中的序列作为"额外旁系同源"单独输出，不丢弃。
"""

import igraph as ig
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple


def default_genome_of(name: str, delimiter: str = '|', field_index: int = 0) -> str:
    """
    从序列名中提取基因组 ID。默认约定：序列名格式为 `<genome>|<gene_id>`，
    取分隔符前的第一段作为基因组 ID；若序列名中不含分隔符，则整个序列名视为基因组 ID。
    """
    if delimiter in name:
        return name.split(delimiter)[field_index]
    return name


def build_reduced_graph(graph: ig.Graph, genome_of: Callable[[str], str],
                         weight_attr: str = 'weight') -> ig.Graph:
    """
    构建基因组级精简图：节点与原图一一对应，但每对基因组之间只保留权重最高的一条边 (BRE)。
    """
    has_weight = weight_attr in graph.edge_attributes()
    names = graph.vs['name']
    best_edge: Dict[Tuple[str, str], Tuple[float, int, int]] = {}

    for e in graph.es:
        u, v = e.tuple
        gu = genome_of(names[u])
        gv = genome_of(names[v])
        if gu == gv:
            continue  # 同一基因组内部的边不参与基因组对比较
        w = e[weight_attr] if has_weight else 1.0
        key = (gu, gv) if gu < gv else (gv, gu)
        if key not in best_edge or w > best_edge[key][0]:
            best_edge[key] = (w, u, v)

    reduced = ig.Graph(n=graph.vcount())
    reduced.vs['name'] = names
    edges = [(u, v) for _, u, v in best_edge.values()]
    weights = [w for w, _, _ in best_edge.values()]
    reduced.add_edges(edges)
    reduced.es['weight'] = weights
    return reduced


def greedy_select_representatives(reduced_graph: ig.Graph,
                                   genome_of: Callable[[str], str]) -> Tuple[Dict[str, int], List[float]]:
    """
    按加权度从高到低贪心选择每个基因组的代表节点。
    加权度高的序列通常与更多其它基因组存在较强的直系同源信号，优先保留。
    """
    names = reduced_graph.vs['name']
    weighted_degree = reduced_graph.strength(weights='weight') if reduced_graph.ecount() > 0 \
        else [0.0] * reduced_graph.vcount()
    order = sorted(range(reduced_graph.vcount()), key=lambda i: weighted_degree[i], reverse=True)

    selected: Dict[str, int] = {}
    for idx in order:
        g = genome_of(names[idx])
        if g not in selected:
            selected[g] = idx
    return selected, weighted_degree


def _induced_weight(reduced_graph: ig.Graph, idx_set: set) -> float:
    total = 0.0
    for e in reduced_graph.es:
        u, v = e.tuple
        if u in idx_set and v in idx_set:
            total += e['weight']
    return total


def local_swap_search(reduced_graph: ig.Graph, selected: Dict[str, int],
                       node_by_genome: Dict[str, List[int]],
                       max_iterations: int = 50) -> Tuple[Dict[str, int], float]:
    """
    局部交换搜索：对拥有多个候选序列的基因组，尝试用备选序列替换当前代表，
    若能提升所选集合在精简图上的总边权重则接受替换，迭代至收敛或达到最大迭代次数。
    仅建议在基因组数较少的簇上启用（复杂度与精简图边数、迭代次数相关）。
    """
    selected = dict(selected)
    sel_idx_set = set(selected.values())
    current_weight = _induced_weight(reduced_graph, sel_idx_set)

    improved = True
    iterations = 0
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        for g, candidates in node_by_genome.items():
            if len(candidates) <= 1:
                continue
            current_idx = selected[g]
            for cand_idx in candidates:
                if cand_idx == current_idx:
                    continue
                trial_set = (sel_idx_set - {current_idx}) | {cand_idx}
                trial_weight = _induced_weight(reduced_graph, trial_set)
                if trial_weight > current_weight:
                    selected[g] = cand_idx
                    sel_idx_set = trial_set
                    current_weight = trial_weight
                    improved = True
                    break
            if improved:
                break
    return selected, current_weight


def refine_cluster_to_single_copy(graph: ig.Graph,
                                   genome_of: Optional[Callable[[str], str]] = None,
                                   delimiter: str = '|',
                                   field_index: int = 0,
                                   weight_attr: str = 'weight',
                                   enable_local_search: bool = True,
                                   max_genomes_for_search: int = 200,
                                   max_iterations: int = 50) -> Tuple[List[ig.Graph], List[dict], dict]:
    """
    对输入子网络（cluster）进行直系同源精修，使输出的每个子簇满足
    "每个基因组恰好一条序列" 的原则。

    :param graph: 输入的子网络（igraph.Graph，节点须有 'name' 属性）
    :param genome_of: 从序列名提取基因组 ID 的函数；默认使用 default_genome_of
    :param delimiter: 当 genome_of 为 None 时，用于拆分序列名的分隔符
    :param field_index: 当 genome_of 为 None 时，取分隔符拆分后的第几段作为基因组 ID
    :param weight_attr: 边权重属性名
    :param enable_local_search: 是否对基因组数较少的簇启用局部交换搜索
    :param max_genomes_for_search: 启用局部交换搜索的基因组数上限
    :param max_iterations: 局部交换搜索的最大迭代次数
    :return: (refined_subclusters, extra_paralogs, stats)
        refined_subclusters: 精修后满足单拷贝原则且连通的子图列表
        extra_paralogs: 落选序列列表，每项为 {'name', 'genome', 'reason'}
        stats: 精修过程统计信息
    """
    if genome_of is None:
        genome_of = lambda name: default_genome_of(name, delimiter, field_index)

    names = graph.vs['name']
    node_by_genome: Dict[str, List[int]] = defaultdict(list)
    for idx, name in enumerate(names):
        node_by_genome[genome_of(name)].append(idx)

    n_genomes = len(node_by_genome)
    n_multi_copy_genomes = sum(1 for v in node_by_genome.values() if len(v) > 1)

    if n_multi_copy_genomes == 0:
        # 已经满足单拷贝原则，仅需按连通分量拆分（通常只有一个分量）
        components = graph.connected_components()
        refined_subclusters = [graph.induced_subgraph(comp) for comp in components]
        stats = {
            'n_genomes': n_genomes,
            'n_multi_copy_genomes': 0,
            'n_representatives': graph.vcount(),
            'n_extra_paralogs': 0,
            'n_low_confidence_genomes': 0,
            'n_refined_subclusters': len(refined_subclusters),
        }
        return refined_subclusters, [], stats

    reduced = build_reduced_graph(graph, genome_of, weight_attr)
    selected, _ = greedy_select_representatives(reduced, genome_of)

    if enable_local_search and n_genomes <= max_genomes_for_search:
        selected, _ = local_swap_search(reduced, selected, node_by_genome, max_iterations)

    # 兜底：若某基因组的代表节点在精简图中孤立（度数为 0），说明该基因组缺乏明确的
    # 跨基因组直系同源信号，改用原图中度数最高的候选节点作代表，并标记为低置信度
    reduced_degree = reduced.degree()
    low_confidence_genomes = set()
    for g, candidates in node_by_genome.items():
        idx = selected[g]
        if reduced_degree[idx] == 0:
            best_idx = max(candidates, key=lambda i: graph.degree(i))
            selected[g] = best_idx
            low_confidence_genomes.add(g)

    selected_idx_set = set(selected.values())

    # 用代表节点集合构造诱导子图，并按连通分量拆分为最终子簇
    induced = graph.induced_subgraph(sorted(selected_idx_set))
    components = induced.connected_components()
    refined_subclusters = [induced.induced_subgraph(comp) for comp in components]
    for sub in refined_subclusters:
        sub['low_confidence_genomes'] = sorted(
            g for g in low_confidence_genomes if g in {genome_of(n) for n in sub.vs['name']}
        )

    # 收集落选序列（额外旁系同源）
    extra_paralogs = []
    for g, candidates in node_by_genome.items():
        rep_idx = selected[g]
        for idx in candidates:
            if idx != rep_idx:
                extra_paralogs.append({
                    'name': names[idx],
                    'genome': g,
                    'reason': 'lower_weighted_degree_than_representative',
                })

    stats = {
        'n_genomes': n_genomes,
        'n_multi_copy_genomes': n_multi_copy_genomes,
        'n_representatives': len(selected_idx_set),
        'n_extra_paralogs': len(extra_paralogs),
        'n_low_confidence_genomes': len(low_confidence_genomes),
        'n_refined_subclusters': len(refined_subclusters),
    }
    return refined_subclusters, extra_paralogs, stats
