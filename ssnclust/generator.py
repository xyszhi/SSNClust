import igraph as ig
from .utils import parse_m8_tsv
from typing import Optional, Dict, Any, List, Tuple
from concurrent.futures import ProcessPoolExecutor


def _filter_chunk(args: Tuple) -> Tuple[set, dict]:
    """
    子进程工作函数：对一批原始行（字符串列表）执行过滤，返回局部 directed_pairs 和 pair_attrs。

    参数元组字段：
        lines            : 原始 TSV 行列表（不含表头）
        headers          : 列名列表
        evalue_threshold : E-value 阈值
        identity_threshold: Identity 阈值
        alnlen_threshold : 比对长度阈值
        coverage_threshold: Coverage 阈值
        coverage_mode    : Coverage 过滤模式
        keep_cols        : 需要保留的列名集合（frozenset）
        extra_filters    : 额外过滤条件字典 {col: threshold}
    返回：
        (directed_pairs, pair_attrs)
        directed_pairs : set of (query, target)
        pair_attrs     : dict {(query, target): {attr: value}}
    """
    (lines, headers, evalue_threshold, identity_threshold, alnlen_threshold,
     coverage_threshold, coverage_mode, keep_cols, extra_filters) = args

    directed_pairs: set = set()
    pair_attrs: dict = {}

    for line in lines:
        line = line.rstrip('\n')
        if not line:
            continue
        values = line.split('\t')
        if len(values) != len(headers):
            continue
        row: dict = {}
        for h, v in zip(headers, values):
            if h in ('query', 'target'):
                row[h] = v
            else:
                try:
                    fv = float(v)
                    row[h] = int(fv) if fv.is_integer() else fv
                except (ValueError, TypeError):
                    row[h] = v

        query = row.get('query', '')
        target = row.get('target', '')

        if query == target:
            continue
        if 'evalue' in row and row['evalue'] > evalue_threshold:
            continue
        if 'fident' in row and row['fident'] < identity_threshold:
            continue
        if 'alnlen' in row and row['alnlen'] < alnlen_threshold:
            continue

        qcov = row.get('qcov', 0.0)
        tcov = row.get('tcov', 0.0)
        if coverage_mode == 'min':
            if qcov < coverage_threshold or tcov < coverage_threshold:
                continue
        elif coverage_mode in ('max', 'any'):
            if qcov < coverage_threshold and tcov < coverage_threshold:
                continue

        skip = False
        for col, threshold in extra_filters.items():
            if col in row:
                if isinstance(threshold, (int, float)) and isinstance(row[col], (int, float)):
                    if row[col] < threshold:
                        skip = True
                        break
                elif row[col] != threshold:
                    skip = True
                    break
        if skip:
            continue

        directed_pairs.add((query, target))
        pair_attrs[(query, target)] = {k: v for k, v in row.items() if k in keep_cols}

    return directed_pairs, pair_attrs


class SSNGenerator:
    """
    SSN 生成器，将比对结果文件转换为序列相似性网络 (SSN)。
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.graph = ig.Graph(directed=False)

    def save(self, output_path: str):
        """
        将生成的图保存到文件。支持格式：.graphml, .gexf, .gt, .pajek 等。
        建议使用 .graphml 以获得最佳的 Gephi 兼容性。
        """
        if self.graph.vcount() == 0:
            print("警告: 图为空，未保存。")
            return
            
        # 根据后缀自动识别格式
        self.graph.write(output_path)
        print(f"SSN 已保存至: {output_path}")

    def generate(self, 
                 evalue_threshold: float = 1e-5, 
                 identity_threshold: float = 0.0,
                 alnlen_threshold: int = 0,
                 coverage_threshold: float = 0.0,
                 coverage_mode: str = 'min',
                 weight_by: Optional[str] = 'fident',
                 bidirectional_only: bool = False,
                 retained_fields: Optional[List[str]] = None,
                 n_workers: Optional[int] = None,
                 chunk_size: int = 200_000,
                 **extra_filters: Any) -> ig.Graph:
        """
        根据过滤条件生成 SSN。
        
        :param evalue_threshold: E-value 阈值，E-value <= 该值则保留。
        :param identity_threshold: Identity 阈值 (0-1)，fident >= 该值则保留。
        :param alnlen_threshold: 比对长度阈值，alnlen >= 该值则保留。
        :param coverage_threshold: Coverage 阈值 (0-1)。
        :param coverage_mode: Coverage 过滤模式: 
                              'min': qcov 和 tcov 都需 >= 阈值 (等同于 both)
                              'max': qcov 或 tcov 有一者 >= 阈值 (等同于 any)
                              'any': 同 max
        :param weight_by: 权重基于哪个指标。'fident', 'bits', 'fident_cov'（fident 与 coverage 的乘积）、'fident_cov_harmonic'（fident 与 coverage 的调和平均数）或任何数值列名。
        :param bidirectional_only: 是否只保留双向比对的边。False（默认）表示保留所有比对边，包括单向比对；True 表示只保留 A->B 和 B->A 都存在的比对。
        :param retained_fields: 需要额外保留为边属性的字段列表（默认 None，仅保留过滤/权重所需列）。
        :param n_workers: 并行过滤使用的进程数（默认 None，表示单进程串行；传入正整数则启用多进程）。
        :param chunk_size: 多进程模式下每个分块的行数（默认 200000）。
        :param extra_filters: 额外的过滤条件 (列名=阈值)，默认执行 '列值 >= 阈值'。
        """
        # 每次调用 generate() 时重置图，防止重复调用时数据叠加
        self.graph = ig.Graph(directed=False)

        nodes = set()
        edges = []
        edge_attrs = {} # 用于存储所有提取的列作为边属性

        # 确定需要保留的边属性列（过滤列 + 权重列 + 用户自定义过滤列）
        # 丢弃对网络构建完全无用的列以节省内存
        _FILTER_COLS = {'evalue', 'fident', 'alnlen', 'qcov', 'tcov'}
        _WEIGHT_COLS = {'fident', 'bits', 'qcov', 'tcov'}  # fident_cov/harmonic 也依赖这些
        _keep_cols = _FILTER_COLS | _WEIGHT_COLS
        if weight_by and weight_by not in ('fident_cov', 'fident_cov_harmonic', 'none'):
            _keep_cols.add(weight_by)
        _keep_cols.update(extra_filters.keys())  # 用户自定义过滤列也需保留
        if retained_fields:
            _keep_cols.update(retained_fields)  # 用户指定的额外保留字段

        if n_workers and n_workers > 1:
            # 并行模式：将文件分块，多进程并行过滤，主进程合并结果
            directed_pairs, pair_attrs = self._generate_parallel(
                evalue_threshold, identity_threshold, alnlen_threshold,
                coverage_threshold, coverage_mode, _keep_cols, extra_filters,
                n_workers, chunk_size
            )
        else:
            # 串行模式（默认）：逐行扫描过滤
            directed_pairs: set = set()
            pair_attrs: dict = {}

            for row in parse_m8_tsv(self.file_path):
                query = row['query']
                target = row['target']
                
                # 自环处理
                if query == target:
                    continue

                # 默认过滤
                if 'evalue' in row and row['evalue'] > evalue_threshold:
                    continue
                if 'fident' in row and row['fident'] < identity_threshold:
                    continue
                if 'alnlen' in row and row['alnlen'] < alnlen_threshold:
                    continue
                
                # Coverage 过滤
                qcov = row.get('qcov', 0.0)
                tcov = row.get('tcov', 0.0)
                if coverage_mode == 'min':
                    if qcov < coverage_threshold or tcov < coverage_threshold:
                        continue
                elif coverage_mode in ('max', 'any'):
                    if qcov < coverage_threshold and tcov < coverage_threshold:
                        continue
                
                # 额外的自定义过滤（在节点加入图之前执行，避免产生游离孤立节点）
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

                # 记录通过过滤的有向比对对及其属性
                # 只保留参与过滤或权重计算的列，丢弃无用列以节省内存
                directed_pairs.add((query, target))
                attrs = {k: v for k, v in row.items() if k in _keep_cols}
                pair_attrs[(query, target)] = attrs

        # 双向匹配过滤：根据 bidirectional_only 决定是否保留单向比对
        for (query, target), attrs in pair_attrs.items():
            if bidirectional_only and (target, query) not in directed_pairs:
                continue  # 单向比对，跳过
            nodes.add(query)
            nodes.add(target)
            edges.append((query, target))
            for k, v in attrs.items():
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

        # 去除重复边（双向比对产生的 A->B 和 B->A）
        # 对数值属性取均值，对非数值属性取第一个值（'first'）
        numeric_attrs = [
            k for k in edge_attrs
            if edge_attrs[k] and isinstance(edge_attrs[k][0], (int, float))
        ]
        non_numeric_attrs = [k for k in edge_attrs if k not in numeric_attrs]
        combine = {k: 'mean' for k in numeric_attrs}
        combine.update({k: 'first' for k in non_numeric_attrs})
        self.graph.simplify(multiple=True, loops=True, combine_edges=combine if combine else 'mean')

        # 计算权重 (如果指定了且在 edge_attrs 中)
        if weight_by:
            if weight_by in ('fident_cov', 'fident_cov_harmonic'):
                # coverage 根据 coverage_mode 决定
                es_attrs = self.graph.es.attributes()
                if 'fident' in es_attrs and 'qcov' in es_attrs and 'tcov' in es_attrs:
                    weights = []
                    for e in self.graph.es:
                        fident = e['fident']
                        qcov = e['qcov']
                        tcov = e['tcov']
                        if coverage_mode == 'min':
                            cov = min(qcov, tcov)
                        elif coverage_mode == 'max':
                            cov = max(qcov, tcov)
                        else:  # 'any'
                            cov = (qcov + tcov) / 2.0
                        if weight_by == 'fident_cov_harmonic':
                            # fident 与 cov 的调和平均数: 2*a*b/(a+b)
                            weights.append(2 * fident * cov / (fident + cov) if (fident + cov) > 0 else 0.0)
                        else:
                            weights.append(fident * cov)
                    self.graph.es['weight'] = weights
            elif weight_by in self.graph.es.attributes():
                # 直接使用该列作为权重
                self.graph.es['weight'] = self.graph.es[weight_by]

        return self.graph

    def _generate_parallel(
        self,
        evalue_threshold: float,
        identity_threshold: float,
        alnlen_threshold: int,
        coverage_threshold: float,
        coverage_mode: str,
        keep_cols: set,
        extra_filters: dict,
        n_workers: int,
        chunk_size: int,
    ):
        """
        并行读取并过滤 TSV 文件，返回合并后的 (directed_pairs, pair_attrs)。
        将文件按 chunk_size 行分块，各块由子进程独立过滤，主进程合并结果。
        """
        from ssnclust.utils import REQUIRED_COLUMNS

        # 读取表头并验证
        with open(self.file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            if not first_line:
                raise ValueError(f"文件为空，缺少标题行：{self.file_path}")
            headers = [h.strip() for h in first_line.rstrip('\n').split('\t')]
            missing = REQUIRED_COLUMNS - set(headers)
            if missing:
                raise ValueError(
                    f"TSV 文件缺少必要的列：{sorted(missing)}。"
                    f"文件路径：{self.file_path}\n"
                    f"当前标题行包含：{headers}"
                )

            # 将文件内容分块（不含表头）
            chunks = []
            buf = []
            for line in f:
                buf.append(line)
                if len(buf) >= chunk_size:
                    chunks.append(buf)
                    buf = []
            if buf:
                chunks.append(buf)

        if not chunks:
            return set(), {}

        keep_cols_frozen = frozenset(keep_cols)
        chunk_args = [
            (chunk, headers, evalue_threshold, identity_threshold, alnlen_threshold,
             coverage_threshold, coverage_mode, keep_cols_frozen, extra_filters)
            for chunk in chunks
        ]

        actual_workers = min(n_workers, len(chunks))
        with ProcessPoolExecutor(max_workers=actual_workers) as executor:
            results = list(executor.map(_filter_chunk, chunk_args))

        # 合并各进程结果
        merged_pairs: set = set()
        merged_attrs: dict = {}
        for dp, pa in results:
            merged_pairs.update(dp)
            merged_attrs.update(pa)

        return merged_pairs, merged_attrs
