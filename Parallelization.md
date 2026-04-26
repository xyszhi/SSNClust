# SSNClust 并行化改造方案

## 一、背景与目标

SSNClust 当前各核心模块均为单线程串行实现。在处理大规模比对结果（数百万行 TSV、数百个社区）时，存在明显的计算瓶颈。本文档梳理各阶段的并行化潜力，并给出具体的改造方案与示例代码。

---

## 二、各阶段瓶颈分析与并行化方案

### 2.1 TSV 文件解析（`ssnclust/utils.py` → `parse_m8_tsv`）

**现状**：单线程逐行读取，对大型比对结果文件（数百万行）是明显瓶颈。

```python
# 当前实现（串行）
for row in reader:
    for key, value in row.items():
        ...  # 逐行类型转换
    yield row
```

**并行化方案**：

- **方案 A（推荐）**：用 `pandas.read_csv` 替代，底层有 C 加速，无需改动调用方：
  ```python
  import pandas as pd

  def parse_m8_tsv_fast(file_path: str):
      df = pd.read_csv(file_path, sep='\t', dtype={'query': str, 'target': str})
      for row in df.itertuples(index=False):
          yield row._asdict()
  ```

- **方案 B**：将文件按行分块，使用 `ProcessPoolExecutor` 并行解析各块，最后合并：
  ```python
  from concurrent.futures import ProcessPoolExecutor

  def parse_chunk(lines, headers):
      # 解析一批行，返回 list[dict]
      ...

  def parse_m8_tsv_parallel(file_path, n_workers=4, chunk_size=100_000):
      with open(file_path) as f:
          headers = f.readline().strip().split('\t')
          chunks = []
          buf = []
          for line in f:
              buf.append(line)
              if len(buf) >= chunk_size:
                  chunks.append(buf)
                  buf = []
          if buf:
              chunks.append(buf)
      with ProcessPoolExecutor(max_workers=n_workers) as pool:
          results = pool.map(parse_chunk, chunks, [headers]*len(chunks))
      return [row for chunk in results for row in chunk]
  ```

---

### 2.2 图构建中的过滤与属性收集（`ssnclust/generator.py` → `generate`）

**现状**：单线程串行扫描所有比对行，逐行过滤并收集 `directed_pairs` 和 `pair_attrs`。

**并行化方案**：

将文件分块后并行过滤，各进程返回局部 `directed_pairs` 和 `pair_attrs`，主进程合并。注意 `bidirectional_only` 的双向匹配需在合并后统一处理。

```python
from concurrent.futures import ProcessPoolExecutor

def filter_chunk(lines, headers, evalue_threshold, identity_threshold, ...):
    """处理一批行，返回 (directed_pairs, pair_attrs)"""
    directed_pairs = set()
    pair_attrs = {}
    # ... 过滤逻辑 ...
    return directed_pairs, pair_attrs

def generate_parallel(file_path, n_workers=4, **filter_kwargs):
    # 分块读取
    chunks = split_file_into_chunks(file_path, n_workers)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results = pool.map(filter_chunk, chunks, ...)
    # 合并各进程结果
    all_pairs = set()
    all_attrs = {}
    for pairs, attrs in results:
        all_pairs |= pairs
        all_attrs.update(attrs)
    # 双向匹配过滤（在合并后统一处理）
    ...
```

---

### 2.3 聚类后各社区的串行统计循环（`main.py` 第 196–247 行）⭐ **最大瓶颈**

**现状**：每个社区独立计算子图统计、Pfam 查询、写文件，完全串行。社区数量多（如数百个）时耗时显著。

```python
# 当前实现（串行）
for cid in range(len(clustering)):
    subgraph = graph.induced_subgraph(clustering[cid])
    sub_analyzer = SSNAnalyzer(subgraph)
    s = sub_analyzer.basic_stats()
    pfam_info = pfam_analyzer.domain_entropy(...)
    subgraph.write(graphml_path)
```

各社区之间**完全独立**，天然适合并行。

**并行化方案**：

```python
from concurrent.futures import ProcessPoolExecutor
import os

def process_cluster(args):
    cid, node_ids, edge_list, edge_attrs, output_dir, prefix, pfam_db, total_genomes = args
    # 在子进程中重建子图（避免序列化大图对象）
    import igraph as ig
    subgraph = ig.Graph()
    subgraph.add_vertices(node_ids)
    subgraph.vs['name'] = node_ids
    subgraph.add_edges(edge_list)
    for attr, vals in edge_attrs.items():
        subgraph.es[attr] = vals

    from ssnclust.analyzer import SSNAnalyzer, PfamDomainAnalyzer
    sub_analyzer = SSNAnalyzer(subgraph)
    s = sub_analyzer.basic_stats()
    sub_names = subgraph.vs['name']
    sub_genomes = len({n.split('|')[0] for n in sub_names if '|' in n})

    pfam_info = None
    if pfam_db:
        pfam_analyzer = PfamDomainAnalyzer(pfam_db)
        pfam_info = pfam_analyzer.domain_entropy(list(sub_names))
        pfam_analyzer.close()

    if output_dir:
        # 写节点列表
        sub_path = os.path.join(output_dir, f"{prefix}_{cid}.txt")
        with open(sub_path, 'w') as f:
            f.write('\n'.join(sub_names) + '\n')
        # 写 graphml
        graphml_path = os.path.join(output_dir, f"{prefix}_{cid}.graphml")
        subgraph.write(graphml_path)

    return cid, s, sub_genomes, pfam_info

# 调用示例
cluster_args = [
    (cid, ..., output_dir, prefix, pfam_db, total_genomes)
    for cid in range(len(clustering))
]
with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
    results = list(executor.map(process_cluster, cluster_args))

# 按 cid 排序后输出
results.sort(key=lambda x: x[0])
for cid, s, sub_genomes, pfam_info in results:
    # 打印表格行、写汇总 TSV ...
    ...
```

> **注意**：`igraph.Graph` 对象支持 pickle，但大图序列化开销较大。建议只传节点/边索引列表，在子进程中重建子图。

---

### 2.4 Pfam 结构域查询（`ssnclust/analyzer.py` → `PfamDomainAnalyzer`）

**现状**：对每个社区串行查询 SQLite，I/O 等待时间长。

**并行化方案**：

SQLite 支持多读连接，可用 `ThreadPoolExecutor`（I/O 密集型，线程即可）并发查询各社区，每个线程持有独立连接：

```python
from concurrent.futures import ThreadPoolExecutor

def query_pfam_for_cluster(args):
    db_path, evalue_threshold, seq_ids = args
    from ssnclust.analyzer import PfamDomainAnalyzer
    analyzer = PfamDomainAnalyzer(db_path, evalue_threshold)
    result = analyzer.domain_entropy(seq_ids)
    analyzer.close()
    return result

all_seq_ids = [list(graph.induced_subgraph(clustering[cid]).vs['name'])
               for cid in range(len(clustering))]

with ThreadPoolExecutor(max_workers=8) as executor:
    pfam_results = list(executor.map(
        query_pfam_for_cluster,
        [(pfam_db, evalue_threshold, ids) for ids in all_seq_ids]
    ))
```

---

### 2.5 Jaccard 矩阵乘法（`ssnclust/analyzer.py` → `_apply_jaccard_weighting_fast`）

**现状**：核心矩阵乘法 `A @ A` 在超大图时仍是单线程瓶颈（尽管已有 numpy/scipy 向量化）。

```python
A2 = A @ A  # 稀疏矩阵乘法，单线程
```

**并行化方案**：

- **方案 A（零代码改动）**：通过环境变量启用多线程 BLAS/OpenMP：
  ```bash
  export OMP_NUM_THREADS=8
  export MKL_NUM_THREADS=8
  ```

- **方案 B（GPU 加速）**：对超大网络，改用 `cupy` 进行 GPU 稀疏矩阵乘法：
  ```python
  import cupy as cp
  import cupyx.scipy.sparse as csp

  A_gpu = csp.csr_matrix(A)
  A2_gpu = A_gpu @ A_gpu
  intersection = cp.asnumpy(A2_gpu[us, vs])
  ```

---

### 2.6 TSV 分发输出（`main.py` 第 252–276 行）

**现状**：单线程流式扫描原始 TSV，将每行写入对应社区文件。

**并行化方案**：

将文件分块，多进程并行扫描，各进程将匹配行写入临时文件，最后合并。由于是 I/O 密集型，收益取决于磁盘带宽，SSD 环境下效果更明显。

```python
from concurrent.futures import ProcessPoolExecutor

def dispatch_chunk(args):
    lines, name_to_cid, query_idx, target_idx, tmp_dir = args
    # 将匹配行写入临时文件（按 cid 分组）
    tmp_files = {}
    for line in lines:
        cols = line.rstrip('\n').split('\t')
        q_cid = name_to_cid.get(cols[query_idx])
        t_cid = name_to_cid.get(cols[target_idx])
        if q_cid is not None and q_cid == t_cid:
            if q_cid not in tmp_files:
                tmp_files[q_cid] = open(os.path.join(tmp_dir, f"chunk_{q_cid}_{id(lines)}.tsv"), 'w')
            tmp_files[q_cid].write(line)
    for f in tmp_files.values():
        f.close()

# 最后合并各临时文件到目标 TSV
```

---

## 三、并行化优先级建议

| 优先级 | 模块 | 推荐方法 | 预期收益 |
|--------|------|----------|----------|
| ⭐⭐⭐ | 社区统计循环（`main.py`） | `ProcessPoolExecutor` | 社区数多时近线性加速 |
| ⭐⭐⭐ | TSV 解析（`utils.py`） | `pandas.read_csv` 或分块多进程 | 大文件显著加速 |
| ⭐⭐ | Pfam 查询（`analyzer.py`） | `ThreadPoolExecutor` | I/O 并发加速 |
| ⭐ | Jaccard 矩阵乘法 | 多线程 BLAS / GPU | 超大图时有效 |
| ⭐ | TSV 分发输出 | 分块多进程 | 受磁盘 I/O 限制 |

---

## 四、注意事项

1. **Python GIL 限制**：CPU 密集型任务（图统计、矩阵计算）应使用 `multiprocessing`（多进程）而非 `threading`（多线程）。

2. **igraph 对象序列化**：`igraph.Graph` 支持 pickle，但大图序列化开销较大，建议只传节点/边索引列表，在子进程中重建子图。

3. **SQLite 并发**：多进程写同一 SQLite 文件会有锁竞争；Pfam 查询为只读，用多线程即可，每个线程持有独立连接。

4. **内存压力**：并行化会增加内存占用，需根据实际数据规模和机器内存合理设置 `max_workers`，建议不超过 `os.cpu_count()`。

5. **输出顺序**：并行处理后结果顺序不确定，需在主进程按 `cid` 排序后再输出表格和写汇总文件。

6. **错误处理**：子进程中的异常不会自动传播，需在 `executor.map` 或 `future.result()` 处捕获并处理。

---

## 五、总结

**社区统计循环的并行化**是投入产出比最高的优化点，实现简单且收益显著，尤其在社区数量较多（>50）时效果明显。建议优先实施 2.3 节方案，再根据实际瓶颈逐步推进 TSV 解析和 Pfam 查询的并行化。
