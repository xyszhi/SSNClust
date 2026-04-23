# SSNClust: 基于序列相似性网络 (SSN) 的蛋白质/核酸序列聚类工具

SSNClust 是一个专为大规模生物序列数据设计的聚类分析工具。它通过将 BLAST、Diamond 或 MMseqs2 的比对结果转换为序列相似性网络 (SSN)，并集成多种先进的网络特征提取和图聚类算法，帮助研究人员发现蛋白质家族、结构域或功能单元。

## 1. 项目架构设计

本项目采用模块化设计，主要包含以下三个核心板块：

### 1.1 SSN 生成模块 (Network Generation)
*   **功能**: 将序列比对软件（如 BLASTp, Diamond, MMseqs2）生成的输出格式（如 m8/outfmt 6）转换为网络图。
*   **实现**: 使用 `python-igraph` 作为核心引擎。
*   **输入**: 比对结果文件（TSV 格式，必须包含标题行，且包含以下列：`query`、`target`、`fident`、`alnlen`、`qcov`、`tcov`、`evalue`、`bits`）。
*   **逻辑**:
    *   节点 (Nodes): 代表每一条唯一的序列。
    *   边 (Edges): 代表序列间的相似性。
    *   边权重/属性: 可根据 E-value、Identity、Bit-score 或 Coverage 设定阈值进行过滤并赋予权重。
    *   支持 `--only-bidirectional` 参数，只保留双向比对的边。
    *   支持多种权重计算方式：`fident`、`bits`、`fident_cov`（fident × coverage 乘积）、`fident_cov_harmonic`（fident 与 coverage 的调和平均数）。
    *   重复边（双向比对产生的 A→B 和 B→A）自动合并：数值属性取均值，非数值属性取第一个值。

### 1.2 网络特征描述模块 (Network Characterization)
*   **功能**: 量化 SSN 的拓扑结构特征。
*   **核心指标**:
    *   **基础统计**: 节点数、边数、密度、连通性、连通分量数、最大连通分量大小、平均/最大/最小度、平均局部聚集系数、权重统计（总和、均值、标准差等）。
    *   **局部聚集系数 (Local Clustering Coefficient)**: 衡量节点邻居间的紧密程度。
    *   **最小切割 (Minimum Cut)**: 识别网络中的薄弱连接。
    *   **最大流 (Maximum Flow)**: 评估节点间的潜在路径容量。
    *   **模块度 (Modularity)**: 评估网络划分为不同社区的质量。
    *   **连通分量 (Connected Components)**: 识别独立的序列簇。
    *   **Jaccard 系数加权 (Jaccard Weighting)**: 通过节点邻居重叠度校正边权重（`--jaccard`）。利用 scipy 稀疏矩阵向量化计算，在千万级边的大图上性能显著优于逐边循环。
    *   **跨 cluster 边比例**: 聚类完成后自动计算，用于评估聚类质量（越低越好）。
*   **Pfam 结构域分析**: 通过 `--pfam-db` 指定 hmmscan 结果 SQLite 数据库，可计算每个 cluster 的结构域信息熵，评价 cluster 内序列的结构域一致程度。

### 1.3 聚类算法集成模块 (Clustering Algorithms)
*   **功能**: 提供多种算法对 SSN 进行自动划分。
*   **算法配置**:
    *   **MCL (Markov Clustering)**: 基于 `markov-clustering` 包实现。生物信息学中 SSN 聚类的标准算法，擅长处理由于同源性导致的流模拟。支持 `--mcl-inflation` 参数调节紧密度（默认 1.2）。
    *   **Leiden 算法**: 基于 `leidenalg` 包实现。支持全部 6 种划分模型（`modularity`, `cpm`, `rb_config`, `rber`, `significance`, `surprise`）。速度快且能保证社区连通性，适合超大规模网络。通过 `--leiden-resolution` 指定分辨率参数（仅 `cpm`、`rber`、`rb_config` 模型有效）。
    *   **非负矩阵分解 (NMF)**: 基于 `scikit-learn` 实现。将网络邻接矩阵 A 分解为两个低秩非负矩阵 W 和 H（$A \approx WH$），用于发现潜在的主题或重叠社区。适合揭示蛋白质超家族中由结构域重组形成的重叠聚集模式。
    *   **随机块模型 (SBM)**: 优先使用 `graph-tool` 进行贝叶斯推断（支持度校正和嵌套结构），若未安装则自动回退至 `scikit-network` 的 Potts 模型模拟。支持 `--sbm-type standard/nested` 和 `--no-deg-corr` 参数。SBM 不仅能发现紧密的团块，还能发现**"核心-边缘" (Core-Periphery)** 结构，适合识别超家族中的祖先核心序列与衍生外围序列。
    *   **谱聚类 (Spectral Clustering)**: 利用拉普拉斯矩阵的特征值进行降维和聚类，适用于全局结构分析。能发现"非球形"的聚集模式，对长链状或复杂流形结构的蛋白超家族尤为灵敏。

## 2. 目录结构

```text
SSNClust/
├── examples/           # 示例比对数据 (TSV)
├── ssnclust/           # 核心源代码
│   ├── generator.py    # SSN 生成逻辑 (igraph 实现)
│   ├── analyzer.py     # 特征计算逻辑 (聚集系数、流、切割、Jaccard 加权、Pfam 分析等)
│   ├── clustering/     # 聚类算法实现
│   │   ├── mcl_wrapper.py   # MCL 聚类
│   │   ├── leiden_alg.py    # Leiden 聚类
│   │   ├── nmf_clust.py     # NMF 聚类
│   │   ├── sbm_model.py     # SBM 聚类
│   │   └── spectral.py      # 谱聚类
│   └── utils.py        # 文件解析、数值转换辅助工具
├── main.py             # 命令行入口
├── pyproject.toml      # 项目依赖与配置
└── README.md
```

## 3. 命令行参数

```
usage: main.py [-h] [--evalue EVALUE] [--identity IDENTITY] [--alnlen ALNLEN]
               [--coverage COVERAGE] [--cov-mode {min,max,any}]
               [--weight {fident,bits,fident_cov,fident_cov_harmonic,none}]
               [--only-bidirectional] [--output-dir OUTPUT_DIR] [--stats]
               [--jaccard] [--cluster {leiden,mcl,spectral,nmf,sbm}]
               [--leiden-method {modularity,cpm,rb_config,rber,significance,surprise}]
               [--leiden-resolution LEIDEN_RESOLUTION]
               [--mcl-inflation MCL_INFLATION]
               [--sbm-type {standard,nested}] [--no-deg-corr]
               [--n-clusters N_CLUSTERS] [--pfam-db PFAM_DB]
               input
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `input` | 输入比对结果文件 (TSV 格式) | 必填 |
| `--evalue` | E-value 阈值 | `1e-5` |
| `--identity` | Identity 阈值 (0.0–1.0) | `0.0` |
| `--alnlen` | 比对长度阈值 | `0` |
| `--coverage` | Coverage 阈值 (0.0–1.0) | `0.0` |
| `--cov-mode` | Coverage 过滤模式：`min`（两者都需满足）/ `max`/`any`（任一满足即可） | `min` |
| `--weight` | 权重计算依据 | `fident` |
| `--only-bidirectional` | 只保留双向比对边 | 关闭 |
| `--output-dir` / `-d` | 聚类结果输出目录 | 无 |
| `--stats` | 显示网络基础统计信息 | 关闭 |
| `--jaccard` | 对边权重应用 Jaccard 加权 | 关闭 |
| `--cluster` | 聚类方法：`leiden` / `mcl` / `spectral` / `nmf` / `sbm` | 无 |
| `--leiden-method` | Leiden 划分模型 | `modularity` |
| `--leiden-resolution` | Leiden/SBM 分辨率参数 | 自动 |
| `--mcl-inflation` | MCL 膨胀系数 | `1.2` |
| `--sbm-type` | SBM 模型类型：`standard` / `nested` | `standard` |
| `--no-deg-corr` | 关闭 SBM 度校正 | 关闭 |
| `--n-clusters` | 聚类数量（谱聚类、NMF） | `8` |
| `--pfam-db` | hmmscan 结果 SQLite 数据库路径 | 无 |

## 4. 使用示例

### 4.1 基础统计与生成
```bash
python main.py examples/cluster_500.tsv --stats
```

### 4.2 MCL 聚类（推荐用于 SSN）
```bash
python main.py examples/cluster_999.tsv \
    --weight fident_cov --cov-mode any \
    --jaccard --cluster mcl --mcl-inflation 1.2 \
    --stats --pfam-db examples/hmmscan_results.db \
    -d ~/Desktop/mcl_results
```

### 4.3 Leiden 聚类 (使用 CPM 模型)
```bash
python main.py examples/cluster_500.tsv \
    --cluster leiden --leiden-method cpm --leiden-resolution 0.05
```

### 4.4 SBM 聚类 (推荐使用嵌套模型)
```bash
python main.py examples/cluster_500.tsv --cluster sbm --sbm-type nested
```

### 4.5 谱聚类
```bash
python main.py examples/cluster_500.tsv --cluster spectral --n-clusters 10
```

### 4.6 输出说明
指定 `--output-dir` 后，程序将在该目录下生成：
- `cluster_<id>.txt`：每个 cluster 的序列 ID 列表
- `cluster_<id>.graphml`：每个 cluster 的子网络（可在 Cytoscape/Gephi 中可视化）
- `cluster_summary.tsv`：所有 cluster 的统计汇总（节点数、边数、密度、基因组数、Pfam 熵值等）
- `ssn.graphml`：完整 SSN 网络

## 5. 服务器部署

### 5.1 一键安装（推荐）

将项目克隆到服务器后，运行安装脚本即可自动创建虚拟环境并安装所有依赖：

```bash
git clone <your-repo-url> SSNClust
cd SSNClust
bash install.sh
```

安装完成后激活环境并运行：

```bash
source .venv/bin/activate
python main.py examples/cluster_10.tsv --stats
```

如需同时提示 `graph-tool` 安装方式（用于高性能 SBM）：

```bash
bash install.sh --with-graph-tool
```

### 5.2 Docker 部署

适合无法修改服务器 Python 环境、或需要隔离运行环境的场景。

**构建镜像：**
```bash
docker build -t ssnclust .
```

**运行（将本地数据目录挂载进容器）：**
```bash
docker run --rm \
    -v /path/to/your/data:/data \
    -v /path/to/output:/output \
    ssnclust \
    /data/your_alignment.tsv \
    --weight fident_cov --cov-mode any \
    --jaccard --cluster mcl \
    -d /output/results
```

**说明：**
- `/data`：挂载输入数据目录（TSV 比对文件、hmmscan SQLite 数据库等）
- `/output`：挂载输出目录，聚类结果将写入此处
- 容器内 `ENTRYPOINT` 已设为 `python /app/main.py`，直接传参即可

### 5.3 在 HPC 集群（SLURM）上提交作业

```bash
#!/bin/bash
#SBATCH --job-name=ssnclust
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=ssnclust_%j.log

source /path/to/SSNClust/.venv/bin/activate

python /path/to/SSNClust/main.py \
    /path/to/cluster_999.tsv \
    --weight fident_cov --cov-mode any \
    --jaccard --cluster mcl --mcl-inflation 1.2 \
    --stats --pfam-db /path/to/hmmscan_results.db \
    -d /path/to/output/mcl_results
```

将上述内容保存为 `run_ssnclust.sh`，然后提交：

```bash
sbatch run_ssnclust.sh
```

## 6. 技术路线与建议

1.  **性能优化**: Jaccard 加权使用 scipy 稀疏矩阵向量化实现，在 500 万条边的网络上相比纯 Python 循环提速约 4 倍。对于千万级别的边，建议使用 `graph-tool` 处理 SBM，因为它具有高效的 C++ 后端。
2.  **权重选择**: `fident_cov`（identity × coverage）综合考虑了比对质量和覆盖度，通常是 SSN 分析的推荐权重；`fident_cov_harmonic` 对低覆盖度的比对惩罚更强。
3.  **Jaccard 加权**: `--jaccard` 参数通过节点邻居重叠度校正边权重，能有效降低高度数枢纽节点对聚类的干扰，推荐在大型网络中使用。
4.  **可视化**: 导出为 `.graphml` 格式，推荐在 Cytoscape 或 Gephi 中进行后期可视化。

## 6. 依赖

| 包 | 用途 |
|----|------|
| `python-igraph` | 核心图引擎 |
| `numpy` / `scipy` | Jaccard 加权向量化计算 |
| `leidenalg` | Leiden 聚类 |
| `markov-clustering` | MCL 聚类 |
| `scikit-learn` | NMF 聚类、谱聚类 |
| `scikit-network` | SBM 回退实现（Potts 模型） |
| `graph-tool` *(可选)* | SBM 贝叶斯推断（高性能，需单独安装） |

## 7. 开发进度

- [x] 实现基础 `SSNGenerator` 类，支持多种比对格式解析、双向过滤、多种权重计算。
- [x] 实现 `SSNAnalyzer` 类，支持网络特征描述（聚集系数、连通分量、最大流/最小切割、模块度、跨 cluster 边比例等）。
- [x] 实现高性能 Jaccard 系数加权（scipy 稀疏矩阵向量化）。
- [x] 集成 `leidenalg`, `markov_clustering`, `scikit-learn`, `scikit-network` 聚类库。
- [x] 深度集成 `graph-tool` SBM 推断（自动回退至 scikit-network）。
- [x] 集成 Pfam 结构域分析（基于 hmmscan SQLite 数据库），计算每个 cluster 的结构域信息熵。
- [x] 开发统一的 CLI 接口，支持聚类结果输出（序列 ID 列表、子网络 graphml、汇总 TSV）。
- [ ] 增加可视化绘图模块 (基于 matplotlib/plotly)。
- [ ] 支持更多序列比对工具的直接调用封装。
