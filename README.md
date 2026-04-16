# SSNClust: 基于序列相似性网络 (SSN) 的蛋白质/核酸序列聚类工具

SSNClust 是一个专为大规模生物序列数据设计的聚类分析工具。它通过将 BLAST、Diamond 或 MMseqs2 的比对结果转换为序列相似性网络 (SSN)，并集成多种先进的网络特征提取和图聚类算法，帮助研究人员发现蛋白质家族、结构域或功能单元。

## 1. 项目架构设计

本项目采用模块化设计，主要包含以下三个核心板块：

### 1.1 SSN 生成模块 (Network Generation)
*   **功能**: 将序列比对软件（如 BLASTp, Diamond, MMseqs2）生成的输出格式（如 m8/outfmt 6）转换为网络图。
*   **实现**: 使用 `python-igraph` 作为核心引擎。
*   **输入**: 比对结果文件（TSV格式，包含 Query, Target, E-value, Identity, Alignment length 等）。
*   **逻辑**:
    *   节点 (Nodes): 代表每一条唯一的序列。
    *   边 (Edges): 代表序列间的相似性。
    *   边权重/属性: 可根据 E-value (通常取 -log10)、Identity 或 Bit-score 设定阈值进行过滤并赋予权重。
    *   支持动态阈值过滤，通过调整 E-value 阈值来观察网络连通性的变化。

### 1.2 网络特征描述模块 (Network Characterization)
*   **功能**: 量化 SSN 的拓扑结构特征。
*   **核心指标**:
    *   **局部聚集系数 (Local Clustering Coefficient)**: 衡量节点邻居间的紧密程度。
    *   **最小切割 (Minimum Cut)**: 识别网络中的薄弱连接。
    *   **最大流 (Maximum Flow)**: 评估节点间的潜在路径容量。
    *   **模块度 (Modularity)**: 评估网络划分为不同社区的质量。
    *   **连通分量 (Connected Components)**: 识别独立的序列簇。
    *   **Jaccard 系数加权 (Jaccard Weighting)**: 通过节点邻居重叠度校正边权重。

### 1.3 聚类算法集成模块 (Clustering Algorithms)
*   **功能**: 提供多种算法对 SSN 进行自动划分。
*   **算法配置**:
    *   **MCL (Markov Clustering)**: 基于 `markov-clustering` 包实现。生物信息学中 SSN 聚类的标准算法，擅长处理由于同源性导致的流模拟。支持 `--mcl-inflation` 参数调节紧密度。
    *   **Leiden 算法**: 基于 `leidenalg` 包实现。支持全部 6 种划分模型（`modularity`, `cpm`, `rb_config`, `rber`, `significance`, `surprise`）。速度快且能保证社区连通性，适合超大规模网络。
    *   **非负矩阵分解 (NMF)**: 基于 `scikit-learn` 实现。将网络邻接矩阵分解，用于发现潜在的主题或重叠社区。
    这是一种将邻接矩阵视为“图像”或“信号”的处理方式。
    数学原理：将邻接矩阵 A 分解为两个低秩的非负矩阵 W 和 H，即 $A \approx WH$。
    生物学意义：NMF 的核心在于**“部分构成整体”**。在蛋白质超家族中，一个蛋白可能包含多个结构域。NMF 可以揭示出哪些蛋白共同拥有某一组“潜在特征”（即结构域模块），从而识别出由结构域重组形成的重叠聚集模式。
    *   **随机块模型 (SBM)**: 优先使用 `graph-tool` 进行贝叶斯推断（支持度校正和嵌套结构），若未安装则自动回退至 `scikit-network` 的 Potts 模型模拟。
    如果说 MCL 是基于过程的，SBM 就是基于结构的概率模型。数学原理：假设网络中的节点属于不同的“块”（Blocks），块内和块间的连接概率由一个概率矩阵控制。通过贝叶斯推断来寻找最可能的块结构。
    针对超家族的威力：SBM 不仅能发现紧密的团块（Clique），还能发现**“核心-边缘” (Core-Periphery)** 结构。这对于识别超家族中的“祖先核心序列”和“衍生的外围序列”非常有效。
    *   **谱聚类 (Spectral Clustering)**: 利用拉普拉斯矩阵的特征值进行降维和聚类，适用于全局结构分析。 
    谱聚类不直接处理邻接矩阵，而是处理拉普拉斯矩阵 (Laplacian Matrix) L=D−A（其中 D 是度矩阵）。
    数学原理：通过计算 L 的前 k 个最小特征值对应的特征向量，将高维的蛋白序列空间投影到低维空间，再利用 K−means 进行聚类。
    生物学意义：它能发现那些**“非球形”**的聚集模式。如果你的超家族 CC 呈现长链状或复杂的流形结构（这在连续演化的蛋白中很常见），谱聚类往往比 MCL 更灵敏。

## 2. 目录结构

```text
SSNClust/
├── examples/           # 示例比对数据 (TSV)
├── ssnclust/           # 核心源代码
│   ├── generator.py    # SSN 生成逻辑 (igraph 实现)
│   ├── analyzer.py     # 特征计算逻辑 (聚集系数、流、切割等)
│   ├── clustering/     # 聚类算法实现
│   │   ├── mcl_wrapper.py
│   │   ├── leiden_alg.py
│   │   ├── nmf_clust.py
│   │   ├── sbm_model.py
│   │   └── spectral.py
│   └── utils.py        # 文件解析、数值转换辅助工具
├── main.py             # 命令行入口
├── pyproject.toml      # 项目依赖与配置
└── README.md
```

## 3. 使用示例

### 3.1 基础统计与生成
```bash
python main.py examples/cluster_500.tsv --stats --output my_ssn.graphml
```

### 3.2 Leiden 聚类 (使用 CPM 模型)
```bash
python main.py examples/cluster_500.tsv --cluster leiden --leiden-method cpm --resolution 0.05
```

### 3.3 MCL 聚类
```bash
python main.py examples/cluster_500.tsv --cluster mcl --mcl-inflation 1.5
```

### 3.4 SBM 聚类 (推荐使用嵌套模型)
```bash
python main.py examples/cluster_500.tsv --cluster sbm --sbm-type nested
```

## 4. 技术路线与建议

1.  **性能优化**: 对于千万级别的边，建议使用 `graph-tool` 处理 SBM，因为它具有高效的 C++ 后端。`igraph` 处理基础特征和通用聚类。
2.  **权重标准化**: 目前支持基于 Identity, Bit-score 或 E-value (-log10) 自动计算权重。
3.  **可视化**: 导出为 `.graphml` 格式，推荐在 Cytoscape 或 Gephi 中进行后期可视化。

## 5. 开发进度

- [x] 实现基础 `SSNGenerator` 类，支持多种比对格式解析。
- [x] 实现 `SSNAnalyzer` 类，支持网络特征描述（聚集系数、连通分量、最大流/最小剪切等）。
- [x] 集成 `leidenalg`, `markov_clustering`, `scikit-learn`, `scikit-network` 聚类库。
- [x] 深度集成 `graph-tool` SBM 推断。
- [x] 开发统一的 CLI 接口，方便用户一键从比对文件得到聚类结果。
- [ ] 增加可视化绘图模块 (基于 matplotlib/plotly)。
- [ ] 支持更多序列比对工具的直接调用封装。
