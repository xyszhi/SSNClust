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
    *   **MCL (Markov Clustering)**: 生物信息学中 SSN 聚类的标准算法，擅长处理由于同源性导致的流模拟。
    *   **Leiden 算法**: Louvain 算法的改进版，速度快且能保证社区连通性，适合超大规模网络。
    *   **非负矩阵分解 (NMF)**: 将网络邻接矩阵分解，用于发现潜在的主题或重叠社区。
    *   **随机块模型 (SBM)**: 基于统计推理的方法，能够发现非典型的社区结构（如星型、二分图）。
    *   **谱聚类 (Spectral Clustering)**: 利用拉普拉斯矩阵的特征值进行降维和聚类，适用于全局结构分析。

## 2. 目录结构建议

```text
SSNClust/
├── data/               # 存放原始比对数据和测试用例
├── docs/               # 详细文档和理论背景
├── ssnclust/           # 核心源代码
│   ├── __init__.py
│   ├── generator.py    # SSN 生成逻辑 (igraph 实现)
│   ├── analyzer.py     # 特征计算逻辑 (聚集系数、流、切割等)
│   ├── clustering/     # 聚类算法实现
│   │   ├── __init__.py
│   │   ├── mcl_wrapper.py
│   │   ├── leiden_alg.py
│   │   ├── nmf_clust.py
│   │   ├── sbm_model.py
│   │   └── spectral.py
│   └── utils.py        # 文件解析、IO 辅助工具
├── tests/              # 单元测试
├── examples/           # 示例脚本
├── main.py             # 命令行入口
├── requirements.txt    # 依赖项 (igraph, louvain, leidenalg, scikit-learn, graph-tool等)
└── README.md
```

## 3. 技术路线与建议

1.  **性能优化**: 对于千万级别的边，建议使用 `graph-tool` 处理 SBM，因为它具有高效的 C++ 后端。`igraph` 处理基础特征和通用聚类。
2.  **权重标准化**: 不同比对工具的得分量度不同，建议在生成 SSN 时提供标准化的权重选项。
3.  **可视化**: 虽然主要作为命令行工具，但可以导出为 GraphML 或 JSON 格式，以便在 Cytoscape 或 Gephi 中进行后期可视化。

## 4. 后续开发计划

- [x] 实现基础 `SSNGenerator` 类，支持多种比对格式解析。
- [x] 实现 `SSNAnalyzer` 类，支持网络特征描述（聚集系数、连通分量、最大流/最小剪切等）。
- [ ] 集成 `leidenalg` 和 `markov_clustering` 库。
- [ ] 开发统一的 CLI 接口，方便用户一键从比对文件得到聚类结果。
