# SSN 网络中 Jaccard 系数趋近于 0 的情况分析

## 背景

在 `SSNAnalyzer.apply_jaccard_weighting()` 中，边 $(u, v)$ 的 Jaccard 系数定义为：

$$J(u,v) = \frac{|N(u) \setminus \{u,v\} \cap N(v) \setminus \{u,v\}|}{|N(u) \setminus \{u,v\} \cup N(v) \setminus \{u,v\}|}$$

即：**排除端点自身后**，两节点邻居集合的交集大小除以并集大小。当交集为空（无共同第三方邻居）时，$J=0$；当并集也为空时，代码返回 $J=0.0$（`union > 0` 的保护分支）。

---

## SSN 网络的构建特点

SSN（序列相似性网络）由 `SSNGenerator` 从 MMseqs2/BLAST m8 格式的比对结果构建：

- **节点**：蛋白质/序列
- **边**：通过 E-value、序列一致性（fident）、比对长度（alnlen）、覆盖度（qcov/tcov）等阈值过滤后保留的比对对
- **权重**：可基于 `fident`、`bits` 或 `fident × coverage` 等指标

过滤阈值越严格，保留的边越少，**网络越稀疏，Jaccard 系数趋近于 0 的边也越多**。

---

## Jaccard 系数趋近于 0 的具体情形

### 1. 叶节点（度为 1 的节点）之间的边

**场景**：节点 $u$ 只与 $v$ 相连，节点 $v$ 只与 $u$ 相连（两者均为叶节点）。

```
u — v
```

- $N(u) \setminus \{u,v\} = \emptyset$
- $N(v) \setminus \{u,v\} = \emptyset$
- 交集 = 0，并集 = 0 → $J = 0.0$（代码保护分支）

**触发原因**：E-value 或 identity 阈值过严，某序列仅与一个序列有显著比对。

---

### 2. 叶节点与高度节点之间的边

**场景**：节点 $u$ 度为 1（仅连接 $v$），节点 $v$ 度数较高（连接多个节点）。

```
u — v — w1
        |— w2
        |— w3
```

- $N(u) \setminus \{u,v\} = \emptyset$
- $N(v) \setminus \{u,v\} = \{w_1, w_2, w_3, \ldots\}$
- 交集 = 0，并集 = $|N(v)|$ → $J = 0$

**触发原因**：稀有序列（孤立蛋白家族成员）仅与一个"枢纽"序列有比对，而该枢纽连接了大量其他序列。

---

### 3. 两个"星形"子图的桥接边

**场景**：两个星形子图（各有一个中心节点连接多个叶节点）之间仅有一条边相连。

```
leaf1 — hub_A — leaf2        leaf3 — hub_B — leaf4
                  \                  /
                   ——————————————————
```

- $N(\text{hub\_A}) \setminus \{\text{hub\_A}, \text{hub\_B}\} = \{\text{leaf1}, \text{leaf2}, \ldots\}$
- $N(\text{hub\_B}) \setminus \{\text{hub\_A}, \text{hub\_B}\} = \{\text{leaf3}, \text{leaf4}, \ldots\}$
- 两个叶节点集合完全不重叠 → 交集 = 0 → $J = 0$

**触发原因**：不同蛋白质家族之间仅有少数序列存在跨家族比对，形成"桥接边"，而两侧家族内部成员互不相似。

---

### 4. 线性链结构中的内部边

**场景**：序列按相似性形成线性链（每个节点仅与前后节点相连）。

```
A — B — C — D — E
```

- 对边 $(B, C)$：$N(B) \setminus \{B,C\} = \{A\}$，$N(C) \setminus \{B,C\} = \{D\}$
- 交集 = 0 → $J = 0$

**触发原因**：序列相似性呈梯度分布（如进化距离均匀的序列集合），相邻序列相似但不共享第三方邻居。

---

### 5. 高过滤阈值导致的整体稀疏化

**场景**：使用严格的 `identity_threshold`（如 ≥ 0.5）或 `coverage_threshold`（如 ≥ 0.8）后，大量潜在边被移除，导致：

- 原本三角形闭合的三元组（$u$-$v$-$w$）因 $u$-$w$ 或 $v$-$w$ 边被过滤而断开
- 共同邻居消失，$J(u,v)$ 从正值降为 0

**触发原因**：SSN 构建参数过严，网络从"小世界"结构退化为树状或森林结构，三角形闭合度（聚集系数）极低。

---

### 6. `bidirectional_only=True` 模式下的额外稀疏化

**场景**：启用双向过滤后，仅保留 A→B 和 B→A 均存在的比对。单向比对边被移除，进一步减少了共同邻居的可能性，加剧了 Jaccard≈0 的情况。

---

## 定量判断标准

| 网络特征 | Jaccard≈0 边的比例（估计） |
|---------|--------------------------|
| 平均度 < 2（极稀疏） | > 80% |
| 平均度 2–5（稀疏） | 50%–80% |
| 平均度 5–20（中等） | 20%–50% |
| 平均度 > 20（稠密） | < 20% |
| 聚集系数 < 0.1 | 大多数边 Jaccard≈0 |

可通过 `SSNAnalyzer.basic_stats()` 中的 `avg_degree` 和 `avg_clustering` 字段预判风险。

---

## 实际影响与建议

### 问题

当大量边的 Jaccard 系数为 0 时，`jaccard_weight = original_weight × 0 = 0`，这些边在后续加权聚类中实际上被"软删除"，可能导致：

1. **孤立节点**：叶节点失去唯一连接，从网络中脱离
2. **小簇消失**：仅通过桥接边连接的小型蛋白质家族与主网络断开
3. **聚类结果偏差**：Leiden/MCL 等算法在权重为 0 的边上无法有效传播社区信息

### 建议策略

1. **设置原始权重下限**：在 Jaccard 加权前，对原始权重设置最低阈值（如 `weight ≥ 0.3`），避免低质量边参与计算
2. **保留低 Jaccard 边**：对 Jaccard=0 的边保留一个小的基础权重（如 `max(jaccard_weight, ε)`，$\varepsilon = 0.01$）
3. **预检网络稀疏度**：在调用 `apply_jaccard_weighting()` 前，先用 `basic_stats()` 检查 `avg_degree` 和 `avg_clustering`，若平均度 < 3 则谨慎使用 Jaccard 加权
4. **分层加权**：对度数为 1 的叶节点边跳过 Jaccard 加权，直接保留原始权重
5. **调整 SSN 构建参数**：适当放宽 `identity_threshold` 或 `evalue_threshold`，使网络在 Jaccard 加权前具有足够的连通性

---

## 代码层面的关键逻辑

```python
# analyzer.py: apply_jaccard_weighting()
neighbors_u = adj_list[u] - {u, v}   # 排除端点自身
neighbors_v = adj_list[v] - {u, v}

intersection = len(neighbors_u.intersection(neighbors_v))
union = len(neighbors_u.union(neighbors_v))

j_coeff = intersection / union if union > 0 else 0.0  # union=0 时直接返回 0
jaccard_weights.append(old_weights[edge.index] * j_coeff)
```

注意：当 `union == 0` 时（两端点均为叶节点），代码返回 `j_coeff = 0.0`，这是**最极端的情形**，对应上述情形 1。

---

三类边的语义区分

| 边的类型 | J=0 的原因 | 网络语义 | 期望处理 | 
|---------|-----------|---------|---------|
| 叶节点边（度=1） | 邻居集合为空，无法计算 | 边本身可能很重要，只是网络太稀疏 | 保留原始权重 |
| 度差大、共享邻居少 | 一端邻居多但不重叠 | 可能是家族内部的"辐射边"，非真正桥接 | 提高 Jaccard 系数 | 
| 真正的桥接边 | 两侧邻居来自不同社区 | 连接不同蛋白质家族，应被削弱 | 施加 Jaccard 惩罚 |