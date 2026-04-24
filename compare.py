#!/usr/bin/env python3
"""
SSNClust 聚类方案比较工具
用法: python compare.py result1.json result2.json [result3.json ...]
"""
import argparse
import json
import math
import os


def load_result(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def describe_params(p):
    """将参数字典转为简短描述字符串"""
    method = p.get('cluster', 'unknown')
    parts = [method]
    if method == 'leiden':
        parts.append(p.get('leiden_method', 'modularity'))
        if p.get('leiden_resolution') is not None:
            parts.append(f"res={p['leiden_resolution']}")
    elif method == 'mcl':
        parts.append(f"inflation={p.get('mcl_inflation', 1.2)}")
    elif method == 'sbm':
        parts.append(p.get('sbm_type', 'standard'))
    elif method in ('spectral', 'nmf'):
        parts.append(f"k={p.get('n_clusters', 8)}")
    parts.append(f"weight={p.get('weight', '?')}")
    if p.get('jaccard'):
        parts.append('jaccard')
    return ' | '.join(parts)


def score_result(d):
    """
    综合评分（越高越好），满分100分，由以下指标加权合成：
      - 跨cluster边比例        权重30  (越低越好)
      - 加权跨cluster边比例    权重15  (越低越好，若无则用边比例代替)
      - pfam平均熵             权重15  (越低越好，若无pfam数据则跳过，按比例重新分配)
      - pfam平均命中率         权重15  (越高越好，若无pfam数据则跳过)
      - 子网络平均密度         权重10  (越高越好，内部连接越紧密)
      - 子网络平均度           权重5   (越高越好，节点连接越丰富)
      - 子网络平均聚集系数     权重10  (越高越好，局部聚集性越强)
      - 子网络平均序列/基因组比 权重0  (仅展示，不参与评分，反映基因组覆盖均匀性)
    各指标先在所有方案中归一化到 [0,1]，再加权求和。
    返回 details_dict
    """
    cl = d.get('clustering') or {}
    details = {
        'inter_ratio': cl.get('inter_cluster_ratio'),
        'inter_w_ratio': cl.get('inter_cluster_weight_ratio'),
        'pfam_entropy': None,
        'pfam_hit_ratio': None,
        'avg_density': None,
        'avg_degree': None,
        'avg_clustering': None,
        'avg_seq_per_genome': None,
    }
    clusters = cl.get('clusters', [])
    entropies = [c['pfam']['domain_entropy'] for c in clusters
                 if c.get('pfam') and c['pfam'].get('domain_entropy') is not None]
    hit_ratios = [c['pfam']['hit_ratio'] for c in clusters
                  if c.get('pfam') and c['pfam'].get('hit_ratio') is not None]
    if entropies:
        details['pfam_entropy'] = sum(entropies) / len(entropies)
    if hit_ratios:
        details['pfam_hit_ratio'] = sum(hit_ratios) / len(hit_ratios)
    densities = [c['density'] for c in clusters if c.get('density') is not None and not (isinstance(c['density'], float) and math.isnan(c['density']))]
    degrees = [c['avg_degree'] for c in clusters if c.get('avg_degree') is not None and not (isinstance(c['avg_degree'], float) and math.isnan(c['avg_degree']))]
    clusterings = [c['avg_clustering'] for c in clusters if c.get('avg_clustering') is not None and not (isinstance(c['avg_clustering'], float) and math.isnan(c['avg_clustering']))]
    spg = [c['seq_per_genome'] for c in clusters
           if c.get('seq_per_genome') is not None and not (isinstance(c['seq_per_genome'], float) and math.isnan(c['seq_per_genome']))]
    if densities:
        details['avg_density'] = sum(densities) / len(densities)
    if degrees:
        details['avg_degree'] = sum(degrees) / len(degrees)
    if clusterings:
        details['avg_clustering'] = sum(clusterings) / len(clusterings)
    if spg:
        details['avg_seq_per_genome'] = sum(spg) / len(spg)
    return details


def normalize(values, lower_is_better=True):
    """将一组值归一化到 [0,1]，None 值保持 None。lower_is_better=True 时最小值得1分。"""
    valid = [v for v in values if v is not None]
    if not valid or len(set(valid)) == 1:
        return [0.5 if v is not None else None for v in values]
    vmin, vmax = min(valid), max(valid)
    result = []
    for v in values:
        if v is None:
            result.append(None)
        else:
            norm = (v - vmin) / (vmax - vmin)
            result.append(1 - norm if lower_is_better else norm)
    return result


def compute_scores(all_details):
    """跨所有方案归一化后计算综合得分"""
    keys = ['inter_ratio', 'inter_w_ratio', 'pfam_entropy', 'pfam_hit_ratio',
            'avg_density', 'avg_degree', 'avg_clustering']
    lower_better = {'inter_ratio': True, 'inter_w_ratio': True,
                    'pfam_entropy': True, 'pfam_hit_ratio': False,
                    'avg_density': False, 'avg_degree': False, 'avg_clustering': False}
    weights = {'inter_ratio': 30, 'inter_w_ratio': 15,
               'pfam_entropy': 15, 'pfam_hit_ratio': 15,
               'avg_density': 10, 'avg_degree': 5, 'avg_clustering': 10}

    norm_map = {}
    for k in keys:
        vals = [d[k] for d in all_details]
        norm_map[k] = normalize(vals, lower_is_better=lower_better[k])

    scores = []
    for i in range(len(all_details)):
        total_w = 0
        total_score = 0
        for k in keys:
            nv = norm_map[k][i]
            if nv is not None:
                total_score += nv * weights[k]
                total_w += weights[k]
        scores.append(total_score / total_w * 100 if total_w > 0 else 0)
    return scores


def print_comparison(results, labels):
    all_details = [score_result(d) for d in results]
    scores = compute_scores(all_details)

    # 表格列定义
    col_label_w = max(len(lb) for lb in labels) + 2
    col_label_w = max(col_label_w, 20)

    print("\n" + "=" * 100)
    print("  SSNClust 聚类方案比较报告")
    print("=" * 100)

    # 参数描述
    print("\n【方案参数】")
    for lb, d in zip(labels, results):
        p = d.get('parameters', {})
        print(f"  {lb:<{col_label_w}} {describe_params(p)}")
        print(f"  {'':>{col_label_w}} evalue={p.get('evalue')}  identity={p.get('identity')}  "
              f"coverage={p.get('coverage')}  cov_mode={p.get('cov_mode')}")

    # 网络信息（应相同，只打印一次作参考）
    net = results[0].get('network', {})
    print(f"\n【网络基本信息】（以第一个文件为准）")
    print(f"  节点数: {net.get('nodes')}  边数: {net.get('edges')}  "
          f"基因组数: {net.get('total_genomes')}  密度: {net.get('density', 0):.6f}")

    # 聚类质量指标对比表
    print("\n【聚类质量指标对比】")
    header = (f"  {'方案':<{col_label_w}}  {'cluster数':>9}  {'跨边比例':>9}  "
              f"{'加权跨边比':>10}  {'pfam熵(avg)':>11}  {'pfam命中率':>10}  "
              f"{'子网密度':>9}  {'子网平均度':>10}  {'聚集系数':>9}  {'序列/基因组':>11}  {'综合得分':>8}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    ranked = sorted(zip(scores, labels, results, all_details), reverse=True)
    best_score = ranked[0][0]

    for sc, lb, d, det in zip(scores, labels, results, all_details):
        cl = d.get('clustering') or {}
        num_cl = cl.get('num_clusters', '-')
        inter = f"{det['inter_ratio']:.4f}" if det['inter_ratio'] is not None else '-'
        inter_w = f"{det['inter_w_ratio']:.4f}" if det['inter_w_ratio'] is not None else '-'
        entropy = f"{det['pfam_entropy']:.4f}" if det['pfam_entropy'] is not None else '-'
        hit = f"{det['pfam_hit_ratio']:.4f}" if det['pfam_hit_ratio'] is not None else '-'
        density = f"{det['avg_density']:.4f}" if det['avg_density'] is not None else '-'
        degree = f"{det['avg_degree']:.2f}" if det['avg_degree'] is not None else '-'
        clust = f"{det['avg_clustering']:.4f}" if det['avg_clustering'] is not None else '-'
        spg = f"{det['avg_seq_per_genome']:.2f}" if det['avg_seq_per_genome'] is not None else '-'
        mark = " ★" if abs(sc - best_score) < 0.01 else ""
        print(f"  {lb:<{col_label_w}}  {num_cl:>9}  {inter:>9}  {inter_w:>10}  "
              f"{entropy:>11}  {hit:>10}  "
              f"{density:>9}  {degree:>10}  {clust:>9}  {spg:>11}  {sc:>7.1f}{mark}")

    # cluster大小分布
    print("\n【Cluster 大小分布】")
    size_header = f"  {'方案':<{col_label_w}}  {'min':>6}  {'max':>6}  {'avg':>8}  {'中位数':>8}  {'单节点数':>8}"
    print(size_header)
    print("  " + "-" * (len(size_header) - 2))
    for lb, d in zip(labels, results):
        cl = d.get('clustering') or {}
        clusters = cl.get('clusters', [])
        if not clusters:
            print(f"  {lb:<{col_label_w}}  {'N/A':>6}")
            continue
        sizes = sorted(c['nodes'] for c in clusters)
        avg = sum(sizes) / len(sizes)
        median = sizes[len(sizes) // 2]
        singletons = sum(1 for s in sizes if s == 1)
        print(f"  {lb:<{col_label_w}}  {min(sizes):>6}  {max(sizes):>6}  {avg:>8.1f}  {median:>8}  {singletons:>8}")

    # pfam 结构域分析（若有）
    has_pfam = any(det['pfam_entropy'] is not None for det in all_details)
    if has_pfam:
        print("\n【Pfam 结构域分析（各 cluster 平均）】")
        pf_header = (f"  {'方案':<{col_label_w}}  {'有pfam数据cluster数':>18}  "
                     f"{'平均熵':>8}  {'最大熵':>8}  {'平均命中率':>10}  {'平均unique域数':>13}")
        print(pf_header)
        print("  " + "-" * (len(pf_header) - 2))
        for lb, d in zip(labels, results):
            cl = d.get('clustering') or {}
            clusters = cl.get('clusters', [])
            pfam_clusters = [c for c in clusters if c.get('pfam')]
            if not pfam_clusters:
                print(f"  {lb:<{col_label_w}}  {'无pfam数据':>18}")
                continue
            entropies = [c['pfam']['domain_entropy'] for c in pfam_clusters
                         if c['pfam'].get('domain_entropy') is not None]
            hits = [c['pfam']['hit_ratio'] for c in pfam_clusters]
            uniq = [c['pfam']['unique_domains'] for c in pfam_clusters]
            avg_e = sum(entropies) / len(entropies) if entropies else float('nan')
            max_e = max(entropies) if entropies else float('nan')
            avg_h = sum(hits) / len(hits)
            avg_u = sum(uniq) / len(uniq)
            print(f"  {lb:<{col_label_w}}  {len(pfam_clusters):>18}  "
                  f"{avg_e:>8.4f}  {max_e:>8.4f}  {avg_h:>10.4f}  {avg_u:>13.1f}")

    # 推荐
    print("\n【综合推荐】")
    print(f"  最佳方案: {ranked[0][1]}")
    print(f"  参数描述: {describe_params(ranked[0][2].get('parameters', {}))}")
    print(f"  综合得分: {ranked[0][0]:.1f} / 100")
    print()
    print("  评分说明:")
    print("    - 跨cluster边比例 (权重30%): 越低说明聚类内聚性越好")
    print("    - 加权跨cluster边比例 (权重15%): 考虑边权重的内聚性指标")
    print("    - pfam结构域平均熵 (权重15%): 越低说明cluster功能越纯")
    print("    - pfam平均命中率 (权重15%): 越高说明序列功能注释覆盖越全")
    print("    - 子网络平均密度 (权重10%): 越高说明cluster内部连接越紧密")
    print("    - 子网络平均度   (权重 5%): 越高说明节点平均连接数越多")
    print("    - 子网络平均聚集系数 (权重10%): 越高说明局部三角结构越丰富")
    print("    - 子网络序列/基因组比 (仅展示，不参与评分): 反映基因组覆盖均匀性")
    print("    （各指标在所有方案间归一化后加权求和，满分100分）")
    print()

    # 排名
    print("  完整排名:")
    for rank, (sc, lb, d, det) in enumerate(ranked, 1):
        print(f"    {rank}. {lb}  ({sc:.1f}分)  {describe_params(d.get('parameters', {}))}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="SSNClust 聚类方案比较工具：对比多个 --json 输出文件，评估并推荐最佳聚类方案",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python compare.py examples/leiden.json examples/mcl.json examples/mcl2.json
  python compare.py results/*.json
        """
    )
    parser.add_argument("files", nargs='+', metavar="JSON_FILE",
                        help="一个或多个由 SSNClust --json 生成的 JSON 结果文件")
    parser.add_argument("--label", nargs='+', metavar="LABEL",
                        help="为每个文件指定显示标签（数量须与文件数一致）")
    args = parser.parse_args()

    if len(args.files) < 2:
        print("警告: 只提供了一个文件，无法进行比较。请提供至少两个 JSON 文件。")

    results = []
    labels = []
    for i, path in enumerate(args.files):
        if not os.path.exists(path):
            print(f"错误: 文件不存在: {path}")
            continue
        try:
            d = load_result(path)
            results.append(d)
            if args.label and i < len(args.label):
                labels.append(args.label[i])
            else:
                labels.append(os.path.basename(path))
        except Exception as e:
            print(f"错误: 无法读取 {path}: {e}")

    if not results:
        print("没有可用的结果文件，退出。")
        return

    print_comparison(results, labels)


if __name__ == "__main__":
    main()
