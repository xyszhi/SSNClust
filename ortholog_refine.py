#!/usr/bin/env python3
"""
ortholog_refine.py: 直系同源精修独立程序

输入：一个（通常是 main.py 聚类后导出的单个 cluster）比对结果 TSV 文件，
重新构建序列相似性网络 (SSN)，并执行"方案 C"精修算法：
从每个基因组的多条候选序列中挑选出一条代表，使输出子簇满足
"每个基因组恰好一条序列" 的直系同源原则。

用法示例：
    python ortholog_refine.py cluster_12.tsv -d ortholog_out --prefix cluster_12
"""

import argparse
import os

from ssnclust.generator import SSNGenerator
from ssnclust.orthology import refine_cluster_to_single_copy, default_genome_of


def main():
    parser = argparse.ArgumentParser(
        description="对输入比对结果 TSV 重新构建网络，并精修为满足'每基因组一条序列'的直系同源子簇（方案 C）"
    )
    parser.add_argument("input", help="输入比对结果文件 (TSV 格式)")

    # 网络构建参数（与 main.py 保持一致，便于复现聚类时的建网条件）
    parser.add_argument("--evalue", type=float, default=1e-5, help="E-value 阈值 (默认: 1e-5)")
    parser.add_argument("--identity", type=float, default=0.0, help="Identity 阈值 (0.0-1.0, 默认: 0.0)")
    parser.add_argument("--alnlen", type=int, default=0, help="比对长度阈值 (默认: 0)")
    parser.add_argument("--coverage", type=float, default=0.0, help="Coverage 阈值 (0.0-1.0, 默认: 0.0)")
    parser.add_argument("--cov-mode", choices=['min', 'max', 'any'], default='min',
                        help="Coverage 过滤模式 (默认: min)")
    parser.add_argument("--weight", choices=['fident', 'bits', 'fident_cov', 'fident_cov_harmonic', 'none'],
                        default='fident_cov', help="权重计算依据 (默认: fident_cov)")
    parser.add_argument("--only-bidirectional", action="store_true",
                        help="只保留双向比对边 (默认保留所有比对边，包括单向)")

    # 基因组 ID 提取参数
    parser.add_argument("--genome-delimiter", default="|",
                        help="序列名中用于分隔基因组 ID 与基因 ID 的分隔符 (默认: '|')")
    parser.add_argument("--genome-field-index", type=int, default=0,
                        help="分隔符拆分后取第几段作为基因组 ID (默认: 0，即取第一段)")

    # 精修算法参数
    parser.add_argument("--no-local-search", action="store_true",
                        help="关闭局部交换搜索优化 (默认开启)")
    parser.add_argument("--max-genomes-for-search", type=int, default=200,
                        help="启用局部交换搜索的基因组数上限 (默认: 200，超过则跳过局部搜索以保证性能)")
    parser.add_argument("--max-iterations", type=int, default=50,
                        help="局部交换搜索的最大迭代次数 (默认: 50)")

    # 输出参数
    parser.add_argument("--output-dir", "-d", required=True, help="精修结果输出目录")
    parser.add_argument("--prefix", default="ortholog", help="输出文件名前缀 (默认: ortholog)")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"正在从 {args.input} 生成 SSN...")
    generator = SSNGenerator(args.input)
    weight_by = args.weight if args.weight != 'none' else None
    graph = generator.generate(
        evalue_threshold=args.evalue,
        identity_threshold=args.identity,
        alnlen_threshold=args.alnlen,
        coverage_threshold=args.coverage,
        coverage_mode=args.cov_mode,
        weight_by=weight_by,
        bidirectional_only=args.only_bidirectional,
    )
    print(f"网络构建完成: {graph.vcount()} 个节点, {graph.ecount()} 条边")

    genome_of = lambda name: default_genome_of(name, args.genome_delimiter, args.genome_field_index)
    total_genomes = len({genome_of(n) for n in graph.vs['name']})
    print(f"输入网络共涉及 {total_genomes} 个基因组")

    refined_subclusters, extra_paralogs, stats = refine_cluster_to_single_copy(
        graph,
        genome_of=genome_of,
        weight_attr='weight' if weight_by else None,
        enable_local_search=not args.no_local_search,
        max_genomes_for_search=args.max_genomes_for_search,
        max_iterations=args.max_iterations,
    )

    print("\n精修结果统计:")
    print(f"  基因组总数:         {stats['n_genomes']}")
    print(f"  多拷贝基因组数:     {stats['n_multi_copy_genomes']}")
    print(f"  精修后子簇数量:     {stats['n_refined_subclusters']}")
    print(f"  代表序列数量:       {stats['n_representatives']}")
    print(f"  额外旁系同源数量:   {stats['n_extra_paralogs']}")
    print(f"  低置信度基因组数:   {stats['n_low_confidence_genomes']}")

    # 读取原始 TSV 表头及 query/target 列索引，用于按节点归属分发原始比对记录
    with open(args.input, 'r', encoding='utf-8') as f:
        tsv_header = f.readline()
    tsv_col_names = [c.strip() for c in tsv_header.rstrip('\n').split('\t')]
    query_idx = tsv_col_names.index('query') if 'query' in tsv_col_names else 0
    target_idx = tsv_col_names.index('target') if 'target' in tsv_col_names else 1
    min_col = max(query_idx, target_idx)

    name_to_sub = {}
    for sub_id, sub in enumerate(refined_subclusters):
        for name in sub.vs['name']:
            name_to_sub[name] = sub_id

    # 输出每个精修子簇：序列 ID 列表 (.txt)、子网络 (.graphml)、原始比对记录 (.tsv)
    sub_handles = {}
    for sub_id in range(len(refined_subclusters)):
        sub = refined_subclusters[sub_id]
        txt_path = os.path.join(args.output_dir, f"{args.prefix}_{sub_id}.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(sub.vs['name']) + '\n')

        graphml_path = os.path.join(args.output_dir, f"{args.prefix}_{sub_id}.graphml")
        sub.write(graphml_path)

        tsv_path = os.path.join(args.output_dir, f"{args.prefix}_{sub_id}.tsv")
        fh = open(tsv_path, 'w', encoding='utf-8')
        fh.write(tsv_header)
        sub_handles[sub_id] = fh

    with open(args.input, 'r', encoding='utf-8') as src:
        src.readline()  # 跳过表头
        for line in src:
            cols = line.rstrip('\n').split('\t')
            if len(cols) > min_col:
                q_sub = name_to_sub.get(cols[query_idx])
                t_sub = name_to_sub.get(cols[target_idx])
                if q_sub is not None and q_sub == t_sub:
                    sub_handles[q_sub].write(line)
    for fh in sub_handles.values():
        fh.close()

    # 输出额外旁系同源（落选序列）
    extra_path = os.path.join(args.output_dir, f"{args.prefix}_extra_paralogs.tsv")
    with open(extra_path, 'w', encoding='utf-8') as f:
        f.write("name\tgenome\treason\n")
        for item in extra_paralogs:
            f.write(f"{item['name']}\t{item['genome']}\t{item['reason']}\n")

    # 输出精修子簇汇总统计
    summary_path = os.path.join(args.output_dir, f"{args.prefix}_summary.tsv")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("subcluster\tnodes\tedges\tgenomes\tlow_confidence_genomes\n")
        for sub_id, sub in enumerate(refined_subclusters):
            low_conf = sub['low_confidence_genomes'] if 'low_confidence_genomes' in sub.attributes() else []
            n_genomes_sub = len({genome_of(n) for n in sub.vs['name']})
            f.write(f"{sub_id}\t{sub.vcount()}\t{sub.ecount()}\t{n_genomes_sub}\t{','.join(low_conf)}\n")

    print(f"\n结果已保存至目录: {args.output_dir}")
    print(f"  各子簇序列ID/子网络/比对记录: {args.prefix}_<id>.txt / .graphml / .tsv")
    print(f"  额外旁系同源列表: {os.path.basename(extra_path)}")
    print(f"  汇总统计: {os.path.basename(summary_path)}")


if __name__ == "__main__":
    main()
