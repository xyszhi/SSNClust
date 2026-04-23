import argparse
import math
import os
from ssnclust.generator import SSNGenerator
from ssnclust.analyzer import SSNAnalyzer, PfamDomainAnalyzer
from ssnclust.clustering.leiden_alg import LeidenClustering
from ssnclust.clustering.spectral import SSNSpectralClustering
from ssnclust.clustering.mcl_wrapper import MCLClustering
from ssnclust.clustering.nmf_clust import NMFClustering
from ssnclust.clustering.sbm_model import SBMClustering


def main():
    parser = argparse.ArgumentParser(description="SSNClust: 基于序列相似性网络 (SSN) 的序列聚类工具")
    parser.add_argument("input", help="输入比对结果文件 (TSV 格式)")
    parser.add_argument("--evalue", type=float, default=1e-5, help="E-value 阈值 (默认: 1e-5)")
    parser.add_argument("--identity", type=float, default=0.0, help="Identity 阈值 (0.0-1.0, 默认: 0.0)")
    parser.add_argument("--alnlen", type=int, default=0, help="比对长度阈值 (默认: 0)")
    parser.add_argument("--coverage", type=float, default=0.0, help="Coverage 阈值 (0.0-1.0, 默认: 0.0)")
    parser.add_argument("--cov-mode", choices=['min', 'max', 'any'], default='min',
                        help="Coverage 过滤模式 (默认: min)")
    parser.add_argument("--weight", choices=['fident', 'bits', 'fident_cov', 'fident_cov_harmonic', 'none'], default='fident',
                        help="权重计算依据 (默认: fident)")
    parser.add_argument("--only-bidirectional", action="store_true",
                        help="只保留双向比对边 (默认保留所有比对边，包括单向)")
    parser.add_argument("--output-dir", "-d", help="聚类结果输出目录：每个社区的序列ID保存为 .txt 文件、子网络保存为 .graphml 文件，并生成汇总 TSV")
    parser.add_argument("--prefix", default="cluster", help="子网络相关文件的名称前缀 (默认: cluster)")
    parser.add_argument("--stats", action="store_true", help="显示网络基础统计信息")
    parser.add_argument("--jaccard", action="store_true", help="对边权重应用 Jaccard 加权")
    parser.add_argument("--cluster", choices=['leiden', 'mcl', 'spectral', 'nmf', 'sbm'], help="执行指定的聚类方法")
    parser.add_argument("--leiden-method",
                        choices=['modularity', 'cpm', 'rb_config', 'rber', 'significance', 'surprise'],
                        default='modularity', help="Leiden 聚类的具体方法 (默认: modularity)")
    parser.add_argument("--leiden-resolution", type=float,
                        help="Leiden 聚类的分辨率参数 (仅--leiden-method为cpm、rber、rb_config时有效)")
    parser.add_argument("--mcl-inflation", type=float, default=1.2, help="MCL 聚类的膨胀系数 (默认: 1.2)")
    parser.add_argument("--sbm-type", choices=['standard', 'nested'], default='standard',
                        help="SBM 模型类型: standard (标准), nested (层次/嵌套, 自动推断层级)")
    parser.add_argument("--no-deg-corr", action="store_true",
                        help="关闭 SBM 的度校正 (默认开启，建议开启以处理 SSN 中的高通量节点)")
    parser.add_argument("--n-clusters", type=int, default=8, help="聚类数量 (用于谱聚类、NMF 等, 默认: 8)")
    parser.add_argument("--pfam-db", help="hmmscan 结果 SQLite 数据库路径，用于计算每个 cluster 的结构域一致性熵值")
    parser.add_argument("--retained-fields", default="",
                        help="要保留在边属性中的额外字段，多个字段用逗号分隔 (默认: 空，仅保留 weight 或 jaccard_weight)")

    args = parser.parse_args()

    clustering = None  # 防止 clustering 变量未定义导致 NameError

    print(f"正在从 {args.input} 生成 SSN...")
    generator = SSNGenerator(args.input)
    weight_by = args.weight if args.weight != 'none' else None

    retained_fields = [f.strip() for f in args.retained_fields.split(',') if f.strip()] if args.retained_fields else []

    graph = generator.generate(
        evalue_threshold=args.evalue,
        identity_threshold=args.identity,
        alnlen_threshold=args.alnlen,
        coverage_threshold=args.coverage,
        coverage_mode=args.cov_mode,
        weight_by=weight_by,
        bidirectional_only=args.only_bidirectional,
        retained_fields=retained_fields
    )

    print(f"SSN 构建完成:")

    analyzer = SSNAnalyzer(graph)

    if args.jaccard:
        # 如果指定了 --jaccard，应用 Jaccard 加权
        # 默认使用 generate 阶段生成的 'weight' 作为基础
        analyzer.apply_jaccard_weighting(base_weight='weight' if weight_by else 'none')

    stats = analyzer.basic_stats()

    # 从节点名中提取基因组ID（格式: 基因组ID|序列ID）
    all_names = graph.vs['name']
    total_genomes = len({n.split('|')[0] for n in all_names if '|' in n})
    avg_seq_per_genome = stats['nodes'] / total_genomes if total_genomes > 0 else float('nan')

    print(f"  节点数: {stats['nodes']}")
    print(f"  边数: {stats['edges']}")
    print(f"  涉及基因组数: {total_genomes}")
    print(f"  平均每基因组序列数: {avg_seq_per_genome:.2f}")

    if args.stats:
        print(f"网络统计信息:")
        print(f"  密度: {stats['density']:.6f}")
        print(f"  是否连通: {'是' if stats['is_connected'] else '否'}")
        print(f"  连通分量数: {stats['components']}")
        print(f"  最大连通分量 (LCC) 节点数: {stats['lcc_size']} ({stats['lcc_percentage']:.2f}%)")
        print(f"  平均度 (Average Degree): {stats['avg_degree']:.2f}")
        print(f"  最大/最小度: {stats['max_degree']} / {stats['min_degree']}")
        print(f"  平均局部聚集系数: {stats['avg_clustering']:.6f}")
        if 'total_weight' in stats:
            print(f"权重统计 ({analyzer.active_weight}):")
            print(f"  总权重: {stats['total_weight']:.2f}")
            print(f"  平均权重: {stats['avg_weight']:.4f}")
            print(f"  最小权重: {stats['min_weight']:.4f}")
            print(f"  最大权重: {stats['max_weight']:.4f}")
            print(f"  权重标准差: {stats['sd_weight']:.4f}")


    if args.cluster == 'leiden':
        print(f"正在使用 Leiden ({args.leiden_method}) 进行聚类...")
        lc = LeidenClustering(graph)

        # 参数映射
        leiden_method_map = {
            'modularity': 'Modularity',
            'cpm': 'CPM',
            'rb_config': 'RBConfiguration',
            'rber': 'RBER',
            'significance': 'Significance',
            'surprise': 'Surprise'
        }
        partition_type = leiden_method_map.get(args.leiden_method, 'Modularity')

        # 确定使用的分辨率参数
        res = args.leiden_resolution
        kwargs = {}
        if partition_type in ['CPM', 'RBConfiguration', 'RBER']:
            if res is None:
                # 提供默认分辨率
                res = 0.01 if partition_type == 'CPM' else 1.0
            kwargs['resolution_parameter'] = res

        clustering = lc.cluster(partition_type=partition_type, weights=analyzer.active_weight, **kwargs)
    elif args.cluster == 'spectral':
        print(f"正在使用 Spectral Clustering (n_clusters={args.n_clusters}) 进行聚类...")
        sc = SSNSpectralClustering(graph)
        clustering = sc.cluster(n_clusters=args.n_clusters, weights=analyzer.active_weight)
    elif args.cluster == 'mcl':
        print(f"正在使用 MCL (inflation={args.mcl_inflation}) 进行聚类...")
        mcl_obj = MCLClustering(graph)
        clustering = mcl_obj.cluster(inflation=args.mcl_inflation, weights=analyzer.active_weight)
    elif args.cluster == 'nmf':
        print(f"正在使用 NMF (n_components={args.n_clusters}) 进行聚类...")
        nmf_obj = NMFClustering(graph)
        clustering = nmf_obj.cluster(n_components=args.n_clusters, weights=analyzer.active_weight)
    elif args.cluster == 'sbm':
        print(f"正在使用 SBM ({args.sbm_type}) 进行聚类...")
        sbm_obj = SBMClustering(graph)
        res = args.leiden_resolution if args.leiden_resolution is not None else 1.0
        clustering = sbm_obj.cluster(
            sbm_type=args.sbm_type,
            degree_corrected=not args.no_deg_corr,
            resolution=res,
            weights=analyzer.active_weight
        )

    if args.cluster and clustering:
        graph.vs["cluster"] = clustering.membership
        print(f"聚类完成，共发现 {len(clustering)} 个社区:")

        # 对每个社区计算网络统计信息并以表格形式输出
        # 列定义: (表头文字, 数据宽度)，宽度按显示宽度（中文占2）手动指定
        col_headers = ["社区", "节点数", "边数", "密度", "平均度", "最大度", "最小度", "平均聚集系数", "基因组数", "基因组占比", "序列/基因组"]
        col_widths = [4, 8, 8, 10, 8, 6, 6, 10, 8, 10, 10]

        # 构建表头：每列右对齐，用空格补足显示宽度（中文字符占2个显示位）
        def pad_header(text, width):
            # 计算中文字符数（每个中文占2个显示位）
            display_len = sum(2 if ord(c) > 127 else 1 for c in text)
            pad = width - display_len
            return " " * max(pad, 0) + text

        header_parts = [pad_header(h, w) for h, w in zip(col_headers, col_widths)]
        header = "  ".join(header_parts)
        sep_width = sum(col_widths) + 2 * (len(col_widths) - 1)
        print(header)
        print("-" * sep_width)

        # 如果指定了 --output-dir，预先读取 TSV 表头，确定 query/target 列索引（不全量读入）
        if args.output_dir:
            with open(args.input, 'r', encoding='utf-8') as _f:
                _tsv_header = _f.readline()
            _tsv_col_names = [c.strip() for c in _tsv_header.rstrip('\n').split('\t')]
            _query_idx = _tsv_col_names.index('query') if 'query' in _tsv_col_names else 0
            _target_idx = _tsv_col_names.index('target') if 'target' in _tsv_col_names else 1

        # 如果指定了 --output-dir，准备输出目录和汇总文件
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            summary_path = os.path.join(args.output_dir, f"summary.tsv")
            summary_file = open(summary_path, 'w', encoding='utf-8')  # noqa: WPS515 — closed explicitly at line 249
            pfam_header = "\tdomain_entropy\tseqs_with_hit\thit_ratio\tunique_domains\ttop_domains" if args.pfam_db else ""
            summary_file.write("cluster\tnodes\tedges\tdensity\tavg_degree\tmax_degree\tmin_degree\tavg_clustering\tgenomes\tgenome_ratio\tseq_per_genome" + pfam_header + "\n")

        pfam_analyzer = PfamDomainAnalyzer(args.pfam_db) if args.pfam_db else None

        for cid in range(len(clustering)):
            subgraph = graph.induced_subgraph(clustering[cid])
            sub_analyzer = SSNAnalyzer(subgraph)
            s = sub_analyzer.basic_stats()
            sub_names = subgraph.vs['name']
            sub_genomes = len({n.split('|')[0] for n in sub_names if '|' in n})
            genome_ratio = sub_genomes / total_genomes if total_genomes > 0 else 0.0
            seq_per_genome = s['nodes'] / sub_genomes if sub_genomes > 0 else float('nan')

            pfam_info = pfam_analyzer.domain_entropy(list(sub_names)) if pfam_analyzer else None

            pfam_suffix = ""
            if pfam_info:
                entropy_str = f"{pfam_info['domain_entropy']:.4f}" if not math.isnan(pfam_info['domain_entropy']) else "nan"
                pfam_suffix = f"  entropy={entropy_str}  hit={pfam_info['seqs_with_hit']}/{pfam_info['total_seqs']}({pfam_info['hit_ratio']:.2%})  uniq_domains={pfam_info['unique_domains']}"

            row = (
                f"{cid:{col_widths[0]}d}  "
                f"{s['nodes']:{col_widths[1]}d}  "
                f"{s['edges']:{col_widths[2]}d}  "
                f"{s['density']:{col_widths[3]}.6f}  "
                f"{s['avg_degree']:{col_widths[4]}.2f}  "
                f"{s['max_degree']:{col_widths[5]}d}  "
                f"{s['min_degree']:{col_widths[6]}d}  "
                f"{s['avg_clustering']:{col_widths[7]}.6f}  "
                f"{sub_genomes:{col_widths[8]}d}  "
                f"{genome_ratio:{col_widths[9]}.4f}  "
                f"{seq_per_genome:{col_widths[10]}.2f}"
                + pfam_suffix
            )
            print(row)

            if args.output_dir:
                # 保存子网络节点名（序列ID）
                sub_path = os.path.join(args.output_dir, f"{args.prefix}_{cid}.txt")
                with open(sub_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(sub_names) + '\n')
                # 保存子网络 graphml
                graphml_path = os.path.join(args.output_dir, f"{args.prefix}_{cid}.graphml")
                subgraph.write(graphml_path)
                # 写入汇总行
                seq_per_genome_str = f"{seq_per_genome:.2f}" if sub_genomes > 0 else "nan"
                pfam_tsv = ""
                if pfam_info:
                    entropy_tsv = f"{pfam_info['domain_entropy']:.4f}" if not math.isnan(pfam_info['domain_entropy']) else "nan"
                    top_str = "|".join(f"{d}:{c}" for d, c in pfam_info['top_domains'])
                    pfam_tsv = f"\t{entropy_tsv}\t{pfam_info['seqs_with_hit']}\t{pfam_info['hit_ratio']:.4f}\t{pfam_info['unique_domains']}\t{top_str}"
                summary_file.write(
                    f"{cid}\t{s['nodes']}\t{s['edges']}\t{s['density']:.6f}\t"
                    f"{s['avg_degree']:.2f}\t{s['max_degree']}\t{s['min_degree']}\t"
                    f"{s['avg_clustering']:.6f}\t{sub_genomes}\t{genome_ratio:.4f}\t{seq_per_genome_str}{pfam_tsv}\n"
                )
        print("-" * sep_width)

        # 如果指定了 --output-dir，一次流式扫描原始 TSV，将每行分发到对应 cluster 的 TSV 文件
        # 避免将整个大文件全量读入内存
        if args.output_dir:
            # 构建 节点名 -> cluster id 的映射
            _name_to_cid = {}
            for _cid in range(len(clustering)):
                for _name in graph.induced_subgraph(clustering[_cid]).vs['name']:
                    _name_to_cid[_name] = _cid
            # 打开所有 cluster 的 TSV 文件句柄
            _min_col = max(_query_idx, _target_idx)
            _tsv_handles = {
                _cid: open(os.path.join(args.output_dir, f"{args.prefix}_{_cid}.tsv"), 'w', encoding='utf-8')
                for _cid in range(len(clustering))
            }
            for _fh in _tsv_handles.values():
                _fh.write(_tsv_header)
            with open(args.input, 'r', encoding='utf-8') as _src:
                _src.readline()  # 跳过表头
                for _line in _src:  # 逐行流式读取，不占用额外内存
                    _cols = _line.rstrip('\n').split('\t')
                    if len(_cols) > _min_col:
                        _q_cid = _name_to_cid.get(_cols[_query_idx])
                        _t_cid = _name_to_cid.get(_cols[_target_idx])
                        if _q_cid is not None and _q_cid == _t_cid:
                            _tsv_handles[_q_cid].write(_line)
            for _fh in _tsv_handles.values():
                _fh.close()

        # 计算并输出跨 cluster 边比例（聚类质量评估）
        ratio_metrics = analyzer.inter_cluster_edge_ratio(clustering)
        print(f"\n聚类质量评估:")
        print(f"  cluster 数量:       {ratio_metrics['num_clusters']}")
        print(f"  总边数:             {ratio_metrics['total_edges']}")
        print(f"  cluster 内部边数:   {ratio_metrics['intra_cluster_edges']}")
        print(f"  跨 cluster 边数:    {ratio_metrics['inter_cluster_edges']}")
        print(f"  跨 cluster 边比例:  {ratio_metrics['inter_cluster_ratio']:.4f}  (越低越好)")
        if 'inter_cluster_weight_ratio' in ratio_metrics:
            print(f"  跨 cluster 加权比例:{ratio_metrics['inter_cluster_weight_ratio']:.4f}  (越低越好)")

        if pfam_analyzer:
            pfam_analyzer.close()

        if args.output_dir:
            summary_file.close()
            print(f"各社区序列ID及子网络已保存至目录: {args.output_dir}")
            print(f"汇总统计文件: {summary_path}")
            ssn_path = os.path.join(args.output_dir, "ssn.graphml")
            generator.save(ssn_path)

if __name__ == "__main__":
    main()
