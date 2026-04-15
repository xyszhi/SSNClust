import argparse
from ssnclust.generator import SSNGenerator
from ssnclust.analyzer import SSNAnalyzer
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
    parser.add_argument("--weight", choices=['fident', 'bits', 'fident_cov', 'none'], default='fident',
                        help="权重计算依据 (默认: fident)")
    parser.add_argument("--only-bidirectional", action="store_true",
                        help="只保留双向比对边 (默认保留所有比对边，包括单向)")
    parser.add_argument("--output", "-o", help="输出图文件路径 (推荐扩展名: .graphml)")
    parser.add_argument("--stats", action="store_true", help="显示网络基础统计信息")
    parser.add_argument("--jaccard", action="store_true", help="对边权重应用 Jaccard 加权")
    parser.add_argument("--cluster", choices=['leiden', 'spectral', 'mcl', 'nmf', 'sbm'], help="执行指定的聚类方法")
    parser.add_argument("--leiden-method",
                        choices=['modularity', 'cpm', 'rb_config', 'rber', 'significance', 'surprise'],
                        default='modularity', help="Leiden 聚类的具体方法 (默认: modularity)")
    parser.add_argument("--leiden-resolution", type=float, help="Leiden 聚类的分辨率参数 (仅--leiden-method为cpm、rber、rb_config时有效)")
    parser.add_argument("--mcl-inflation", type=float, default=2.0, help="MCL 聚类的膨胀系数 (默认: 2.0)")
    parser.add_argument("--sbm-type", choices=['standard', 'nested'], default='standard',
                        help="SBM 模型类型: standard (标准), nested (层次/嵌套, 自动推断层级)")
    parser.add_argument("--no-deg-corr", action="store_true",
                        help="关闭 SBM 的度校正 (默认开启，建议开启以处理 SSN 中的高通量节点)")
    parser.add_argument("--n-clusters", type=int, default=8, help="聚类数量 (用于谱聚类、NMF 等, 默认: 8)")

    args = parser.parse_args()

    clustering = None  # 防止 clustering 变量未定义导致 NameError

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
        bidirectional_only=args.only_bidirectional
    )

    print(f"SSN 构建完成:")

    analyzer = SSNAnalyzer(graph)

    if args.jaccard:
        # 如果指定了 --jaccard，应用 Jaccard 加权
        # 默认使用 generate 阶段生成的 'weight' 作为基础
        analyzer.apply_jaccard_weighting(base_weight='weight' if weight_by else 'none')

    stats = analyzer.basic_stats()
    print(f"  节点数: {stats['nodes']}")
    print(f"  边数: {stats['edges']}")

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

    if args.output:
        generator.save(args.output)

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
        res = args.resolution if args.resolution is not None else 1.0
        clustering = sbm_obj.cluster(
            sbm_type=args.sbm_type,
            degree_corrected=not args.no_deg_corr,
            resolution=res,
            weights=analyzer.active_weight
        )

    if args.cluster and clustering:
        graph.vs["cluster"] = clustering.membership
        print(f"聚类完成，共发现 {len(clustering)} 个社区。")
        # 如果有输出文件，重新保存一次以包含聚类结果
        if args.output:
            generator.save(args.output)


if __name__ == "__main__":
    main()
