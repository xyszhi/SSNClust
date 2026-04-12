import argparse
from ssnclust.generator import SSNGenerator
from ssnclust.analyzer import SSNAnalyzer

def main():
    parser = argparse.ArgumentParser(description="SSNClust: 基于序列相似性网络 (SSN) 的序列聚类工具")
    parser.add_argument("input", help="输入比对结果文件 (TSV 格式)")
    parser.add_argument("--evalue", type=float, default=1e-5, help="E-value 阈值 (默认: 1e-5)")
    parser.add_argument("--identity", type=float, default=0.0, help="Identity 阈值 (0.0-1.0, 默认: 0.0)")
    parser.add_argument("--alnlen", type=int, default=0, help="比对长度阈值 (默认: 0)")
    parser.add_argument("--coverage", type=float, default=0.0, help="Coverage 阈值 (0.0-1.0, 默认: 0.0)")
    parser.add_argument("--cov-mode", choices=['min', 'max', 'any'], default='min', help="Coverage 过滤模式 (默认: min)")
    parser.add_argument("--weight", choices=['evalue', 'fident', 'bits', 'none'], default='evalue', help="权重计算依据 (默认: evalue)")
    parser.add_argument("--output", "-o", help="输出图文件路径 (推荐扩展名: .graphml)")
    parser.add_argument("--stats", action="store_true", help="显示网络基础统计信息")
    parser.add_argument("--jaccard", action="store_true", help="对边权重应用 Jaccard 加权")
    
    args = parser.parse_args()
    
    print(f"正在从 {args.input} 生成 SSN...")
    generator = SSNGenerator(args.input)
    weight_by = args.weight if args.weight != 'none' else None
    
    graph = generator.generate(
        evalue_threshold=args.evalue,
        identity_threshold=args.identity,
        alnlen_threshold=args.alnlen,
        coverage_threshold=args.coverage,
        coverage_mode=args.cov_mode,
        weight_by=weight_by
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
            print(f"  最大权重: {stats['max_weight']:.4f}")
    
    if args.output:
        generator.save(args.output)
    
    # 以后可以在这里添加分析或聚类逻辑

if __name__ == "__main__":
    main()
