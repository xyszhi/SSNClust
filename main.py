import argparse
from ssnclust.generator import SSNGenerator

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
    print(f"  节点数: {graph.vcount()}")
    print(f"  边数: {graph.ecount()}")
    
    if args.output:
        generator.save(args.output)
    
    # 以后可以在这里添加分析或聚类逻辑

if __name__ == "__main__":
    main()
