import argparse
from ssnclust.generator import SSNGenerator

def main():
    parser = argparse.ArgumentParser(description="SSNClust: 基于序列相似性网络 (SSN) 的序列聚类工具")
    parser.add_argument("input", help="输入比对结果文件 (TSV 格式)")
    parser.add_argument("--evalue", type=float, default=1e-5, help="E-value 阈值 (默认: 1e-5)")
    parser.add_argument("--identity", type=float, default=0.0, help="Identity 阈值 (0.0-1.0, 默认: 0.0)")
    parser.add_argument("--weight", choices=['evalue', 'fident', 'bits', 'none'], default='evalue', help="权重计算依据 (默认: evalue)")
    
    args = parser.parse_args()
    
    print(f"正在从 {args.input} 生成 SSN...")
    generator = SSNGenerator(args.input)
    weight_by = args.weight if args.weight != 'none' else None
    
    graph = generator.generate(
        evalue_threshold=args.evalue,
        identity_threshold=args.identity,
        weight_by=weight_by
    )
    
    print(f"SSN 构建完成:")
    print(f"  节点数: {graph.vcount()}")
    print(f"  边数: {graph.ecount()}")
    
    # 以后可以在这里添加分析或聚类逻辑

if __name__ == "__main__":
    main()
