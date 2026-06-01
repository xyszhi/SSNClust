import argparse
import os

try:
    import pandas as pd
except ImportError:
    print("Error: pandas is not installed. Please install it using 'pip install pandas'.")
    exit(1)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT_LIBS = True
except ImportError:
    print("Warning: matplotlib or seaborn not installed. Plotting will be skipped.")
    print("To enable plotting, install them using: pip install matplotlib seaborn")
    HAS_PLOT_LIBS = False

def analyze_tsv(file_path, output_prefix=None):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    # 读取 TSV 文件
    print(f"Reading {file_path}...")
    df = pd.read_csv(file_path, sep='\t')

    target_cols = ['fident', 'alnlen', 'qlen', 'tlen', 'qcov', 'tcov']
    
    # 检查列是否存在
    missing_cols = [col for col in target_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns {missing_cols} in {file_path}")
        return

    # 描述性统计
    stats = df[target_cols].describe()
    print("\nDescriptive Statistics:")
    print(stats)

    # 计算平均值用于绘图
    df['avg_cov'] = (df['qcov'] + df['tcov']) / 2
    df['avg_len'] = (df['qlen'] + df['tlen']) / 2

    # 如果没有指定输出前缀，则使用文件名（不含扩展名）
    if output_prefix is None:
        output_prefix = os.path.splitext(os.path.basename(file_path))[0]

    # 保存统计结果到 CSV
    stats_file = f"{output_prefix}_stats.csv"
    stats.to_csv(stats_file)
    print(f"\nSaved statistics to {stats_file}")

    # 绘图
    if HAS_PLOT_LIBS:
        print("\nGenerating plots...")
        sns.set_theme(style="whitegrid")
        # 修改为 2x2 子图布局
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=False)
        axes = axes.flatten() # 展平以便于循环
        
        plot_configs = [
            ('fident', 'Distribution of fident'),
            ('avg_cov', 'Distribution of Avg Coverage (qcov & tcov)'),
            ('alnlen', 'Distribution of alnlen'),
            ('avg_len', 'Distribution of Avg Length (qlen & tlen)')
        ]
        
        for i, (col, title) in enumerate(plot_configs):
            sns.histplot(df[col], kde=True, ax=axes[i], color='skyblue', bins=30)
            axes[i].set_title(title, fontsize=12)
            axes[i].set_xlabel(col, fontsize=10)
            axes[i].set_ylabel('Frequency', fontsize=10)
            
            # 统一设置刻度字号
            axes[i].tick_params(axis='both', which='major', labelsize=9)

        # 设置整个图的 title 为文件名（去掉扩展名，并加粗）
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        fig.suptitle(base_name, fontsize=16, fontweight='bold')

        plt.tight_layout()
        plot_file = f"{output_prefix}_distribution.png"
        plt.savefig(plot_file, dpi=300)
        print(f"Saved plot to {plot_file}")
        # plt.show() # 在无 GUI 环境下可能报错，默认关闭
    else:
        print("\nSkipping plotting as dependencies are missing.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze TSV file (fident, qcov, tcov) and plot distributions.")
    parser.add_argument("input", help="Path to the input TSV file.")
    parser.add_argument("--prefix", help="Output prefix for stats and plots.")
    
    args = parser.parse_args()
    analyze_tsv(args.input, args.prefix)
