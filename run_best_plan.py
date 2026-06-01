#!/usr/bin/env python3
import os
import json
import subprocess
import argparse
import sys
from compare import load_result, score_result, compute_scores

def find_best_plan(json_files):
    results = []
    valid_files = []
    for f in json_files:
        try:
            results.append(load_result(f))
            valid_files.append(f)
        except Exception as e:
            print(f"警告: 无法加载 {f}: {e}")
    
    if not results:
        return None, None, 0
    
    all_details = [score_result(d) for d in results]
    scores = compute_scores(all_details)
    
    best_idx = scores.index(max(scores))
    best_score = scores[best_idx]
    return valid_files[best_idx], results[best_idx], best_score

def main():
    parser = argparse.ArgumentParser(description="自动选择评分最高的 SSNClust 方案并重新运行以输出聚类文件")
    parser.add_argument("input_tsv", help="原始比对结果 TSV 文件路径")
    parser.add_argument("json_dir", help="包含各方案 JSON 结果的目录")
    parser.add_argument("--output-dir", "-o", help="重新运行后的结果输出目录（默认：./clusters/<tsv_name>）")
    parser.add_argument("--pfam-db", help="覆盖 JSON 中的 Pfam 数据库路径（如果服务器路径不一致）")
    parser.add_argument("--min-score", type=float, default=80.0, help="执行聚类的最低评分阈值 (默认: 80.0)")
    parser.add_argument("--dry-run", action="store_true", help="仅显示将执行的命令，不实际运行")
    
    args = parser.parse_args()
    
    # 1. 获取相关的 JSON 文件
    tsv_basename = os.path.splitext(os.path.basename(args.input_tsv))[0]
    json_files = [
        os.path.join(args.json_dir, f) 
        for f in os.listdir(args.json_dir) 
        if f.startswith(f"{tsv_basename}_P") and f.endswith('.json')
    ]
    
    if not json_files:
        print(f"错误: 目录 {args.json_dir} 中未找到与 {tsv_basename} 相关的 JSON 文件 (格式应为 {tsv_basename}_P*.json)")
        sys.exit(1)
    
    # 2. 找到评分最高的方案
    print(f"正在分析 {len(json_files)} 个方案以寻找最优方案...")
    best_file, best_data, best_score = find_best_plan(json_files)
    
    if not best_file:
        print("未找到有效方案")
        sys.exit(1)
    
    print(f"\n找到最优方案: {os.path.basename(best_file)}")
    print(f"该方案得分: {best_score:.4f}")
    
    # 检查分值阈值
    if best_score < args.min_score:
        print(f"警告: 最优方案得分 ({best_score:.4f}) 低于设定的阈值 ({args.min_score:.4f})。")
        print("取消执行聚类。")
        sys.exit(0)
    
    params = best_data.get('parameters', {})
    print(f"该方案参数: {params}")
    
    # 3. 构造 main.py 命令
    tsv_name = os.path.splitext(os.path.basename(args.input_tsv))[0]
    out_dir = args.output_dir or os.path.join("clusters", tsv_name)
    
    # 基础命令
    cmd = [
        sys.executable, "/home/mselab/projects/Streptomyces/03.pangenome/denovo/SSNClust-main/main.py",
        args.input_tsv,
        "--output-dir", out_dir,
        "--prefix", tsv_name
    ]
    
    # 添加来自 JSON 的参数
    mapping = {
        'evalue': '--evalue',
        'identity': '--identity',
        'alnlen': '--alnlen',
        'coverage': '--coverage',
        'cov_mode': '--cov-mode',
        'weight': '--weight',
        'cluster': '--cluster',
        'leiden_method': '--leiden-method',
        'leiden_resolution': '--leiden-resolution',
        'mcl_inflation': '--mcl-inflation',
        'sbm_type': '--sbm-type',
        'n_clusters': '--n-clusters'
    }
    
    for key, flag in mapping.items():
        val = params.get(key)
        if val is not None:
            cmd.extend([flag, str(val)])
            
    if params.get('only_bidirectional'):
        cmd.append('--only-bidirectional')
    if params.get('jaccard'):
        cmd.append('--jaccard')
    if params.get('no_deg_corr'):
        cmd.append('--no-deg-corr')
        
    # 处理 Pfam DB
    pfam = args.pfam_db or params.get('pfam_db')
    if pfam:
        cmd.extend(['--pfam-db', pfam])
        
    # 添加默认开启的 --stats
    cmd.append('--stats')

    print(f"\n准备执行重新聚类...")
    print(f"执行命令: {' '.join(cmd)}")
    
    if args.dry_run:
        print("\n[Dry Run] 命令未实际执行。")
    else:
        try:
            subprocess.run(cmd, check=True)
            print(f"\n重新聚类完成！结果保存在: {out_dir}")
        except subprocess.CalledProcessError as e:
            print(f"\n执行失败: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
