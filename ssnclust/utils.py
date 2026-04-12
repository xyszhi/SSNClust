import csv
from typing import Iterator, Dict, Any

def parse_m8_tsv(file_path: str) -> Iterator[Dict[str, Any]]:
    """
    解析 mmseqs2/blast m8 (outfmt 6) 格式的 TSV 文件。
    假设第一行是标题行（如 examples/cluster_999_3.tsv 所示）。
    """
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            # 自动转换数值类型，除了特定的标识符字段
            for key, value in row.items():
                if key in ('query', 'target'):
                    continue
                try:
                    # 尝试转换为浮点数
                    float_val = float(value)
                    # 如果转换成功，判断是否为整数
                    if float_val.is_integer():
                        row[key] = int(float_val)
                    else:
                        row[key] = float_val
                except (ValueError, TypeError):
                    # 保持原始字符串
                    pass
            yield row
