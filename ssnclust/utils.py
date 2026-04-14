import csv
from typing import Iterator, Dict, Any

REQUIRED_COLUMNS = {'query', 'target', 'fident', 'alnlen', 'qcov', 'tcov', 'evalue', 'bits'}

def parse_m8_tsv(file_path: str) -> Iterator[Dict[str, Any]]:
    """
    解析 mmseqs2/blast m8 (outfmt 6) 格式的 TSV 文件。
    要求文件必须包含标题行，且标题行中必须包含以下列：
    query、target、fident、alnlen、qcov、tcov、evalue、bits。
    """
    try:
        with open(file_path, 'r') as f:
            first_line = f.readline()
            if not first_line:
                raise ValueError(f"文件为空，缺少标题行：{file_path}")
            headers = [h.strip() for h in first_line.rstrip('\n').split('\t')]
            missing = REQUIRED_COLUMNS - set(headers)
            if missing:
                raise ValueError(
                    f"TSV 文件缺少必要的列：{sorted(missing)}。"
                    f"文件路径：{file_path}\n"
                    f"当前标题行包含：{headers}"
                )
            f.seek(0)
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
    except FileNotFoundError:
        raise FileNotFoundError(f"文件未找到：{file_path}")