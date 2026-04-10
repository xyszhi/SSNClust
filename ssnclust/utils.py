import csv
from typing import Iterator, Dict, Any

def parse_m8_tsv(file_path: str) -> Iterator[Dict[str, Any]]:
    """
    解析 mmseqs2/blast m8 (outfmt 6) 格式的 TSV 文件。
    假设第一行是标题行（如 examples/cluster_999_3.tsv 所示）。
    """
    with open(file_path, 'r') as f:
        # 我们可以根据 examples/cluster_999_3.tsv 的第一行来判断字段
        # query	target	fident	alnlen	mismatch	gapopen	qstart	qend	qlen	qcov	tstart	tend	tlen	qcov	evalue	bits
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            # 转换数值类型
            try:
                if 'fident' in row: row['fident'] = float(row['fident'])
                if 'alnlen' in row: row['alnlen'] = int(row['alnlen'])
                if 'evalue' in row: row['evalue'] = float(row['evalue'])
                if 'bits' in row: row['bits'] = float(row['bits'])
            except (ValueError, KeyError):
                pass
            yield row
