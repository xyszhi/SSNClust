#!/usr/bin/env bash
# SSNClust 服务器一键安装脚本
# 用法: bash install.sh [--with-graph-tool]
set -e

WITH_GT=0
for arg in "$@"; do
    [ "$arg" = "--with-graph-tool" ] && WITH_GT=1
done

echo "=== SSNClust 安装脚本 ==="

# 检查 Python 版本
PY=$(command -v python3 || command -v python || true)
if [ -z "$PY" ]; then
    echo "错误: 未找到 Python，请先安装 Python 3.11。"
    exit 1
fi
PY_VER=$("$PY" -c "import sys; print('%d.%d' % sys.version_info[:2])")
echo "检测到 Python $PY_VER ($PY)"

# 创建虚拟环境
if [ ! -d ".venv" ]; then
    echo "正在创建虚拟环境 .venv ..."
    "$PY" -m venv .venv
fi
source .venv/bin/activate

# 升级 pip
pip install --upgrade pip -q

# 安装核心依赖
echo "正在安装核心依赖..."
pip install -q \
    "igraph>=0.10.0,<0.12" \
    "leidenalg>=0.10.0,<0.11.0" \
    "scipy>=1.11.0,<1.14" \
    "markov-clustering>=0.0.6.dev0" \
    "scikit-learn>=1.3.0,<1.6" \
    "scikit-network>=0.32.0,<0.34" \
    "numpy<2"

# 可选：graph-tool（需要 conda 或系统包管理器，pip 不支持）
if [ "$WITH_GT" -eq 1 ]; then
    echo ""
    echo "注意: graph-tool 无法通过 pip 安装。"
    echo "请参考官方文档手动安装: https://graph-tool.skewed.de/installation"
    echo "  Debian/Ubuntu: sudo apt install python3-graph-tool"
    echo "  conda:         conda install -c conda-forge graph-tool"
fi

echo ""
echo "=== 安装完成 ==="
echo "激活环境: source .venv/bin/activate"
echo "运行示例: python main.py examples/cluster_10.tsv --stats"
