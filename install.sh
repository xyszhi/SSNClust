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
    echo "错误: 未找到 Python，请先安装 Python 3.13+。"
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
    "igraph>=1.0.0" \
    "leidenalg>=0.11.0" \
    "markov-clustering>=0.0.6.dev0" \
    "scikit-learn>=1.8.0" \
    "scikit-network>=0.33.5" \
    "scipy>=1.17.1" \
    "numpy"

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
