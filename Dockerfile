FROM python:3.13-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY pyproject.toml ./
COPY ssnclust/ ./ssnclust/
COPY main.py ./

# 安装 Python 依赖
RUN pip install --no-cache-dir \
    "igraph>=1.0.0" \
    "leidenalg>=0.11.0" \
    "markov-clustering>=0.0.6.dev0" \
    "scikit-learn>=1.8.0" \
    "scikit-network>=0.33.5" \
    "scipy>=1.17.1" \
    "numpy"

# 创建数据挂载目录
RUN mkdir -p /data /output

ENTRYPOINT ["python", "/app/main.py"]
