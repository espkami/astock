# ── 构建阶段 ──────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# 1. 安装所有依赖（requirements.txt 已包含 akshare 的全部子依赖 + akracer）
# 2. 单独 --no-deps 安装 akshare，跳过 py-mini-racer（arm64 无预编译 wheel）
#    akracer 已在 requirements.txt 中，提供 py_mini_racer 模块的完整替代
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt \
    && pip install --no-cache-dir --prefix=/install --no-deps akshare==1.18.62

# ── 运行阶段 ──────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 从构建阶段复制已安装的包（/install → /usr/local，路径自动对齐）
COPY --from=builder /install /usr/local

# 复制源码
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# 数据持久化目录（挂载 volume 到此处）
RUN mkdir -p /app/data

ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# DEBUG=1 启用 debugpy 调试模式，否则生产模式
CMD ["sh", "-c", "\
  if [ \"$DEBUG\" = '1' ]; then \
    python -m debugpy --listen 0.0.0.0:5678 --wait-for-client \
      -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload; \
  else \
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 1; \
  fi"]
