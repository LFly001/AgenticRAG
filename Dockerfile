# ============================================================
# 多 Agent 协作智能知识库 — 后端 Dockerfile
# 基于 Python 3.10-slim
# ============================================================

FROM python:3.10-slim

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    # NLTK 数据路径
    NLTK_DATA=/app/nltk_data \
    # HuggingFace 缓存目录（必须可写，独立于只读的模型目录）
    HF_HOME=/app/hf_cache \
    # 国内用户可开启镜像加速
    # HF_ENDPOINT=https://hf-mirror.com \
    TZ=Asia/Shanghai

# 安装系统依赖
# - poppler-utils: PDF 解析
# - tesseract-ocr + 中英文语言包: OCR 文字识别
# - libmagic1: 文件类型检测（python-magic）
# - libxml2 / libxslt: XML/HTML 解析
# - build-essential: 部分 Python 包编译需要
RUN apt-get update && apt-get install -y --no-install-recommends \
    # PDF 解析
    poppler-utils \
    # OCR 支持
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    tesseract-ocr-eng \
    # 文件类型检测
    libmagic1 \
    # XML / HTML 解析
    libxml2 \
    libxslt1.1 \
    # 编译工具（chromadb / hnswlib 等需要）
    build-essential \
    # OpenGL 依赖（OpenCV / unstructured PDF 解析需要）
    libgl1 \
    libglib2.0-0 \
    # 其他实用工具
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=300 \
    -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# --- NLTK 数据（通过 volume 挂载，不在此处下载，避免国内网络超时）---
# 请在宿主机执行 download_nltk_data.py 脚本预先下载
# RUN python /app/scripts/download_nltk_data.py

# --- 复制应用代码 ---
COPY . .

# 确保启动脚本可执行
RUN chmod +x /app/entrypoint.sh

# 创建必要的目录
RUN mkdir -p /app/data/raw /app/chroma_db /app/ml_models /app/nltk_data /app/hf_cache

# 暴露 FastAPI 端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/docs || exit 1

# 入口脚本（自动下载模型 + 启动服务）
ENTRYPOINT ["/app/entrypoint.sh"]
# 默认启动 FastAPI（可在 docker-compose 中覆盖）
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--loop", "uvloop"]
