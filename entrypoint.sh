#!/bin/bash
# ============================================================
# 多 Agent 知识库 — 容器启动脚本
# 职责：
#   1. 验证 NLTK 数据完整性
#   2. 检查/下载 BGE-M3 嵌入模型 + BGE-Reranker 重排序模型
#   3. 启动 FastAPI 服务（含多 Agent 协作图）
# ============================================================

set -e

echo "=============================================="
echo "  Multi-Agent Knowledge Base — Entrypoint"
echo "=============================================="

# --- 颜色输出 ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info()  { echo -e "${GREEN}[INFO]${NC}  $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ==========================================
# 0. 检查是否跳过模型检查（前端服务不需要模型）
# ==========================================
if [ "${SKIP_MODEL_CHECK}" = "true" ]; then
    log_info "SKIP_MODEL_CHECK=true，跳过模型检查，直接启动服务..."
    exec "$@"
fi

# ==========================================
# 1. 检查 NLTK 数据（应通过 volume 挂载，宿主机预先下载）
# ==========================================
log_info "检查 NLTK 数据..."

NLTK_RESOURCES=(
    "tokenizers/punkt"
    "tokenizers/punkt_tab"
    "taggers/averaged_perceptron_tagger_eng"
    "corpora/stopwords"
)

MISSING_NLTK=()
for res in "${NLTK_RESOURCES[@]}"; do
    if ! python -c "import nltk; nltk.data.find('$res')" 2>/dev/null; then
        MISSING_NLTK+=("$res")
    fi
done

if [ ${#MISSING_NLTK[@]} -gt 0 ]; then
    log_error "缺少 NLTK 资源: ${MISSING_NLTK[*]}"
    log_error "请在宿主机执行: python scripts/download_nltk_data.py"
    log_error "然后重新挂载 ./nltk_data 目录"
    exit 1
else
    log_info "NLTK 数据完整 ✓"
fi

# ==========================================
# 2. 检查嵌入模型 (BGE-M3)
# ==========================================
MODEL_PATH="${LOCAL_EMBEDDING_PATH:-./ml_models/bge-m3}"

if [ -d "$MODEL_PATH" ] && [ -f "$MODEL_PATH/config.json" ]; then
    log_info "BGE-M3 嵌入模型已就绪: $MODEL_PATH ✓"
else
    log_warn "BGE-M3 模型未找到: $MODEL_PATH"
    log_info "正在从 HuggingFace 下载 BGE-M3..."

    python -c "
import os
from sentence_transformers import SentenceTransformer

model_path = os.environ.get('LOCAL_EMBEDDING_PATH', './ml_models/bge-m3')
print(f'下载 BAAI/bge-m3 到 {model_path}...')
os.makedirs(os.path.dirname(model_path), exist_ok=True)
model = SentenceTransformer('BAAI/bge-m3')
model.save(model_path)
print('BGE-M3 下载完成。')
"
    log_info "BGE-M3 下载完成 ✓"
fi

# ==========================================
# 3. 检查重排序模型 (BGE-Reranker)
# ==========================================
RERANKER_PATH="${RERANKER_MODEL_NAME:-./ml_models/bge-reranker-v2-m3}"

if [ -d "$RERANKER_PATH" ] && [ -f "$RERANKER_PATH/config.json" ]; then
    log_info "重排序模型已就绪: $RERANKER_PATH ✓"
else
    log_warn "重排序模型未找到: $RERANKER_PATH"
    log_info "正在从 HuggingFace 下载 BGE-Reranker-v2-M3 模型..."

    python -c "
import os
from sentence_transformers import CrossEncoder

model_name = 'BAAI/bge-reranker-v2-m3'
model_path = os.environ.get('RERANKER_MODEL_NAME', './ml_models/bge-reranker-v2-m3')

print(f'下载模型 {model_name} 到 {model_path}...')
os.makedirs(os.path.dirname(model_path), exist_ok=True)
model = CrossEncoder(model_name)
model.save(model_path)
print('BGE-Reranker 模型下载完成。')
"
    log_info "重排序模型下载完成 ✓"
fi

# ==========================================
# 4. 启动应用
# ==========================================
log_info "所有依赖检查完毕，启动应用..."

exec "$@"
