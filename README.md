# 🤖 8-Agent 协作智能知识库

基于 **LangGraph 8-Agent 协作架构** 的企业级 RAG 系统。

## 架构

```
orchestrator → intent → retriever → doc_filter → context_compress → reason → writer → anti_hallucination
                  │                                                       │
                  ├── 闲聊短路 / 澄清终止                                   └── 证据不足 → 二次检索回路
                  │
                  └── END
```

| Agent | 职责 |
|---|---|
| 🎯 OrchestratorAgent | 总调度唯一入口，全局分支分发，trace_id 追踪 |
| 💬 IntentAgent | 闲聊检测 + 指代消解 + 问题拆解 + 澄清判断 |
| 🔍 RetrieveAgent | 并行混合检索（向量 + BM25 + RRF 融合 + CrossEncoder 重排序） |
| 🫧 DocFilterAgent | 四重过滤（碎片/低分/过期/去重）+ 文档冲突检测 |
| 🗜️ ContextCompressAgent | 结构化 XML 上下文 + token 预算控制 + LLM 智能压缩 |
| 🧠 ReasonAgent | CoT 链式推理 + 证据充足度判断 + 触发二次检索 |
| ✍️ WriterAgent | 基于推理框架生成答案 + 自动引用标注 + 敏感信息脱敏 |
| 🛡️ AntiHallucinationAgent | 逐句核查 + 修正错误 + 幻觉风险评级 |

### 链路

| 链路 | 路径 | 触发条件 |
|---|---|---|
| 闲聊短路 | O → I → END | 问候/感谢/告别等非业务对话 |
| 澄清终止 | O → I → END | 问题信息缺失/歧义，无法回答 |
| 正常主链 | O → I → R → D → C → Re → W → A → END | 证据充足，一次通过 |
| 二次检索 | O → I → R → D → C → Re → R → D → C → Re → W → A → END | 证据不足，补充检索 |
| 高风险输出 | 同主链/二次检索，A 标记 `hallucination_risk="high"` | 检测到编造/虚假数据 |

## 快速开始

### 1. 克隆项目

```bash
git clone <repo-url>
cd AgenticRAG
```

### 2. 配置 Docker 镜像加速（国内用户）

Docker Hub 在国内访问极慢，**首次启动前**先配置镜像加速：

```bash
bash setup_docker_mirror.sh
```

或手动创建 `daemon.json`：

**Linux** (`/etc/docker/daemon.json`) / **Windows** (`%USERPROFILE%\.docker\daemon.json`)：

```json
{
  "registry-mirrors": [
    "https://docker.m.daocloud.io",
    "https://docker.1ms.run"
  ]
}
```

改完后重启 Docker Desktop 或 `sudo systemctl restart docker`。

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 DEEPSEEK_API_KEY
```

### 4. 下载 NLTK 数据（必须）

容器启动前，先在宿主机执行以下脚本，NLTK 数据会通过 volume 挂载进容器：

```bash
pip install nltk
python scripts/download_nltk_data.py
```

> ⚠️ 如果缺少 NLTK 数据，容器启动会直接报错退出。

### 5. 启动服务

首次启动时，`entrypoint.sh` 会自动检测并下载缺失的模型（BGE-M3、BGE-Reranker），无需手动操作：

```bash
docker compose up -d
```

> 💡 如果想提前下载模型避免首次启动等待，可参考 `docker-compose.yml` 中的方式 A。

### 6. 访问

| 服务 | 地址 |
|---|---|
| Web 界面 | http://localhost:8501 |
| API 文档 | http://localhost:8000/docs |

## 常用命令

```bash
docker compose up -d              # 后台启动
docker compose logs -f            # 查看日志
docker compose logs -f backend    # 仅后端日志
docker compose down               # 停止并删除容器
docker compose down -v            # 同时删除数据卷
```

## 技术栈

Python · FastAPI · Streamlit · LangGraph · DeepSeek · ChromaDB · BGE-M3 · BGE-Reranker-v2-M3 · Redis · Docker
