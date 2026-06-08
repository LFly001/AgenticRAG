# 🤖 多 Agent 协作智能知识库

基于 **LangGraph 5-Agent 协作架构** 的企业级 RAG 系统。

## 架构

| Agent | 职责 |
|---|---|
| 💬 ConversationAgent | 对话记忆 + 追问识别 + 上下文融合 |
| 📋 QueryPlanner | 复合问题自动拆解为子查询 |
| 🔎 RetrieverAgent | 自主选择检索策略（向量/BM25/混合），自评质量并自动改写 |
| ⚡ 并行检索 | 多子查询并行执行，结果合并去重 |
| 🔍 CriticAgent | 三维度评审（事实/引用/完整性），拦截幻觉 |

## 快速开始

### 1. 克隆项目

```bash
git clone <repo-url>
cd MultiAgentRAG
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
docker compose up -d          # 后台启动
docker compose logs -f        # 查看日志
docker compose logs -f backend # 仅后端日志
docker compose down           # 停止并删除容器
docker compose down -v        # 同时删除数据卷
```

## 技术栈

Python · FastAPI · Streamlit · LangGraph · DeepSeek · ChromaDB · BGE-M3 · BGE-Reranker-v2-M3 · Redis · Docker
