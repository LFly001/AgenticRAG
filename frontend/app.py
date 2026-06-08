import streamlit as st
import requests
import time
import os
import uuid

# 配置 — Docker 环境中通过环境变量指定后端地址
API_BASE_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# —— 轮询配置 ——
MAX_POLL_RETRIES = 120       # 最多轮询 120 次 (120 × 5s = 600s = 10分钟)
POLL_INTERVAL = 5             # 每次轮询间隔 (秒)，6分钟处理时间无需高频轮询

st.set_page_config(page_title="5-Agent 多轮对话知识库", page_icon="🤖", layout="wide")

st.title("🤖 多 Agent 协作智能知识库")
st.markdown("💬 对话记忆 · 📋 查询拆解 · 🔍 自主检索 · 📝 智能生成 · 🛡️ 幻觉检测")
st.markdown("---")

# —— 对话记忆 ——
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# 侧边栏：文件上传
with st.sidebar:
    st.header("📂 知识入库")
    uploaded_file = st.file_uploader("上传 PDF/Word 文档", type=["pdf", "docx"])

    # ——— 轮询后台任务状态 ———
    if "pending_task_id" in st.session_state and st.session_state.pending_task_id:
        task_id = st.session_state.pending_task_id

        # 初始化轮询计数器
        if "poll_count" not in st.session_state:
            st.session_state.poll_count = 0

        # 超过最大重试次数 → 终止轮询
        if st.session_state.poll_count >= MAX_POLL_RETRIES:
            st.error(f"⏰ 任务超时：已等待 {MAX_POLL_RETRIES * POLL_INTERVAL} 秒，请检查后端服务状态。")
            st.session_state.pending_task_id = None
            st.session_state.poll_count = 0

        else:
            try:
                resp = requests.get(f"{API_BASE_URL}/task-status/{task_id}", timeout=5)
                if resp.status_code == 200:
                    status = resp.json()["status"]

                    if status == "completed":
                        st.success("✅ 文档入库完成！")
                        st.session_state.pending_task_id = None
                        st.session_state.poll_count = 0

                    elif status.startswith("failed"):
                        st.error(f"❌ 入库失败: {status}")
                        st.session_state.pending_task_id = None
                        st.session_state.poll_count = 0

                    elif status == "not_found":
                        st.error(f"❌ 任务状态丢失：后端找不到此任务 ({task_id[:8]}...)，请重新上传文件。")
                        st.session_state.pending_task_id = None
                        st.session_state.poll_count = 0

                    else:
                        st.session_state.poll_count += 1
                        cnt = st.session_state.poll_count
                        st.info(f"⏳ 正在后台处理文档...（第 {cnt}/{MAX_POLL_RETRIES} 次检查）")
                        time.sleep(POLL_INTERVAL)
                        st.rerun()
                else:
                    st.session_state.poll_count += 1
                    cnt = st.session_state.poll_count
                    st.warning(
                        f"⚠️ 服务器返回异常 ({resp.status_code})，"
                        f"正在重试...（{cnt}/{MAX_POLL_RETRIES}）"
                    )
                    time.sleep(POLL_INTERVAL)
                    st.rerun()

            except requests.exceptions.ConnectionError:
                st.session_state.poll_count += 1
                cnt = st.session_state.poll_count
                st.warning(
                    f"⚠️ 无法连接后端服务，"
                    f"正在重试...（{cnt}/{MAX_POLL_RETRIES}）"
                )
                time.sleep(POLL_INTERVAL)
                st.rerun()
            except Exception as e:
                st.session_state.poll_count += 1
                cnt = st.session_state.poll_count
                st.warning(
                    f"⚠️ 请求异常 ({type(e).__name__})，"
                    f"正在重试...（{cnt}/{MAX_POLL_RETRIES}）"
                )
                time.sleep(POLL_INTERVAL)
                st.rerun()

    if uploaded_file is not None:
        if st.button("开始解析并入库"):
            with st.spinner("正在上传文件..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                try:
                    resp = requests.post(f"{API_BASE_URL}/upload", files=files, timeout=120)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state.pending_task_id = data["task_id"]
                        st.session_state.poll_count = 0
                        st.success("📤 文件已上传，后台处理中...")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"上传失败: {resp.text}")
                except Exception as e:
                    st.error(f"连接服务器失败: {e}")

    st.markdown("---")
    st.header("⚙️ 系统能力")
    st.markdown("""
    本系统基于 **LangGraph 5-Agent 协作架构**：

    💬 **ConversationAgent** — 对话记忆 + 追问识别 + 上下文融合

    📋 **QueryPlanner** — 分析问题复杂度，复合问题自动拆解为子查询

    🔎 **RetrieverAgent** — 自主选择检索策略（向量/BM25/混合），自评质量并自动改写

    ⚡ **并行检索** — 多个子查询并行执行，结果合并去重

    🔍 **CriticAgent** — 三维度评审（事实准确性 + 引用准确性 + 完整性），拦截幻觉

    🔄 **自修正** — 评审不通过自动重新生成，最多修正 2 次
    """)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.caption(f"🆔 会话: `{st.session_state.session_id[:8]}...`")
    with col2:
        if st.button("🗑️ 新对话"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()

# 主界面：聊天
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # 如果是助手消息，显示思考过程和来源
        if message["role"] == "assistant":
            # 显示思考过程
            if "thought_process" in message and message["thought_process"]:
                with st.expander("🧠 查看 AI 思考过程"):
                    for step in message["thought_process"]:
                        st.text(f"• {step}")

            # 显示来源
            if "sources" in message and message["sources"]:
                with st.expander("📚 查看引用来源"):
                    for src in message["sources"]:
                        st.markdown(
                            f"**ID**: `{src.get('id', 'N/A')}` | "
                            f"**文件**: {src.get('source_file', 'Unknown')} | "
                            f"**页码**: {src.get('page', 'N/A')}"
                        )
                        if src.get("type"):
                            st.caption(f"类型: {src['type']}")
                        if src.get("snippet"):
                            st.caption(f"摘要: {src['snippet']}")

# 输入框
if prompt := st.chat_input("请输入您的问题..."):
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 调用 Agentic RAG
    with st.chat_message("assistant"):
        with st.spinner("🤖 多 Agent 协作中 — 规划 → 检索 → 生成 → 评审..."):
            try:
                resp = requests.post(
                    f"{API_BASE_URL}/query",
                    json={
                        "question": prompt,
                        "session_id": st.session_state.session_id,
                    },
                    timeout=120,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    answer = data["answer"]
                    thought_process = data.get("thought_process", [])

                    # 始终显示思考过程
                    if thought_process:
                        with st.expander("🧠 AI 思考过程"):
                            for step in thought_process:
                                st.text(f"• {step}")

                    st.markdown(answer)

                    # 保存消息，包含来源和思考过程
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": data.get("sources", []),
                        "thought_process": thought_process,
                    })
                else:
                    st.error(f"请求失败: {resp.text}")
            except Exception as e:
                st.error(f"连接错误: {e}")

# 底部说明
st.markdown("---")
st.caption("Powered by LangGraph Multi-Agent · DeepSeek · ChromaDB · BGE-M3 · BGE-Reranker-v2-M3")
