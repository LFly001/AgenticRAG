import streamlit as st
import requests
import time
import os
import uuid
from urllib.parse import quote

API_BASE_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
MAX_POLL_RETRIES = 120
POLL_INTERVAL = 5

st.set_page_config(page_title="Agentic RAG 知识库问答系统", page_icon="🤖", layout="wide")

st.title("🤖 Agentic RAG 知识库问答系统")
st.markdown("🎯 意图解析 · 🔍 混合检索 · 🫧 文档清洗 · 🗜️ 上下文压缩 · 🧠 逻辑推理 · ✍️ 答案生成 · 🛡️ 幻觉检测")
st.markdown("---")

# —— 会话初始化 ——
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_task_id" not in st.session_state:
    st.session_state.pending_task_id = None
if "poll_count" not in st.session_state:
    st.session_state.poll_count = 0
if "uploading" not in st.session_state:
    st.session_state.uploading = False  # 上传锁，防止轮询 rerun 触发重复提交

# ============================================================================
# 侧边栏
# ============================================================================
with st.sidebar:
    st.header("📂 知识入库")

    # ——— 上传区（始终显示） ———
    uploaded_file = st.file_uploader(
        "上传 PDF/Word 文档",
        type=["pdf", "docx"],
        key="file_uploader",
        label_visibility="visible",
    )

    # 上传按钮（有任务在跑时禁用，防重复提交）
    upload_disabled = (
        uploaded_file is None
        or st.session_state.pending_task_id is not None
        or st.session_state.uploading
    )
    upload_clicked = st.button(
        "🚀 开始解析并入库",
        disabled=upload_disabled,
    )

    # ——— 执行上传（uploading 锁防 rerun 重复触发）———
    if upload_clicked and uploaded_file is not None and not st.session_state.uploading:
        st.session_state.uploading = True
        with st.spinner("正在上传文件..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            try:
                resp = requests.post(f"{API_BASE_URL}/upload", files=files, timeout=120)
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state.pending_task_id = data["task_id"]
                    st.session_state.poll_count = 0
                    # 清掉已选文件
                    if "file_uploader" in st.session_state:
                        del st.session_state["file_uploader"]
                    st.success("📤 文件已上传，后台处理中...")
                    time.sleep(0.5)
                    st.session_state.uploading = False
                    st.rerun()
                else:
                    st.error(f"上传失败: {resp.text}")
                    st.session_state.uploading = False
            except Exception as e:
                st.error(f"连接服务器失败: {e}")
                st.session_state.uploading = False

    # ——— 轮询（仅在活跃任务时） ———
    task_id = st.session_state.pending_task_id
    if task_id:
        st.divider()
        st.caption(f"任务 `{task_id[:8]}...`")

        # 进度条
        progress = min(st.session_state.poll_count / MAX_POLL_RETRIES, 1.0)
        st.progress(progress, text=f"后台处理中...（{st.session_state.poll_count}/{MAX_POLL_RETRIES}）")

        # 取消按钮
        if st.button("❌ 取消等待"):
            st.session_state.pending_task_id = None
            st.session_state.poll_count = 0
            st.rerun()

        # 超时
        if st.session_state.poll_count >= MAX_POLL_RETRIES:
            st.error(f"⏰ 超时（{MAX_POLL_RETRIES * POLL_INTERVAL}s）")
            st.session_state.pending_task_id = None
            st.session_state.poll_count = 0
            time.sleep(1)
            st.rerun()

        # 轮询
        try:
            resp = requests.get(f"{API_BASE_URL}/task-status/{task_id}", timeout=5)
            if resp.status_code == 200:
                status = resp.json()["status"]

                if status == "completed":
                    st.success("✅ 文档入库完成！")
                    st.session_state.pending_task_id = None
                    st.session_state.poll_count = 0
                    time.sleep(1)
                    st.rerun()

                elif status.startswith("failed"):
                    st.error(f"❌ 入库失败: {status}")
                    st.session_state.pending_task_id = None
                    st.session_state.poll_count = 0

                elif status == "not_found":
                    st.error("❌ 任务丢失，请重新上传")
                    st.session_state.pending_task_id = None
                    st.session_state.poll_count = 0

                else:
                    # 处理中，递增计数器并自动 rerun
                    st.session_state.poll_count += 1
                    time.sleep(POLL_INTERVAL)
                    st.rerun()

        except requests.exceptions.ConnectionError:
            st.session_state.poll_count += 1
            st.warning(f"⚠️ 无法连接后端（{st.session_state.poll_count}/{MAX_POLL_RETRIES}）")
            time.sleep(POLL_INTERVAL)
            st.rerun()
        except Exception as e:
            st.session_state.poll_count += 1
            st.warning(f"⚠️ {type(e).__name__}")
            time.sleep(POLL_INTERVAL)
            st.rerun()

    # ========================================================================
    # 📋 知识库文档管理
    # ========================================================================
    st.divider()
    st.header("📋 知识库管理")

    # 初始化状态
    if "doc_list" not in st.session_state:
        st.session_state.doc_list = None
    if "delete_target" not in st.session_state:
        st.session_state.delete_target = None

    # ---- 刷新 ----
    if st.button("🔄 刷新文档列表", use_container_width=True):
        try:
            resp = requests.get(f"{API_BASE_URL}/documents", timeout=10)
            if resp.status_code == 200:
                st.session_state.doc_list = resp.json()
            else:
                st.error(f"获取文档列表失败: {resp.text}")
        except requests.exceptions.ConnectionError:
            st.warning("⚠️ 无法连接后端")
        except Exception as e:
            st.error(f"错误: {e}")

    # ---- 显示文档列表 ----
    doc_list = st.session_state.doc_list
    if doc_list and doc_list.get("documents"):
        docs = doc_list["documents"]
        st.caption(f"{len(docs)} 份文档 · {doc_list['total_chunks']} 个切片")

        for i, doc in enumerate(docs):
            filename = doc["filename"]
            count = doc["chunk_count"]

            col1, col2 = st.columns([4, 1])
            with col1:
                st.text(f"📄 {filename}\n   {count} 切片")
            with col2:
                # 按钮只设 flag，不执行任何 IO
                if st.button("🗑️", key=f"delbtn_{i}", help=f"删除 {filename}"):
                    st.session_state.delete_target = filename
                    st.rerun()
    elif doc_list is not None and not doc_list.get("documents"):
        st.info("知识库为空，暂无文档。")

    # ---- 执行删除（在循环外部，避免 widget 状态冲突） ----
    target = st.session_state.delete_target
    if target:
        try:
            encoded_name = quote(target, safe="")
            del_resp = requests.delete(
                f"{API_BASE_URL}/documents/{encoded_name}",
                timeout=30,
            )
            if del_resp.status_code == 200:
                del_data = del_resp.json()
                if del_data["success"]:
                    st.success(del_data["message"])
                else:
                    st.error(del_data["message"])
            else:
                st.error(f"删除失败 ({del_resp.status_code}): {del_resp.text}")
        except Exception as e:
            st.error(f"连接错误: {e}")
        finally:
            # 清除目标并自动刷新列表
            st.session_state.delete_target = None
            try:
                refresh_resp = requests.get(f"{API_BASE_URL}/documents", timeout=10)
                if refresh_resp.status_code == 200:
                    st.session_state.doc_list = refresh_resp.json()
            except Exception:
                st.session_state.doc_list = None
            time.sleep(0.3)
            st.rerun()

    st.divider()
    st.caption(f"🆔 会话: `{st.session_state.session_id[:8]}...`")
    if st.button("🗑️ 新对话"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        # 也清除文档列表缓存
        st.session_state.doc_list = None
        st.rerun()

# ============================================================================
# 主界面：聊天（输入框置顶，避免消息多了之后滚动卡死）
# ============================================================================

# —— 输入框置顶，始终在视口内 ——
if prompt := st.chat_input("请输入您的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("🤖 思考中..."):
        try:
            resp = requests.post(
                f"{API_BASE_URL}/query",
                json={"question": prompt, "session_id": st.session_state.session_id},
                timeout=180,
            )
            if resp.status_code == 200:
                data = resp.json()
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": data["answer"],
                    "sources": data.get("sources", []),
                    "thought_process": data.get("thought_process", []),
                })
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ 请求失败: {resp.text}",
                    "sources": [],
                    "thought_process": [],
                })
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ 连接错误: {e}",
                "sources": [],
                "thought_process": [],
            })
    st.rerun()

# —— 历史消息（输入框下方） ——
st.divider()
MAX_VISIBLE = 10
messages = st.session_state.messages
total = len(messages)

if total > MAX_VISIBLE:
    with st.expander(f"📜 更早的消息（{total - MAX_VISIBLE} 条）", expanded=False):
        for message in messages[:total - MAX_VISIBLE]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

visible = messages[max(0, total - MAX_VISIBLE):]
for message in visible:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            if message.get("thought_process"):
                with st.expander("🧠 思考过程"):
                    for step in message["thought_process"]:
                        st.text(f"• {step}")
            if message.get("sources"):
                with st.expander("📚 引用来源"):
                    for src in message["sources"]:
                        st.markdown(
                            f"**{src.get('source_file', '?')}** · "
                            f"p{src.get('page', '?')} · "
                            f"`{src.get('id', '?')}`"
                        )

st.caption("LangGraph 8-Agent · DeepSeek · ChromaDB · BGE-M3 · BGE-Reranker-v2-M3")
