import streamlit as st
import requests
import json
import uuid
import time

# 页面配置
st.set_page_config(
    page_title="AI 聊天演示",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI 聊天演示（支持RAG检索）")
st.markdown("---")

# 获取RAG状态
@st.cache_data(ttl=30)  # 10秒缓存
def get_rag_status():
    """获取后端RAG状态"""
    try:
        response = requests.get("http://localhost:8000/rag/status", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "error", "message": "无法获取状态"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def update_rag_settings(retrieval_k, rerank_top_k):
    """更新后端RAG设置"""
    try:
        payload = {
            "retrieval_k": retrieval_k,
            "rerank_top_k": rerank_top_k
        }
        response = requests.post(
            "http://localhost:8000/rag/update_settings",
            json=payload,
            timeout=5
        )
        if response.status_code == 200:
            return True, "设置更新成功"
        else:
            return False, f"更新失败: {response.text}"
    except Exception as e:
        return False, f"更新失败: {str(e)}"

def parse_stream_response(response):
    """解析流式响应"""
    for line in response.iter_lines():
        if line:
            line_str = line.decode('utf-8')
            if line_str.startswith('data: '):
                try:
                    data = json.loads(line_str[6:])  # 去掉 'data: ' 前缀
                    yield data
                except json.JSONDecodeError:
                    continue

def handle_stream_chat(payload):
    """处理流式聊天响应"""
    try:
        response = requests.post(
            "http://localhost:8000/chat",
            json=payload,
            stream=True,
            timeout=120  # 增加超时时间
        )
        
        if response.status_code == 200:
            return parse_stream_response(response)
        else:
            return [{"type": "error", "message": f"服务器错误: {response.status_code}"}]
            
    except requests.exceptions.ConnectionError:
        return [{"type": "error", "message": "无法连接到后端服务器，请确保服务器正在运行"}]
    except requests.exceptions.Timeout:
        return [{"type": "error", "message": "请求超时，服务器可能正在处理中"}]
    except Exception as e:
        return [{"type": "error", "message": f"发生错误: {str(e)}"}]

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "您好！我是支持知识检索的 AI 助手，有什么可以帮助您的吗？ 👋"}
    ]

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "use_rag" not in st.session_state:
    st.session_state.use_rag = True

if "use_rerank" not in st.session_state:
    st.session_state.use_rerank = False

if "use_stream" not in st.session_state:
    st.session_state.use_stream = True

if "retrieval_k" not in st.session_state:
    st.session_state.retrieval_k = 5

if "rerank_top_k" not in st.session_state:
    st.session_state.rerank_top_k = 50

# 侧边栏配置
with st.sidebar:
    st.header("会话信息")
    st.write(f"**会话ID:** `{st.session_state.session_id[:8]}...`")
    st.write(f"**消息数量:** {len(st.session_state.messages)}")
    
    st.markdown("---")
    st.header("响应设置")
    
    # 流式响应开关
    use_stream = st.toggle("启用流式响应", value=st.session_state.use_stream)
    st.session_state.use_stream = use_stream
    if use_stream:
        st.info("🔄 将实时显示处理进度")
    
    st.markdown("---")
    st.header("RAG 设置")
    
    # 获取并显示RAG状态
    rag_status = get_rag_status()
    
    if rag_status["status"] == "active":
        st.success("✅ RAG系统运行中")
        st.write(f"📚 文档数: {rag_status.get('document_count', 0)}")
        st.write(f"🔧 嵌入方式: {rag_status.get('embedding_type', 'N/A')}")
        st.write(f"📖 模型名称: {rag_status.get('model_name', 'N/A')}")
        
        # 显示当前检索设置
        current_k = rag_status.get('retrieval_k', 5)
        current_rerank_k = rag_status.get('rerank_top_k', 50)
        st.write(f"🔍 当前检索条数: {current_k}")
        st.write(f"🔄 重排序候选数: {current_rerank_k}")
        
        st.markdown("**检索参数设置:**")
        
        # 检索条数滑块
        retrieval_k = st.slider(
            "检索文档条数 (k)",
            min_value=1,
            max_value=20,
            value=st.session_state.retrieval_k,
            step=1,
            help="最终返回的文档片段数量，越多质量越高但速度越慢"
        )
        st.session_state.retrieval_k = retrieval_k
        
        # 重排序候选数滑块
        rerank_top_k = st.slider(
            "重排序候选数",
            min_value=retrieval_k,  # 至少等于检索条数
            max_value=100,
            value=max(st.session_state.rerank_top_k, retrieval_k),
            step=5,
            help="重排序时的候选文档数，通常是检索条数的2-10倍"
        )
        st.session_state.rerank_top_k = rerank_top_k
        
        # 参数说明
        if retrieval_k != current_k or rerank_top_k != current_rerank_k:
            st.info("💡 参数已调整，下次查询时生效")
        
        # RAG控制选项
        use_rag = st.toggle("启用知识检索 (RAG)", value=st.session_state.use_rag)
        st.session_state.use_rag = use_rag
        
        if use_rag:
            use_rerank = st.checkbox("启用重排序 (Rerank)", value=st.session_state.use_rerank)
            st.session_state.use_rerank = use_rerank
            if use_rerank:
                st.info(f"🔄 将从{rerank_top_k}个候选中重排序出{retrieval_k}个最佳结果")
        
    elif rag_status["status"] == "inactive":
        st.warning("⚠️ RAG系统未激活")
        st.write("可能原因：")
        st.write("- 未找到文档")
        st.write("- 向量库初始化失败")
        st.session_state.use_rag = False
        
    else:  # error
        st.error("❌ 无法连接到后端RAG系统")
        st.write(f"错误信息: {rag_status.get('message', '未知错误')}")
        st.session_state.use_rag = False
    
    st.markdown("---")
    
    # 更新RAG设置按钮
    if st.button("🔧 更新RAG设置", use_container_width=True):
        if rag_status["status"] == "active":
            success, message = update_rag_settings(
                st.session_state.retrieval_k,
                st.session_state.rerank_top_k
            )
            if success:
                st.success(message)
                st.cache_data.clear()  # 清除缓存以刷新状态
                time.sleep(0.5)
                st.rerun()
            else:
                st.error(message)
        else:
            st.warning("RAG系统未激活，无法更新设置")
    
    # 刷新RAG状态按钮
    if st.button("🔄 刷新RAG状态", use_container_width=True):
        st.cache_data.clear()  # 清除缓存以强制刷新
        st.rerun()
    
    # 清空对话按钮
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "会话已重置！有什么可以帮助您的吗？ 👋"}
        ]
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

# 显示聊天历史
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# 聊天输入
if prompt := st.chat_input("请输入您的问题..."):
    # 添加用户消息到聊天历史
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 显示助手回复
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        status_placeholder = st.empty()
        full_response = ""
        
        # 构建请求载荷
        payload = {
            "session_id": st.session_state.session_id,
            "question": prompt,
            "use_rag": st.session_state.use_rag,
            "use_rerank": st.session_state.use_rerank,
            "stream": st.session_state.use_stream,
            "retrieval_k": st.session_state.retrieval_k,
            "rerank_top_k": st.session_state.rerank_top_k
        }
        
        if st.session_state.use_stream:
            # 流式响应处理
            rag_used = False
            error_occurred = False
            
            try:
                for stream_data in handle_stream_chat(payload):
                    data_type = stream_data.get("type")
                    message = stream_data.get("message", "")
                    
                    if data_type == "start":
                        status_placeholder.info("🚀 " + message)
                    
                    elif data_type == "rag_start":
                        status_placeholder.info("🔍 " + message)
                    
                    elif data_type == "rag_success":
                        context_length = stream_data.get("context_length", 0)
                        retrieval_count = stream_data.get("retrieval_count", st.session_state.retrieval_k)
                        status_placeholder.success(f"✅ {message} (检索到{retrieval_count}个文档片段，总长度: {context_length})")
                        rag_used = True
                    
                    elif data_type == "rag_empty":
                        status_placeholder.warning("⚠️ " + message)
                    
                    elif data_type == "rag_error":
                        status_placeholder.error("❌ " + message)
                    
                    elif data_type == "rag_unavailable":
                        status_placeholder.warning("⚠️ " + message)
                    
                    elif data_type == "llm_start":
                        status_placeholder.info("🤖 " + message)
                    
                    elif data_type == "response":
                        # 清除状态提示
                        status_placeholder.empty()
                        
                        # 获取最终响应
                        full_response = stream_data.get("content", "")
                        rag_used = stream_data.get("rag_used", False)
                        is_fallback = stream_data.get("fallback", False)
                        
                        # 显示RAG使用状态
                        if rag_used:
                            rerank_status = f"（{st.session_state.rerank_top_k}→{st.session_state.retrieval_k}重排序）" if st.session_state.use_rerank else f"（检索{st.session_state.retrieval_k}条）"
                            st.info(f"📚 已使用知识检索{rerank_status}")
                        elif st.session_state.use_rag and not is_fallback:
                            st.warning("🔍 未找到相关知识，使用AI基础知识回答")
                        elif is_fallback:
                            st.warning("⚠️ RAG处理失败，已降级到基础AI回答")
                        
                        # 模拟打字效果
                        words = full_response.split()
                        displayed_text = ""
                        for i, word in enumerate(words):
                            displayed_text += word + " "
                            message_placeholder.markdown(displayed_text + "▌")
                            time.sleep(0.02)
                        
                        # 显示最终结果
                        message_placeholder.markdown(full_response)
                    
                    elif data_type == "done":
                        status_placeholder.success("✅ " + message)
                        time.sleep(0.5)
                        status_placeholder.empty()
                    
                    elif data_type == "error":
                        status_placeholder.empty()
                        full_response = "❌ " + message
                        message_placeholder.markdown(full_response)
                        error_occurred = True
                        break
                
                # 如果没有收到响应且没有错误，显示默认错误
                if not full_response and not error_occurred:
                    full_response = "❌ 未收到有效响应"
                    message_placeholder.markdown(full_response)
                    
            except Exception as e:
                status_placeholder.empty()
                full_response = f"❌ 流式处理错误: {str(e)}"
                message_placeholder.markdown(full_response)
                print(f"流式聊天错误详情: {e}")
        
        else:
            # 普通响应处理（保持原有逻辑）
            try:
                with st.spinner("🤖 AI 正在思考中..."):
                    response = requests.post(
                        "http://localhost:8000/chat",
                        json=payload,
                        timeout=90
                    )
                
                if response.status_code == 200:
                    result = response.json()
                    assistant_response = result["response"]
                    rag_used = result.get("rag_used", False)
                    is_fallback = result.get("fallback", False)
                    
                    # 显示RAG使用状态
                    if rag_used:
                        rerank_status = f"（{st.session_state.rerank_top_k}→{st.session_state.retrieval_k}重排序）" if st.session_state.use_rerank else f"（检索{st.session_state.retrieval_k}条）"
                        st.info(f"📚 已使用知识检索{rerank_status}")
                    elif st.session_state.use_rag and not is_fallback:
                        st.warning("🔍 未找到相关知识，使用AI基础知识回答")
                    elif is_fallback:
                        st.warning("⚠️ RAG处理失败，已降级到基础AI回答")
                    
                    # 模拟流式输出效果
                    for chunk in assistant_response.split():
                        full_response += chunk + " "
                        time.sleep(0.02)
                        message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response.strip())
                    
                else:
                    full_response = f"❌ 服务器错误 ({response.status_code}): {response.text}"
                    message_placeholder.markdown(full_response)
                    
            except requests.exceptions.ConnectionError:
                full_response = "❌ 无法连接到后端服务器，请确保服务器正在运行 (http://localhost:8000)"
                message_placeholder.markdown(full_response)
            except requests.exceptions.Timeout:
                full_response = "⏱️ 请求超时，请稍后重试"
                message_placeholder.markdown(full_response)
            except Exception as e:
                full_response = f"❌ 发生错误: {str(e)}"
                message_placeholder.markdown(full_response)
                print(f"聊天错误详情: {e}")
    
    # 添加助手回复到聊天历史
    st.session_state.messages.append({"role": "assistant", "content": full_response.strip()})

# 页面底部信息
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.caption("💡 提示：可在侧边栏调整检索参数")
with col2:
    st.caption("🔧 RAG系统在后端运行")
with col3:
    st.caption("🚀 流式响应提供实时进度")
with col4:
    st.caption(f"📊 当前设置: {st.session_state.retrieval_k}条文档")