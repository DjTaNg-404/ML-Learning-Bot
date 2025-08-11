from fastapi import FastAPI
from llm.graph_app import app_graph
from llm.rag_chain import rag_chain  # 导入RAG链
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import json
import asyncio
import time
from typing import AsyncGenerator

app = FastAPI()

class ChatRequest(BaseModel):
    session_id: str
    question: str
    use_rag: bool = True
    use_rerank: bool = False
    stream: bool = False
    retrieval_k: int = 5
    rerank_top_k: int = 50

class RAGSettingsRequest(BaseModel):
    retrieval_k: int
    rerank_top_k: int
    use_rag: bool = True
    use_rerank: bool = False
    stream: bool = False  # 新增流式响应开关

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """聊天接口 - 支持普通和流式响应"""
    if request.stream:
        # 返回流式响应
        return StreamingResponse(
            chat_stream_generator(request),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # 禁用nginx缓冲
            }
        )
    else:
        # 返回普通响应
        return await chat_normal_response(request)

async def chat_stream_generator(request: ChatRequest) -> AsyncGenerator[str, None]:
    """流式响应生成器"""
    try:
        # 1. 发送开始信号
        yield f"data: {json.dumps({'type': 'start', 'message': '开始处理请求...', 'timestamp': time.time()})}\n\n"
        await asyncio.sleep(0.1)
        
        # 2. RAG检索阶段
        rag_context = ""
        if request.use_rag and rag_chain.vector_store and rag_chain.vector_store.is_initialized():
            yield f"data: {json.dumps({'type': 'rag_start', 'message': '🔍 正在检索相关知识...', 'timestamp': time.time()})}\n\n"
            await asyncio.sleep(0.1)
            
            try:
                # 临时更新RAG链的检索设置（仅用于本次查询）
                original_k = rag_chain.retrieval_k
                original_rerank_k = rag_chain.rerank_top_k
                
                # 设置本次查询的参数
                rag_chain.retrieval_k = request.retrieval_k
                rag_chain.rerank_top_k = request.rerank_top_k
                
                # 执行RAG检索
                rag_context = await asyncio.to_thread(
                    rag_chain.retrieve_context,
                    request.question,
                    request.use_rerank
                )
                
                # 恢复原始设置
                rag_chain.retrieval_k = original_k
                rag_chain.rerank_top_k = original_rerank_k
                
                if rag_context:
                    yield f"data: {json.dumps({'type': 'rag_success', 'message': '✅ 找到相关知识', 'context_length': len(rag_context), 'retrieval_count': request.retrieval_k, 'timestamp': time.time()})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'rag_empty', 'message': '⚠️ 未找到相关知识，使用AI基础知识回答', 'timestamp': time.time()})}\n\n"
                    
            except Exception as e:
                yield f"data: {json.dumps({'type': 'rag_error', 'message': f'❌ 检索失败: {str(e)}', 'timestamp': time.time()})}\n\n"
        
        elif request.use_rag:
            yield f"data: {json.dumps({'type': 'rag_unavailable', 'message': '⚠️ RAG系统未初始化，使用AI基础知识回答', 'timestamp': time.time()})}\n\n"
        
        # 3. LLM推理阶段
        yield f"data: {json.dumps({'type': 'llm_start', 'message': '🤖 AI正在思考...', 'timestamp': time.time()})}\n\n"
        await asyncio.sleep(0.1)
        
        try:
            # 执行LLM推理
            input_messages = [HumanMessage(content=request.question)]
            
            response = await asyncio.to_thread(
                rag_chain.invoke_with_rag,
                messages=input_messages,
                use_rag=request.use_rag,
                use_rerank=request.use_rerank
            )
            
            # 4. 发送最终结果
            yield f"data: {json.dumps({'type': 'response', 'content': response, 'rag_used': bool(rag_context), 'timestamp': time.time()})}\n\n"
            
        except Exception as e:
            print(f"LLM推理错误: {e}")
            # 降级到普通聊天
            try:
                config = {"configurable": {"thread_id": request.session_id}}
                input_messages = [HumanMessage(content=request.question)]
                output = await asyncio.to_thread(
                    app_graph.invoke,
                    {"messages": input_messages},
                    config
                )
                
                yield f"data: {json.dumps({'type': 'response', 'content': output['messages'][-1].content, 'rag_used': False, 'fallback': True, 'timestamp': time.time()})}\n\n"
                
            except Exception as fallback_error:
                yield f"data: {json.dumps({'type': 'error', 'message': f'❌ 处理失败: {str(fallback_error)}', 'timestamp': time.time()})}\n\n"
        
        # 5. 发送完成信号
        yield f"data: {json.dumps({'type': 'done', 'message': '✅ 处理完成', 'timestamp': time.time()})}\n\n"
        
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': f'❌ 系统错误: {str(e)}', 'timestamp': time.time()})}\n\n"

async def chat_normal_response(request: ChatRequest) -> dict:
    """普通响应处理"""
    config = {"configurable": {"thread_id": request.session_id}}
    input_messages = [HumanMessage(content=request.question)]
    
    try:
            # 临时更新RAG链的检索设置（仅用于本次查询）
            original_k = rag_chain.retrieval_k
            original_rerank_k = rag_chain.rerank_top_k
            
            # 设置本次查询的参数
            rag_chain.retrieval_k = request.retrieval_k
            rag_chain.rerank_top_k = request.rerank_top_k
            
            # 直接使用RAG链处理
            response = await asyncio.to_thread(
                rag_chain.invoke_with_rag,
                messages=input_messages,
                use_rag=request.use_rag,
                use_rerank=request.use_rerank
            )
            
            # 恢复原始设置
            rag_chain.retrieval_k = original_k
            rag_chain.rerank_top_k = original_rerank_k
            
            return {
                "response": response,
                "rag_used": request.use_rag and rag_chain.vector_store is not None,
                "retrieval_settings": {
                    "retrieval_k": request.retrieval_k,
                    "rerank_top_k": request.rerank_top_k
                }
            }
        
    except Exception as e:
        print(f"聊天处理错误: {e}")
        # 降级到普通聊天
        try:
            output = await asyncio.to_thread(
                app_graph.invoke,
                {"messages": input_messages},
                config
            )
            return {
                "response": output["messages"][-1].content,
                "rag_used": False,
                "fallback": True
            }
        except Exception as fallback_error:
            return {
                "response": f"处理失败: {str(fallback_error)}",
                "rag_used": False,
                "error": True
            }

@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """专门的流式聊天接口"""
    request.stream = True  # 强制启用流式响应
    return await chat_endpoint(request)

@app.post("/rag/update_settings")
async def update_rag_settings(request: RAGSettingsRequest):
    """更新RAG检索设置"""
    try:
        rag_chain.update_retrieval_settings(
            retrieval_k=request.retrieval_k,
            rerank_top_k=request.rerank_top_k
        )
        return {
            "status": "success",
            "message": "RAG设置更新成功",
            "settings": {
                "retrieval_k": request.retrieval_k,
                "rerank_top_k": request.rerank_top_k
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"更新失败: {str(e)}"
        }

@app.get("/rag/status")
async def get_rag_status():
    """获取RAG系统状态"""
    try:
        return await asyncio.to_thread(rag_chain.get_status)
    except Exception as e:
        return {
            "status": "error",
            "message": f"获取状态失败: {str(e)}"
        }

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "rag_available": rag_chain.vector_store is not None
    }