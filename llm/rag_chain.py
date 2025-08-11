from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import Document
from typing import List, Optional
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from rag.vector_store import VectorStore
from rag.document_loader import load_documents_from_directory
from llm.chain_app import chat_model

class RAGChain:
    """RAG增强的聊天链"""
    
    def __init__(self, retrieval_k: int = 5, rerank_top_k: int = 50):
        self.retrieval_k = retrieval_k  # 最终返回的文档数
        self.rerank_top_k = rerank_top_k  # 重排序候选数
        self.vector_store = None
        self.initialize_rag()
        self.setup_chain()
    
    def initialize_rag(self):
        """初始化RAG系统"""
        try:
            self.vector_store = VectorStore(
                embedding_type="siliconflow",
                model_name="BAAI/bge-large-zh-v1.5",
                persist_path="data/vector_db"
            )
            
            try:
                self.vector_store.load()
                print("✅ RAG向量库加载成功")
            except:
                documents = load_documents_from_directory("data/ml_books")
                if documents:
                    self.vector_store.create_from_documents(documents)
                    self.vector_store.save()
                    print("✅ RAG向量库创建成功")
                else:
                    self.vector_store = None
                    print("⚠️ 未找到文档，RAG功能不可用")
        
        except Exception as e:
            print(f"❌ RAG初始化失败: {e}")
            self.vector_store = None
    
    def setup_chain(self):
        """设置聊天链"""
        # RAG增强的提示模板
        self.rag_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """你是一个智能管家，名字叫Sugar-AI，服务于机器学习可视化学习平台。
你的职责包括：解答用户关于机器学习、数据分析、模型训练、平台使用等相关问题。

{context_instruction}

请用简洁、准确的语言回复，必要时可结合平台功能进行说明。"""
            ),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        # 普通聊天提示模板
        self.normal_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "你是一个智能管家，名字叫Sugar-AI，服务于机器学习可视化学习平台。"
                "你的职责包括：解答用户关于机器学习、数据分析、模型训练、平台使用等相关问题。"
                "请用简洁、准确的语言回复，必要时可结合平台功能进行说明。"
            ),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        self.output_parser = StrOutputParser()
    
    def retrieve_context(self, query: str, use_rerank: bool = False) -> str:
        """检索相关上下文"""
        if not self.vector_store or not self.vector_store.is_initialized():
            return ""
        
        try:
            search_results = self.vector_store.similarity_search(
                query=query,
                k=self.retrieval_k,  # 使用类属性
                use_rerank=use_rerank,
                rerank_top_k=self.rerank_top_k if use_rerank else self.retrieval_k
            )
            
            if search_results:
                return "\n\n".join([doc.page_content for doc in search_results])
            
        except Exception as e:
            print(f"RAG检索失败: {e}")
        
        return ""
    
    def invoke_with_rag(self, messages: List, use_rag: bool = True, use_rerank: bool = False) -> str:
        """调用RAG增强的聊天链"""
        
        if use_rag and self.vector_store:
            # 获取最后一条用户消息作为查询
            last_human_msg = None
            for msg in reversed(messages):
                if hasattr(msg, 'type') and msg.type == 'human':
                    last_human_msg = msg.content
                    break
            
            if last_human_msg:
                # 检索相关上下文
                context = self.retrieve_context(last_human_msg, use_rerank)
                
                if context:
                    # 使用RAG提示
                    context_instruction = f"""你是一个专业的机器学习助手，擅长解答机器学习相关问题。

以下是从《机器学习》教材（南瓜书）中检索到的相关内容：

{context}

**回答指导原则：**

1. **内容质量评估**：
   - 首先判断检索内容是否与用户问题直接相关
   - 如果相关度高，优先使用这些内容作为答案基础
   - 如果相关度低或内容不完整，可以补充你的专业知识

2. **信息整合策略**：
   - 提取检索内容中的关键概念、公式和定义
   - 将有价值的信息与你的知识体系结合
   - 忽略明显无关或错误的片段

3. **回答要求**：
   - 优先给出准确、完整的答案
   - 如果检索内容支持答案，可以说"根据教材内容..."
   - 如果检索内容不足以回答问题，请基于机器学习基础知识补充
   - 确保答案的逻辑性和连贯性

4. **特别注意**：
   - 不要被低质量或无关的检索片段误导
   - 如果检索内容与问题明显不匹配，可以说"检索到的内容相关性较低，基于机器学习原理..."
   - 始终以提供准确、有用的信息为目标

请基于以上原则回答用户问题。"""
                    
                    chain = self.rag_prompt | chat_model | self.output_parser
                    return chain.invoke({
                        "messages": messages,
                        "context_instruction": context_instruction
                    })
        
        # 使用普通聊天链
        chain = self.normal_prompt | chat_model | self.output_parser
        return chain.invoke({"messages": messages})
    
    def update_retrieval_settings(self, retrieval_k: int = None, rerank_top_k: int = None):
        """动态更新检索设置"""
        if retrieval_k is not None:
            self.retrieval_k = retrieval_k
        if rerank_top_k is not None:
            self.rerank_top_k = rerank_top_k
        print(f"检索设置已更新: k={self.retrieval_k}, rerank_top_k={self.rerank_top_k}")
    
    def get_retrieval_settings(self) -> dict:
        """获取当前检索设置"""
        return {
            "retrieval_k": self.retrieval_k,
            "rerank_top_k": self.rerank_top_k
        }
    
    def get_status(self) -> dict:
        """获取RAG状态"""
        if self.vector_store and self.vector_store.is_initialized():
            stats = self.vector_store.get_stats()
            return {
                "status": "active",
                "document_count": stats.get("document_count", 0),
                "embedding_type": stats.get("embedding_type", "N/A"),
                "model_name": stats.get("model_name", "N/A"),
                "retrieval_k": self.retrieval_k,
                "rerank_top_k": self.rerank_top_k
            }
        else:
            return {
                "status": "inactive",
                "retrieval_k": self.retrieval_k,
                "rerank_top_k": self.rerank_top_k
            }

# 创建全局RAG链实例
rag_chain = RAGChain()