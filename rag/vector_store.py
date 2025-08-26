import os
import pickle
from pathlib import Path
from typing import List, Optional, Union
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# 修改导入方式，避免相对导入问题
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from embedding import EmbeddingFactory
except ImportError:
    # 如果上面的导入失败，尝试相对导入
    from .embedding import EmbeddingFactory

try:
    from rerank import rerank_base
except ImportError:
    # 如果上面的导入失败，尝试相对导入
    from .rerank import rerank_base

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("提示: 安装 python-dotenv 可以使用.env文件管理环境变量")

# 从环境变量读取API密钥
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "your_default_key")

class VectorStore:
    def __init__(self, 
                 embedding_type: str = "siliconflow",  # 默认使用siliconflow
                 model_name: str = "BAAI/bge-large-zh-v1.5",  # 使用SiliconFlow推荐模型
                 persist_path: Optional[str] = None,
                 **embedding_kwargs):
        """
        初始化向量存储
        
        Args:
            embedding_type: 嵌入模型类型，只支持 "siliconflow"
            model_name: 模型名称，推荐使用 "BAAI/bge-large-zh-v1.5"
            persist_path: 向量库持久化路径
            **embedding_kwargs: 传递给嵌入服务的其他参数（如api_key等）
        """
        self.embedding_type = embedding_type
        self.model_name = model_name
        self.persist_path = persist_path
        self.embedding_kwargs = embedding_kwargs
        self.vector_store = None
        
        # 初始化嵌入模型
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """初始化嵌入模型"""
        try:
            # 只支持SiliconFlow
            if self.embedding_type == "siliconflow":
                # 自动添加API密钥
                if "api_key" not in self.embedding_kwargs:
                    self.embedding_kwargs["api_key"] = SILICONFLOW_API_KEY
                
                self.embeddings = EmbeddingFactory.create_embeddings(
                    "siliconflow",
                    model=self.model_name,
                    **self.embedding_kwargs
                )
            else:
                raise ValueError(f"只支持SiliconFlow嵌入服务，当前类型: {self.embedding_type}")
                
            print(f"嵌入模型初始化成功: {self.embedding_type} - {self.model_name}")
        except Exception as e:
            print(f"嵌入模型初始化失败: {e}")
            raise
    
    def create_from_documents(self, documents: List[Document]) -> None:
        """
        从文档列表创建向量存储
        
        Args:
            documents: 文档列表
        """
        if not documents:
            raise ValueError("文档列表为空，无法创建向量存储")
        
        print(f"正在创建向量存储，文档数量: {len(documents)}")
        
        try:
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            print("向量存储创建完成")
        except Exception as e:
            print(f"创建向量存储失败: {e}")
            raise
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        向现有向量存储添加文档
        
        Args:
            documents: 要添加的文档列表
        """
        if not documents:
            print("文档列表为空，跳过添加")
            return
        
        if self.vector_store is None:
            print("向量存储未初始化，将创建新的向量存储")
            self.create_from_documents(documents)
        else:
            try:
                self.vector_store.add_documents(documents)
                print(f"已添加 {len(documents)} 个文档到向量存储")
            except Exception as e:
                print(f"添加文档失败: {e}")
                raise
    
    def similarity_search(self, 
                         query: str, 
                         k: int = 5, 
                         use_rerank: bool = False,
                         rerank_top_k: int = 10) -> List[Document]:
        """
        相似性搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            use_rerank: 是否使用重排序
            rerank_top_k: 重排序前的候选数量
            
        Returns:
            相关文档列表
        """
        if self.vector_store is None:
            raise ValueError("向量存储未初始化，请先创建向量存储")
        
        try:
            if use_rerank:
                # 先检索更多候选文档
                candidates = self.vector_store.similarity_search(query, k=min(rerank_top_k, k*2))
                
                if len(candidates) <= k:
                    # 候选文档数量不足，直接返回
                    return candidates
                
                # 提取文档内容进行重排序
                docs_contents = [doc.page_content for doc in candidates]
                reranked_results = rerank_base(query, docs_contents)
                
                # 根据重排序结果重新组织文档
                reranked_docs = []
                for content, score in reranked_results[:k]:
                    for doc in candidates:
                        if doc.page_content == content:
                            # 可以将重排序分数添加到元数据中
                            doc.metadata['rerank_score'] = float(score)
                            reranked_docs.append(doc)
                            break
                
                return reranked_docs
            else:
                return self.vector_store.similarity_search(query, k=k)
                
        except Exception as e:
            print(f"相似性搜索失败: {e}")
            raise
    
    def similarity_search_with_score(self, 
                                   query: str, 
                                   k: int = 5) -> List[tuple]:
        """
        带分数的相似性搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            (文档, 分数) 元组列表
        """
        if self.vector_store is None:
            raise ValueError("向量存储未初始化，请先创建向量存储")
        
        try:
            return self.vector_store.similarity_search_with_score(query, k=k)
        except Exception as e:
            print(f"带分数的相似性搜索失败: {e}")
            raise
    
    def save(self, path: Optional[str] = None) -> None:
        """
        保存向量存储到本地
        
        Args:
            path: 保存路径，如果为None则使用初始化时的persist_path
        """
        if self.vector_store is None:
            raise ValueError("向量存储未初始化，无法保存")
        
        save_path = path or self.persist_path
        if save_path is None:
            raise ValueError("未指定保存路径")
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # 保存FAISS索引
            self.vector_store.save_local(str(save_path))
            
            # 保存配置信息
            config = {
                "embedding_type": self.embedding_type,
                "model_name": self.model_name,
                "created_at": str(save_path),
                "document_count": self._get_document_count()
            }
            with open(save_path / "config.pkl", "wb") as f:
                pickle.dump(config, f)
            
            print(f"向量存储已保存到: {save_path}")
            
        except Exception as e:
            print(f"保存向量存储失败: {e}")
            raise
    
    def load(self, path: Optional[str] = None) -> None:
        """
        从本地加载向量存储
        
        Args:
            path: 加载路径，如果为None则使用初始化时的persist_path
        """
        load_path = path or self.persist_path
        if load_path is None:
            raise ValueError("未指定加载路径")
        
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"路径不存在: {load_path}")
        
        try:
            # 加载配置信息
            config_path = load_path / "config.pkl"
            if config_path.exists():
                with open(config_path, "rb") as f:
                    config = pickle.load(f)
                print(f"加载配置: {config}")
                
                # 验证模型配置是否匹配
                if config.get("embedding_type") != self.embedding_type:
                    print(f"警告: 保存的嵌入类型({config.get('embedding_type')}) "
                         f"与当前设置({self.embedding_type})不匹配")
            
            # 加载FAISS索引
            self.vector_store = FAISS.load_local(
                str(load_path), 
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"向量存储已从 {load_path} 加载完成")
            
        except Exception as e:
            print(f"加载向量存储失败: {e}")
            raise
    
    def _get_document_count(self) -> int:
        """获取文档数量"""
        if self.vector_store is None:
            return 0
        
        try:
            return self.vector_store.index.ntotal
        except:
            return 0
    
    def get_stats(self) -> dict:
        """
        获取向量存储统计信息
        
        Returns:
            统计信息字典
        """
        if self.vector_store is None:
            return {
                "status": "未初始化", 
                "document_count": 0,
                "embedding_type": self.embedding_type,
                "model_name": self.model_name
            }
        
        doc_count = self._get_document_count()
        
        return {
            "status": "已初始化",
            "embedding_type": self.embedding_type,
            "model_name": self.model_name,
            "document_count": doc_count,
            "persist_path": self.persist_path
        }
    
    def delete_documents(self, ids: List[str]) -> None:
        """
        删除指定ID的文档（如果支持）
        
        Args:
            ids: 要删除的文档ID列表
        """
        if self.vector_store is None:
            raise ValueError("向量存储未初始化")
        
        try:
            self.vector_store.delete(ids)
            print(f"已删除 {len(ids)} 个文档")
        except AttributeError:
            print("当前向量存储不支持删除操作")
        except Exception as e:
            print(f"删除文档失败: {e}")
            raise
    
    def is_initialized(self) -> bool:
        """检查是否已初始化"""
        return self.vector_store is not None


# 便捷函数保持不变，但增加错误处理
def create_vector_store_from_files(file_paths: Union[str, List[str]], 
                                  embedding_type: str = "siliconflow",  # 默认使用siliconflow
                                  model_name: str = "BAAI/bge-large-zh-v1.5",  # 使用SiliconFlow推荐模型
                                  chunk_size: int = 1024,
                                  chunk_overlap: int = 128,
                                  persist_path: Optional[str] = None,
                                  **embedding_kwargs) -> Optional[VectorStore]:
    """
    便捷函数：从文件创建向量存储
    
    Args:
        file_paths: 要处理的文件路径（字符串或字符串列表）
        embedding_type: 嵌入服务类型，只支持 "siliconflow"
        model_name: 模型名称
        chunk_size: 文档分块大小
        chunk_overlap: 分块重叠大小
        persist_path: 向量库持久化路径
        **embedding_kwargs: 传递给嵌入服务的其他参数
        
    Returns:
        VectorStore实例或None（如果失败）
    """
    try:
        try:
            from document_loader import DocumentLoader
        except ImportError:
            from .document_loader import DocumentLoader
        
        # 加载文档
        loader = DocumentLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap, split_mode="semantic")
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        
        documents = loader.process_multiple_files(file_paths)
        if not documents:
            print("没有成功加载任何文档")
            return None
        
        # 创建向量存储
        vector_store = VectorStore(
            embedding_type=embedding_type,
            model_name=model_name,
            persist_path=persist_path,
            **embedding_kwargs  # 传递额外的参数
        )
        vector_store.create_from_documents(documents)
        
        # 保存
        if persist_path:
            vector_store.save()
        
        return vector_store
        
    except Exception as e:
        print(f"创建向量存储失败: {e}")
        return None


# 使用示例
if __name__ == "__main__":
    print(f"当前环境变量中的API密钥: {SILICONFLOW_API_KEY}")
    
    # 检查API密钥是否正确设置
    if SILICONFLOW_API_KEY == "your_default_key":
        print("❌ SiliconFlow API密钥未设置，请在.env文件中设置SILICONFLOW_API_KEY")
        print("创建.env文件，内容如下:")
        print("SILICONFLOW_API_KEY=sk-your-actual-api-key")
    else:
        print("✅ SiliconFlow API密钥已设置")
        
        # 测试向量存储
        try:
            print("\n测试SiliconFlow向量存储...")
            vector_store = VectorStore(
                embedding_type="siliconflow",
                model_name="BAAI/bge-large-zh-v1.5",
                persist_path="./test_siliconflow_index"
            )
            
            # 创建测试文档
            from langchain.schema import Document
            test_docs = [
                Document(page_content="这是第一个测试文档，内容关于人工智能技术的发展", metadata={"source": "test1"}),
                Document(page_content="这是第二个测试文档，内容关于机器学习算法的应用", metadata={"source": "test2"}),
                Document(page_content="这是第三个测试文档，内容关于深度学习神经网络", metadata={"source": "test3"}),
            ]
            
            # 创建向量存储
            vector_store.create_from_documents(test_docs)
            print("✅ 向量存储创建成功")
            
            # 测试搜索
            search_results = vector_store.similarity_search("人工智能", k=2)
            print(f"✅ 搜索测试成功，找到 {len(search_results)} 个相关文档")
            for i, doc in enumerate(search_results):
                print(f"  文档{i+1}: {doc.page_content[:50]}...")
            
            # 测试保存
            vector_store.save()
            print("✅ 向量存储保存成功")
            
            # 测试加载
            new_vector_store = VectorStore(
                embedding_type="siliconflow",
                model_name="BAAI/bge-large-zh-v1.5",
                persist_path="./test_siliconflow_index"
            )
            new_vector_store.load()
            print("✅ 向量存储加载成功")
            
            # 显示统计信息
            stats = new_vector_store.get_stats()
            print(f"✅ 向量存储统计: {stats}")
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
    
    print("\n向量存储测试完成")