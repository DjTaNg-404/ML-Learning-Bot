from typing import List, Optional, Dict, Any
from langchain.schema import Document
import os
import sys

# 修改导入方式，避免相对导入问题
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from vector_store import VectorStore
except ImportError:
    from .vector_store import VectorStore

try:
    from document_loader import DocumentLoader
except ImportError:
    from .document_loader import DocumentLoader

try:
    from rerank import rerank_base
except ImportError:
    from .rerank import rerank_base


class RetrievalSystem:
    def __init__(self, 
                 embedding_type: str = "siliconflow",  # 默认使用siliconflow
                 model_name: str = "BAAI/bge-large-zh-v1.5",  # 使用SiliconFlow推荐模型
                 persist_path: Optional[str] = None,
                 chunk_size: int = 500,  # 减少chunk size以避免API 413错误
                 chunk_overlap: int = 100,  # 相应减少overlap
                 **embedding_kwargs):  # 添加这个参数
        """
        初始化检索系统
        
        Args:
            embedding_type: 嵌入模型类型，只支持 "siliconflow"
            model_name: 模型名称，推荐使用 "BAAI/bge-large-zh-v1.5"
            persist_path: 持久化路径
            chunk_size: 文档分割大小，默认500以避免API请求过大
            chunk_overlap: 文档分割重叠，默认100
            **embedding_kwargs: 传递给嵌入服务的其他参数（如api_key等）
        """
        self.vector_store = VectorStore(
            embedding_type=embedding_type,
            model_name=model_name,
            persist_path=persist_path,
            **embedding_kwargs  # 传递额外参数
        )
        
        self.document_loader = DocumentLoader(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        self.is_initialized = False
    
    def build_index_from_files(self, file_paths: List[str]) -> bool:
        """
        从文件构建索引
        
        Args:
            file_paths: 文件路径列表
            
        Returns:
            构建是否成功
        """
        try:
            print("开始构建检索索引...")
            
            # 加载文档
            all_documents = self.document_loader.process_multiple_files(file_paths)
            
            if not all_documents:
                print("没有成功加载任何文档")
                return False
            
            print(f"成功加载 {len(all_documents)} 个文档片段")
            
            # 创建向量存储
            self.vector_store.create_from_documents(all_documents)
            self.is_initialized = True
            
            # 如果指定了持久化路径，自动保存
            if self.vector_store.persist_path:
                self.vector_store.save()
                print(f"索引已保存到: {self.vector_store.persist_path}")
            
            print("检索索引构建完成")
            return True
            
        except Exception as e:
            print(f"构建索引时出错: {e}")
            return False
    
    def load_index(self, path: Optional[str] = None) -> bool:
        """
        加载已有的索引
        
        Args:
            path: 索引路径
            
        Returns:
            加载是否成功
        """
        try:
            self.vector_store.load(path)
            self.is_initialized = True
            print("索引加载成功")
            return True
        except Exception as e:
            print(f"加载索引失败: {e}")
            return False
    
    def add_documents_from_files(self, file_paths: List[str]) -> bool:
        """
        向现有索引添加新文档
        
        Args:
            file_paths: 新文件路径列表
            
        Returns:
            添加是否成功
        """
        if not self.is_initialized:
            print("检索系统未初始化，请先构建或加载索引")
            return False
        
        try:
            new_documents = self.document_loader.process_multiple_files(file_paths)
            if new_documents:
                self.vector_store.add_documents(new_documents)
                print(f"成功添加 {len(new_documents)} 个新文档片段")
                
                # 如果有持久化路径，保存更新后的索引
                if self.vector_store.persist_path:
                    self.vector_store.save()
                    print("索引已更新并保存")
                
                return True
            else:
                print("没有成功加载新文档")
                return False
                
        except Exception as e:
            print(f"添加文档时出错: {e}")
            return False
    
    def retrieve(self, 
                query: str, 
                k: int = 5,
                use_rerank: bool = True,
                rerank_top_k: int = 10,
                return_scores: bool = False) -> List[Document]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            k: 返回结果数量
            use_rerank: 是否使用重排序
            rerank_top_k: 重排序前的候选数量
            return_scores: 是否返回分数
            
        Returns:
            相关文档列表
        """
        if not self.is_initialized:
            raise ValueError("检索系统未初始化，请先构建或加载索引")
        
        print(f"正在检索查询: '{query}'")
        
        if return_scores and not use_rerank:
            # 返回带分数的结果
            results = self.vector_store.similarity_search_with_score(query, k=k)
            print(f"检索到 {len(results)} 个相关文档")
            return results
        else:
            # 使用向量存储的检索方法（内置重排序逻辑）
            results = self.vector_store.similarity_search(
                query=query,
                k=k,
                use_rerank=use_rerank,
                rerank_top_k=rerank_top_k
            )
            print(f"检索到 {len(results)} 个相关文档")
            return results
    
    def batch_retrieve(self, 
                      queries: List[str], 
                      k: int = 5,
                      use_rerank: bool = True) -> Dict[str, List[Document]]:
        """
        批量检索
        
        Args:
            queries: 查询列表
            k: 每个查询返回的结果数量
            use_rerank: 是否使用重排序
            
        Returns:
            查询到文档列表的映射
        """
        results = {}
        
        for query in queries:
            try:
                results[query] = self.retrieve(
                    query=query, 
                    k=k, 
                    use_rerank=use_rerank
                )
            except Exception as e:
                print(f"查询 '{query}' 失败: {e}")
                results[query] = []
        
        return results
    
    def get_document_by_metadata(self, 
                               metadata_filter: Dict[str, Any]) -> List[Document]:
        """
        根据元数据筛选文档（简单实现）
        
        Args:
            metadata_filter: 元数据筛选条件
            
        Returns:
            匹配的文档列表
        """
        # 这是一个简化实现，实际使用中可能需要更复杂的筛选逻辑
        # 或者使用支持元数据筛选的向量数据库
        print("元数据筛选功能需要向量数据库支持，当前为简化实现")
        return []
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取检索系统统计信息
        
        Returns:
            统计信息字典
        """
        vector_stats = self.vector_store.get_stats()
        
        loader_stats = {
            "chunk_size": self.document_loader.chunk_size,
            "chunk_overlap": self.document_loader.chunk_overlap,
            "supported_formats": self.document_loader.get_supported_formats()
        }
        
        return {
            "system_status": "已初始化" if self.is_initialized else "未初始化",
            "vector_store": vector_stats,
            "document_loader": loader_stats
        }
    
    def save_index(self, path: Optional[str] = None):
        """保存索引"""
        if not self.is_initialized:
            raise ValueError("检索系统未初始化，无法保存")
        
        self.vector_store.save(path)
    
    def update_chunk_settings(self, chunk_size: int = None, chunk_overlap: int = None):
        """更新文档分割设置"""
        self.document_loader.update_chunk_settings(chunk_size, chunk_overlap)


class MultiIndexRetrieval:
    """多索引检索系统 - 支持多个独立的向量存储"""
    
    def __init__(self):
        self.retrievers: Dict[str, RetrievalSystem] = {}
    
    def add_retriever(self, name: str, retriever: RetrievalSystem):
        """添加检索器"""
        self.retrievers[name] = retriever
    
    def create_retriever(self, 
                        name: str,
                        file_paths: List[str],
                        embedding_type: str = "siliconflow",  # 默认使用siliconflow
                        model_name: str = "BAAI/bge-large-zh-v1.5",  # 使用SiliconFlow推荐模型
                        persist_path: Optional[str] = None,
                        chunk_size: int = 500,  # 添加chunk_size参数
                        chunk_overlap: int = 100,  # 添加chunk_overlap参数
                        **embedding_kwargs) -> bool:  # 添加这个参数
        """创建新的检索器"""
        retriever = RetrievalSystem(
            embedding_type=embedding_type,
            model_name=model_name,
            persist_path=persist_path,
            chunk_size=chunk_size,  # 传递chunk_size
            chunk_overlap=chunk_overlap,  # 传递chunk_overlap
            **embedding_kwargs  # 传递额外参数
        )
        
        success = retriever.build_index_from_files(file_paths)
        if success:
            self.retrievers[name] = retriever
            print(f"检索器 '{name}' 创建成功")
            return True
        else:
            print(f"检索器 '{name}' 创建失败")
            return False
    
    def retrieve_from_all(self, 
                         query: str, 
                         k_per_retriever: int = 3,
                         total_k: int = 5,
                         use_rerank: bool = True) -> List[Document]:
        """从所有检索器中检索并合并结果"""
        all_results = []
        
        for name, retriever in self.retrievers.items():
            try:
                results = retriever.retrieve(
                    query=query, 
                    k=k_per_retriever, 
                    use_rerank=False  # 先不用重排序，最后统一重排
                )
                all_results.extend(results)
            except Exception as e:
                print(f"从检索器 '{name}' 检索失败: {e}")
        
        # 如果使用重排序，对所有结果统一重排
        if use_rerank and all_results:
            docs_contents = [doc.page_content for doc in all_results]
            reranked_results = rerank_base(query, docs_contents)
            
            # 重新组织文档
            final_results = []
            for content, score in reranked_results[:total_k]:
                for doc in all_results:
                    if doc.page_content == content:
                        final_results.append(doc)
                        break
            
            return final_results
        
        return all_results[:total_k]
    
    def get_retriever_names(self) -> List[str]:
        """获取所有检索器名称"""
        return list(self.retrievers.keys())
    
    def get_retriever(self, name: str) -> Optional[RetrievalSystem]:
        """获取指定检索器"""
        return self.retrievers.get(name)


# 使用示例
if __name__ == "__main__":
    import os
    
    print("RAG检索系统测试")
    
    # 检查API密钥
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("提示: 建议安装 python-dotenv 包")
    
    SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "your_default_key")
    
    if SILICONFLOW_API_KEY == "your_default_key":
        print("❌ SiliconFlow API密钥未设置，请在.env文件中设置SILICONFLOW_API_KEY")
        print("创建.env文件，内容如下:")
        print("SILICONFLOW_API_KEY=sk-your-actual-api-key")
    else:
        print("✅ SiliconFlow API密钥已设置")
    
    # 获取测试文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_file = os.path.join(current_dir, "pumpkin_book.pdf")
    
    if not os.path.exists(test_file):
        print(f"❌ 测试文件不存在: {test_file}")
        print("请确保 pumpkin_book.pdf 文件在当前目录下")
        exit(1)
    
    print(f"✅ 找到测试文件: {os.path.basename(test_file)}")
    
    # 示例1: 基本使用
    print("\n=== 基本检索系统测试 ===")
    try:
        retrieval = RetrievalSystem(
            embedding_type="siliconflow",
            model_name="BAAI/bge-large-zh-v1.5",
            persist_path="./test_retrieval_index",
            chunk_size=500,  # 使用更小的chunk size
            chunk_overlap=50,  # 相应减少overlap
            api_key=SILICONFLOW_API_KEY
        )
        print("✅ 检索系统初始化成功")
        
        # 实际构建索引
        print(f"开始处理文件: {os.path.basename(test_file)}")
        success = retrieval.build_index_from_files([test_file])
        
        if success:
            print("✅ 索引构建成功")
            
            # 测试查询
            test_queries = [
                "什么是机器学习？",
                "深度学习的基本概念",
                "神经网络如何工作？"
            ]
            
            for query in test_queries:
                print(f"\n🔍 查询: {query}")
                try:
                    results = retrieval.retrieve(query, k=3, use_rerank=True)
                    print(f"✅ 检索到 {len(results)} 个结果")
                    
                    for i, doc in enumerate(results):
                        content_preview = doc.page_content[:100].replace('\n', ' ')
                        source = doc.metadata.get('source', '未知来源')
                        print(f"  {i+1}. {content_preview}...")
                        print(f"     来源: {source}")
                
                except Exception as e:
                    print(f"❌ 查询失败: {e}")
        
        else:
            print("❌ 索引构建失败")
        
        # 获取统计信息
        stats = retrieval.get_stats()
        print(f"\n📊 系统统计:")
        print(f"  系统状态: {stats['system_status']}")
        print(f"  向量存储: {stats['vector_store']}")
        print(f"  文档处理: chunk_size={stats['document_loader']['chunk_size']}, "
              f"chunk_overlap={stats['document_loader']['chunk_overlap']}")
        
    except Exception as e:
        print(f"❌ 基本检索系统测试失败: {e}")
    
    print("\n=== 多索引检索系统测试 ===")
    try:
        multi_retrieval = MultiIndexRetrieval()
        print("✅ 多索引检索系统初始化成功")
        
        # 创建检索器（使用同一个文件，但可以模拟不同的配置）
        success1 = multi_retrieval.create_retriever(
            name="南瓜书-完整版", 
            file_paths=[test_file],
            embedding_type="siliconflow",
            model_name="BAAI/bge-large-zh-v1.5",
            chunk_size=500,  # 使用更小的chunk size
            chunk_overlap=50,  # 相应减少overlap
            api_key=SILICONFLOW_API_KEY,
            persist_path="./multi_index_1"
        )
        
        if success1:
            print("✅ 多索引检索器创建成功")
            
            # 跨索引检索测试
            test_query = "机器学习的基本原理"
            print(f"\n🔍 跨索引查询: {test_query}")
            
            try:
                combined_results = multi_retrieval.retrieve_from_all(
                    query=test_query, 
                    k_per_retriever=2,
                    total_k=3,
                    use_rerank=True
                )
                print(f"✅ 跨索引检索成功，共 {len(combined_results)} 个结果")
                
                for i, doc in enumerate(combined_results):
                    content_preview = doc.page_content[:80].replace('\n', ' ')
                    print(f"  {i+1}. {content_preview}...")
            
            except Exception as e:
                print(f"❌ 跨索引检索失败: {e}")
        
        else:
            print("❌ 多索引检索器创建失败")
        
        retriever_names = multi_retrieval.get_retriever_names()
        print(f"📋 当前检索器列表: {retriever_names}")
        
    except Exception as e:
        print(f"❌ 多索引检索系统测试失败: {e}")
    
    print("\n🎉 检索系统测试完成")
    
    # 清理测试文件（可选）
    print("\n🧹 清理测试索引文件...")
    try:
        import shutil
        test_dirs = ["./test_retrieval_index", "./multi_index_1"]
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
                print(f"✅ 已删除: {test_dir}")
    except Exception as e:
        print(f"⚠️ 清理失败: {e} (可手动删除测试目录)")