import requests
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("提示: 安装 python-dotenv 可以使用.env文件管理环境变量")

# 从环境变量读取API密钥
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "your_default_key")


class BaseReranker(ABC):
    """重排序服务的基类"""
    
    def __init__(self, model: str):
        self.model = model
    
    @abstractmethod
    def rerank(self, query: str, documents: List[str], top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        重排序方法
        
        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回前k个结果，None表示返回全部
            
        Returns:
            [(文档内容, 相关性分数)] 列表，按分数降序排列
        """
        pass


class BaseAPIReranker(BaseReranker):
    """API重排序服务的基类"""
    
    def __init__(self, 
                 model: str,
                 api_url: str,
                 api_key: Optional[str] = None):
        super().__init__(model)
        self.api_url = api_url
        self.api_key = api_key
    
    @abstractmethod
    def _prepare_payload(self, query: str, documents: List[str]) -> dict:
        """准备API请求载荷"""
        pass
    
    @abstractmethod
    def _prepare_headers(self) -> dict:
        """准备API请求头"""
        pass
    
    @abstractmethod
    def _extract_results(self, response_data: dict, documents: List[str]) -> List[Tuple[str, float]]:
        """从API响应中提取重排序结果"""
        pass
    
    def rerank(self, query: str, documents: List[str], top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        """API重排序实现"""
        if not documents:
            return []
        
        payload = self._prepare_payload(query, documents)
        headers = self._prepare_headers()
        
        try:
            response = requests.post(self.api_url, json=payload, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            results = self._extract_results(data, documents)
            
            # 如果指定了top_k，只返回前k个结果
            if top_k is not None:
                results = results[:top_k]
            
            return results
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"重排序API请求失败: {e}")
        except ValueError as e:
            raise RuntimeError(f"解析API响应失败: {e}")
        except Exception as e:
            raise RuntimeError(f"重排序时发生未知错误: {e}")


class SiliconFlowReranker(BaseAPIReranker):
    """SiliconFlow API重排序服务"""
    
    def __init__(self, 
                 api_key: str,
                 model: str = "BAAI/bge-reranker-v2-m3",
                 api_url: str = "https://api.siliconflow.cn/v1/rerank"):
        super().__init__(
            model=model,
            api_url=api_url,
            api_key=api_key
        )
    
    def _prepare_payload(self, query: str, documents: List[str]) -> dict:
        return {
            "model": self.model,
            "query": query,
            "documents": documents
        }
    
    def _prepare_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def _extract_results(self, response_data: dict, documents: List[str]) -> List[Tuple[str, float]]:
        if "results" not in response_data:
            raise ValueError(f"API返回异常: {response_data}")
        
        # SiliconFlow返回格式可能与Xinference略有不同，需要适配
        results = response_data["results"]
        
        # 如果已经按分数排序，直接使用；否则手动排序
        if not self._is_sorted_by_score(results):
            results = sorted(results, key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        # 返回(文档内容, 分数)元组列表
        reranked_results = []
        for item in results:
            doc_index = item["index"]
            score = item.get("relevance_score", item.get("score", 0))  # 兼容不同字段名
            reranked_results.append((documents[doc_index], score))
        
        return reranked_results
    
    def _is_sorted_by_score(self, results: List[dict]) -> bool:
        """检查结果是否已按分数降序排列"""
        scores = [item.get("relevance_score", item.get("score", 0)) for item in results]
        return scores == sorted(scores, reverse=True)


class RerankFactory:
    """重排序服务工厂类"""
    
    @staticmethod
    def create_reranker(rerank_type: str, **kwargs) -> BaseReranker:
        """
        创建重排序服务实例
        
        Args:
            rerank_type: 重排序服务类型，只支持 "siliconflow"
            **kwargs: 其他参数
            
        Returns:
            重排序服务实例
        """
        if rerank_type.lower() == "siliconflow":
            api_key = kwargs.get("api_key")
            if not api_key:
                # 如果没有提供api_key，尝试使用环境变量
                api_key = SILICONFLOW_API_KEY
                if api_key == "your_default_key":
                    raise ValueError("SiliconFlow需要提供api_key参数或设置SILICONFLOW_API_KEY环境变量")
            
            return SiliconFlowReranker(
                api_key=api_key,
                model=kwargs.get("model", "BAAI/bge-reranker-v2-m3"),
                api_url=kwargs.get("api_url", "https://api.siliconflow.cn/v1/rerank")
            )
        
        else:
            raise ValueError(f"只支持SiliconFlow重排序服务，当前类型: {rerank_type}")
    
    @staticmethod
    def get_supported_types() -> List[str]:
        """获取支持的重排序服务类型"""
        return ["siliconflow"]


# 兼容旧接口的便捷函数
def rerank(query: str, docs: List[str], model: str = "BAAI/bge-reranker-v2-m3") -> List[str]:
    """
    兼容旧接口的重排序函数（使用SiliconFlow）
    
    Args:
        query: 查询文本
        docs: 文档列表
        model: 模型名称
        
    Returns:
        重排序后的文档列表（只返回文档内容，不含分数）
    """
    reranker = RerankFactory.create_reranker("siliconflow", model=model)
    results = reranker.rerank(query, docs)
    return [doc for doc, score in results]


def rerank_base(query: str, docs: List[str]) -> List[Tuple[str, float]]:
    """
    兼容旧接口的重排序函数（使用SiliconFlow，返回分数）
    
    Args:
        query: 查询文本
        docs: 文档列表
        
    Returns:
        重排序后的(文档内容, 分数)元组列表
    """
    reranker = RerankFactory.create_reranker("siliconflow")
    return reranker.rerank(query, docs)


# 便捷函数
def rerank_documents(query: str, 
                    documents: List[str], 
                    rerank_type: str = "siliconflow",  # 默认使用siliconflow
                    top_k: Optional[int] = None,
                    **kwargs) -> List[Tuple[str, float]]:
    """
    便捷函数：重排序文档
    
    Args:
        query: 查询文本
        documents: 文档列表
        rerank_type: 重排序服务类型，只支持 "siliconflow"
        top_k: 返回前k个结果
        **kwargs: 其他参数
        
    Returns:
        重排序后的(文档内容, 分数)元组列表
    """
    reranker = RerankFactory.create_reranker(rerank_type, **kwargs)
    return reranker.rerank(query, documents, top_k=top_k)


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
        
        # 测试数据
        query = "人工智能技术的发展"
        documents = [
            "人工智能是计算机科学的一个分支，致力于创建能够模拟人类智能的系统",
            "机器学习是人工智能的一个子领域，专注于算法的自动学习",
            "深度学习使用多层神经网络来处理复杂的数据模式",
            "自然语言处理是AI领域的重要应用，用于理解和生成人类语言",
            "计算机视觉技术使机器能够理解和解释视觉信息",
            "今天的天气很好，适合户外活动"
        ]
        
        try:
            print("\n测试SiliconFlow重排序服务...")
            
            # 方式1: 使用工厂模式创建
            siliconflow_reranker = RerankFactory.create_reranker(
                "siliconflow",
                api_key=SILICONFLOW_API_KEY,
                model="BAAI/bge-reranker-v2-m3"
            )
            print("✅ SiliconFlow重排序服务创建成功")
            
            # 测试重排序
            print(f"\n查询: {query}")
            print("原始文档顺序:")
            for i, doc in enumerate(documents):
                print(f"  {i+1}. {doc[:50]}...")
            
            results = siliconflow_reranker.rerank(query, documents, top_k=3)
            
            print(f"\n✅ 重排序后的前3个结果:")
            for i, (doc, score) in enumerate(results):
                print(f"  {i+1}. {doc[:50]}... (分数: {score:.4f})")
            
            # 方式2: 使用便捷函数
            print("\n测试便捷函数...")
            results2 = rerank_documents(
                query=query,
                documents=documents,
                rerank_type="siliconflow",
                api_key=SILICONFLOW_API_KEY,
                top_k=3
            )
            
            print("✅ 便捷函数测试成功:")
            for i, (doc, score) in enumerate(results2):
                print(f"  {i+1}. {doc[:50]}... (分数: {score:.4f})")
            
            # 测试兼容函数
            print("\n测试兼容函数...")
            reranked_docs = rerank(query, documents[:3])
            print("✅ 兼容函数测试成功:")
            for i, doc in enumerate(reranked_docs):
                print(f"  {i+1}. {doc[:50]}...")
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
    
    print("\n重排序服务测试完成")