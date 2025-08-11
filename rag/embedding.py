import requests
from abc import ABC, abstractmethod
from typing import List, Optional
from langchain.embeddings.base import Embeddings
import os

try:
    from dotenv import load_dotenv
    load_dotenv()  # 这会加载项目根目录的.env文件
except ImportError:
    print("提示: 安装 python-dotenv 可以使用.env文件管理环境变量")

# 从环境变量读取API密钥
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "your_default_key") 

class BaseAPIEmbeddings(Embeddings, ABC):
    """API嵌入服务的基类"""
    
    def __init__(self, 
                 model: str,
                 api_url: str,
                 api_key: Optional[str] = None,
                 batch_size: int = 16):
        self.model = model
        self.api_url = api_url
        self.api_key = api_key
        self.batch_size = batch_size
    
    @abstractmethod
    def _prepare_payload(self, texts: List[str]) -> dict:
        """准备API请求载荷"""
        pass
    
    @abstractmethod
    def _prepare_headers(self) -> dict:
        """准备API请求头"""
        pass
    
    @abstractmethod
    def _extract_embeddings(self, response_data: dict) -> List[List[float]]:
        """从API响应中提取嵌入向量"""
        pass
    
    def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """批量获取嵌入向量"""
        payload = self._prepare_payload(texts)
        headers = self._prepare_headers()
        
        try:
            response = requests.post(self.api_url, json=payload, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            return self._extract_embeddings(data)
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API请求失败: {e}")
        except ValueError as e:
            raise RuntimeError(f"解析API响应失败: {e}")
        except Exception as e:
            raise RuntimeError(f"获取嵌入向量时发生未知错误: {e}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入多个文档"""
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self._get_embeddings_batch(batch)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询"""
        return self._get_embeddings_batch([text])[0]

class SiliconFlowEmbeddings(BaseAPIEmbeddings):
    """SiliconFlow API嵌入服务"""
    
    def __init__(self, 
                 api_key: str,
                 model: str = "BAAI/bge-large-zh-v1.5",
                 api_url: str = "https://api.siliconflow.cn/v1/embeddings",
                 batch_size: int = 16):
        super().__init__(
            model=model,
            api_url=api_url,
            api_key=api_key,
            batch_size=batch_size
        )
    
    def _prepare_payload(self, texts: List[str]) -> dict:
        # SiliconFlow支持单个文本或文本列表
        input_data = texts if len(texts) > 1 else texts[0]
        return {
            "model": self.model,
            "input": input_data
        }
    
    def _prepare_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def _extract_embeddings(self, response_data: dict) -> List[List[float]]:
        # 处理单个文本或多个文本的响应
        embeddings = [item["embedding"] for item in response_data["data"]]
        return embeddings


class EmbeddingFactory:
    """嵌入服务工厂类"""
    
    @staticmethod
    def create_embeddings(embedding_type: str, **kwargs) -> Embeddings:
        """
        创建嵌入服务实例
        
        Args:
            embedding_type: 嵌入服务类型，只支持 "siliconflow"
            **kwargs: 其他参数
            
        Returns:
            嵌入服务实例
        """
        if embedding_type.lower() == "siliconflow":
            api_key = kwargs.get("api_key")
            if not api_key:
                # 如果没有提供api_key，尝试使用环境变量
                api_key = SILICONFLOW_API_KEY
                if api_key == "your_default_key":
                    raise ValueError("SiliconFlow需要提供api_key参数或设置SILICONFLOW_API_KEY环境变量")
            
            return SiliconFlowEmbeddings(
                api_key=api_key,
                model=kwargs.get("model", "BAAI/bge-large-zh-v1.5"),
                api_url=kwargs.get("api_url", "https://api.siliconflow.cn/v1/embeddings"),
                batch_size=kwargs.get("batch_size", 16)
            )
        
        else:
            raise ValueError(f"只支持SiliconFlow嵌入服务，当前类型: {embedding_type}")
    
    @staticmethod
    def get_supported_types() -> List[str]:
        """获取支持的嵌入服务类型"""
        return ["siliconflow"]


# 便捷函数
def get_embeddings(texts: List[str], 
                  embedding_type: str = "siliconflow", 
                  **kwargs) -> List[List[float]]:
    """
    便捷函数：获取文本嵌入向量
    
    Args:
        texts: 文本列表
        embedding_type: 嵌入服务类型，只支持 "siliconflow"
        **kwargs: 其他参数
        
    Returns:
        嵌入向量列表
    """
    embeddings = EmbeddingFactory.create_embeddings(embedding_type, **kwargs)
    return embeddings.embed_documents(texts)


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
        
        # 测试SiliconFlow嵌入服务
        try:
            print("\n测试SiliconFlow嵌入服务...")
            
            # 方式1: 使用工厂模式创建
            siliconflow_emb = EmbeddingFactory.create_embeddings(
                "siliconflow",
                api_key=SILICONFLOW_API_KEY,
                model="BAAI/bge-large-zh-v1.5"
            )
            print("✅ SiliconFlow嵌入服务创建成功")
            
            # 测试文本
            test_texts = [
                "这是一个关于人工智能的测试文本", 
                "这是另一个关于机器学习的测试文本",
                "这是第三个关于深度学习的测试文本"
            ]
            
            # 方式2: 使用便捷函数
            print("\n测试便捷函数...")
            embeddings = get_embeddings(
                test_texts, 
                embedding_type="siliconflow",
                api_key=SILICONFLOW_API_KEY
            )
            
            print(f"✅ 成功获取到 {len(embeddings)} 个嵌入向量")
            print(f"向量维度: {len(embeddings[0])}")
            print(f"第一个向量前5个值: {embeddings[0][:5]}")
            
            # 测试单个查询
            print("\n测试单个查询...")
            query_vector = siliconflow_emb.embed_query("什么是人工智能？")
            print(f"✅ 查询向量维度: {len(query_vector)}")
            print(f"查询向量前5个值: {query_vector[:5]}")
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
    
    print("\n嵌入服务测试完成")