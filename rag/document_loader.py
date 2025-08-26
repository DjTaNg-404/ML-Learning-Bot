from pathlib import Path
from typing import List, Optional, Union
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
import pandas as pd
import re
import numpy as np
import warnings
from sklearn.metrics.pairwise import cosine_similarity

# 抑制数值计算警告
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')
np.seterr(divide='ignore', invalid='ignore', over='ignore')

# 导入本地的嵌入服务
try:
    from embedding import EmbeddingFactory
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    try:
        from rag.embedding import EmbeddingFactory
    except ImportError:
        print("警告: 无法导入embedding模块，语义分割功能可能不可用")


class DynamicSemanticChunker:
    def __init__(self, 
                 embedding_type: str = "siliconflow",
                 embedding_model: str = "BAAI/bge-large-zh-v1.5",
                 api_key: Optional[str] = None,
                 max_chunk_length: int = 512,
                 min_chunk_length: int = 50):
        """
        初始化动态语义分块器
        
        Args:
            embedding_type: 嵌入服务类型，目前支持 "siliconflow"
            embedding_model: 嵌入模型名称
            api_key: API密钥（如果为None，将使用环境变量）
            max_chunk_length: 最大块长度
            min_chunk_length: 最小块长度
        """
        self.embedding_type = embedding_type
        self.embedding_model_name = embedding_model
        self.api_key = api_key
        self.max_chunk_length = max_chunk_length
        self.min_chunk_length = min_chunk_length
        self.embedding_service = None
        
        # 延迟初始化嵌入服务
        self._init_embedding_service()
    
    def _init_embedding_service(self):
        """初始化嵌入服务"""
        try:
            kwargs = {
                "model": self.embedding_model_name
            }
            if self.api_key:
                kwargs["api_key"] = self.api_key
                
            self.embedding_service = EmbeddingFactory.create_embeddings(
                self.embedding_type, 
                **kwargs
            )
            print(f"✓ 嵌入服务初始化成功: {self.embedding_type}")
        except Exception as e:
            print(f"❌ 嵌入服务初始化失败: {e}")
            self.embedding_service = None
    
    def split_text(self, text: str) -> List[str]:
        """将文本分割成语义块，返回文本列表"""
        if not self.embedding_service:
            print("❌ 嵌入服务未初始化，使用简单分割")
            return self._fallback_split(text)
            
        sentences = self._split_into_sentences(text)
        if len(sentences) == 0:
            return []
        
        try:
            # 使用你的嵌入服务获取句子嵌入
            sentence_embeddings = self.embedding_service.embed_documents(sentences)
            sentence_embeddings = np.array(sentence_embeddings)
            
            # 验证嵌入结果
            if not self._validate_embeddings(sentence_embeddings, sentences):
                print("❌ 嵌入结果验证失败，使用简单分割")
                return self._fallback_split(text)
            
            gamma_values = self._compute_semantic_discrepancy(sentence_embeddings)
            
            total_tokens = sum(len(s.split()) for s in sentences)
            baseline_chunks = max(1, total_tokens // self.max_chunk_length)
            alpha = max(0.1, (len(sentences) - baseline_chunks) / len(sentences))
            threshold = np.quantile(gamma_values, alpha) if len(gamma_values) > 0 else 0.5
            
            boundaries = self._identify_boundaries(gamma_values, threshold)
            initial_chunks = self._create_initial_chunks(sentences, boundaries)
            final_chunks = self._enforce_length_constraints(initial_chunks)
            
            return final_chunks
            
        except Exception as e:
            print(f"❌ 语义分割失败: {e}")
            print("回退到简单分割模式")
            return self._fallback_split(text)
    
    def _fallback_split(self, text: str) -> List[str]:
        """回退分割方法，当嵌入服务不可用时使用"""
        # 简单的基于句号的分割
        sentences = text.split('。')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            test_chunk = current_chunk + sentence + "。"
            if len(test_chunk.split()) <= self.max_chunk_length:
                current_chunk = test_chunk
            else:
                if current_chunk and len(current_chunk.split()) >= self.min_chunk_length:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + "。"
        
        if current_chunk and len(current_chunk.split()) >= self.min_chunk_length:
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text]
    
    def _validate_embeddings(self, embeddings: np.ndarray, sentences: List[str]) -> bool:
        """验证嵌入结果的有效性"""
        if embeddings.shape[0] != len(sentences):
            print(f"❌ 嵌入数量不匹配: {embeddings.shape[0]} vs {len(sentences)}")
            return False
        
        # 检查是否有异常值
        if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
            print("❌ 嵌入包含NaN或无穷大值")
            return False
        
        # 检查是否所有向量都是零向量
        norms = np.linalg.norm(embeddings, axis=1)
        zero_vectors = np.sum(norms == 0)
        if zero_vectors > len(sentences) * 0.1:  # 超过10%的零向量
            print(f"❌ 过多零向量: {zero_vectors}/{len(sentences)}")
            return False
        
        return True
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """将文本分割成句子，并进行清理"""
        sentence_pattern = r'[。！？；\n]+'
        sentences = re.split(sentence_pattern, text)
        
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            # 增加更严格的过滤条件
            if len(sentence) > 5 and len(sentence) < 1000:  # 避免过长句子
                # 移除特殊字符，只保留中英文、数字和基本标点
                cleaned_sentence = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s\.,!?;:]', '', sentence)
                if len(cleaned_sentence) > 5:
                    cleaned_sentences.append(cleaned_sentence)
        
        return cleaned_sentences
    
    def _compute_semantic_discrepancy(self, embeddings: np.ndarray) -> List[float]:
        """计算相邻句子嵌入之间的语义差异，使用安全的数值计算"""
        gamma_values = []
        
        for i in range(1, len(embeddings)):
            vec1 = embeddings[i-1]
            vec2 = embeddings[i]
            
            # 检查异常值
            if np.any(np.isnan(vec1)) or np.any(np.isnan(vec2)) or \
               np.any(np.isinf(vec1)) or np.any(np.isinf(vec2)):
                print(f"警告: 发现异常向量值在位置 {i-1} 或 {i}")
                gamma_values.append(0.5)  # 默认中等差异
                continue
            
            # 计算向量模长
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            # 检查零向量
            if norm1 == 0 or norm2 == 0:
                print(f"警告: 发现零向量在位置 {i-1} 或 {i}")
                gamma_values.append(1.0)  # 最大差异
                continue
            
            # 使用手动计算余弦相似度，避免sklearn的数值问题
            try:
                dot_product = np.dot(vec1, vec2)
                similarity = dot_product / (norm1 * norm2)
                
                # 确保相似度在有效范围内 [-1, 1]
                similarity = np.clip(similarity, -1.0, 1.0)
                
                gamma = 1 - similarity
                gamma_values.append(float(gamma))
                
            except Exception as e:
                print(f"警告: 计算相似度异常 {e}, 位置 {i}")
                gamma_values.append(0.5)
        
        return gamma_values
    
    def _identify_boundaries(self, gamma_values: List[float], threshold: float) -> List[int]:
        boundaries = [0]
        
        for i, gamma in enumerate(gamma_values):
            if gamma > threshold:
                boundaries.append(i + 1)
        
        boundaries.append(len(gamma_values) + 1)
        return sorted(set(boundaries))
    
    def _create_initial_chunks(self, sentences: List[str], boundaries: List[int]) -> List[str]:
        chunks = []
        
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            
            chunk_sentences = sentences[start:end]
            chunk_text = ' '.join(chunk_sentences)
            chunks.append(chunk_text)
        
        return chunks
    
    def _enforce_length_constraints(self, chunks: List[str]) -> List[str]:
        final_chunks = []
        
        for chunk in chunks:
            chunk_length = len(chunk.split())
            
            if chunk_length <= self.max_chunk_length:
                if chunk_length >= self.min_chunk_length:
                    final_chunks.append(chunk)
            else:
                sub_chunks = self._split_long_chunk(chunk)
                final_chunks.extend(sub_chunks)
        
        return final_chunks
    
    def _split_long_chunk(self, chunk: str) -> List[str]:
        sentences = chunk.split('。')
        sub_chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if sentence.strip():
                test_chunk = current_chunk + sentence + "。"
                if len(test_chunk.split()) <= self.max_chunk_length:
                    current_chunk = test_chunk
                else:
                    if current_chunk:
                        sub_chunks.append(current_chunk.strip())
                    current_chunk = sentence + "。"
        
        if current_chunk:
            sub_chunks.append(current_chunk.strip())
        
        return sub_chunks


class DocumentLoader:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 200, split_mode: str = "recursive",
                 embedding_type: str = "siliconflow", embedding_model: str = "BAAI/bge-large-zh-v1.5", 
                 api_key: Optional[str] = None):
        """
        初始化文档加载器
        
        Args:
            chunk_size: 文档分割大小
            chunk_overlap: 文档分割重叠
            split_mode: 分割模式，"recursive" 或 "semantic"
            embedding_type: 嵌入服务类型（仅在semantic模式下使用）
            embedding_model: 嵌入模型名称（仅在semantic模式下使用）
            api_key: API密钥（仅在semantic模式下使用，如果为None将使用环境变量）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.split_mode = split_mode
        self.embedding_type = embedding_type
        self.embedding_model = embedding_model
        self.api_key = api_key
        
        # 初始化分割器
        if split_mode == "semantic":
            self.semantic_chunker = DynamicSemanticChunker(
                embedding_type=embedding_type,
                embedding_model=embedding_model,
                api_key=api_key,
                max_chunk_length=chunk_size,
                min_chunk_length=max(50, chunk_size // 10)
            )
            self.text_splitter = None
        else:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                add_start_index=True
            )
            self.semantic_chunker = None
        
        # 支持的文件格式
        self.supported_formats = {
            '.pdf': self.load_pdf,
            '.csv': self.load_csv,
            '.xlsx': self.load_excel,
            '.xls': self.load_excel,
            '.txt': self.load_text,
            '.md': self.load_text,
        }
    
    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """根据分割模式对文档进行分割"""
        if self.split_mode == "semantic":
            return self._semantic_split_documents(documents)
        else:
            return self.text_splitter.split_documents(documents)
    
    def _semantic_split_documents(self, documents: List[Document]) -> List[Document]:
        """使用语义分割器分割文档"""
        result_chunks = []
        
        for doc in documents:
            chunks = self.semantic_chunker.split_text(doc.page_content)
            
            for i, chunk_text in enumerate(chunks):
                # 创建新的Document对象，保留原有元数据
                chunk_doc = Document(
                    page_content=chunk_text,
                    metadata=doc.metadata.copy()
                )
                # 添加chunk序号
                chunk_doc.metadata['chunk_index'] = i
                result_chunks.append(chunk_doc)
        
        return result_chunks
    
    def process_file(self, file_path: Union[str, Path]) -> Optional[List[Document]]:
        """
        根据文件扩展名自动选择处理方法
        
        Args:
            file_path: 文件路径
            
        Returns:
            分割后的文档块列表
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            print(f"文件不存在: {file_path}")
            return None
        
        extension = file_path.suffix.lower()
        
        if extension not in self.supported_formats:
            print(f"不支持的文件格式: {extension}")
            print(f"支持的格式: {list(self.supported_formats.keys())}")
            return None
        
        try:
            print(f"正在处理 {extension} 文件: {file_path.name}")
            loader_func = self.supported_formats[extension]
            documents = loader_func(file_path)
            
            if documents:
                print(f"成功加载文档，共 {len(documents)} 个片段")
                return documents
            else:
                print("未能成功加载文档")
                return None
                
        except Exception as e:
            print(f"处理文件时出错: {e}")
            return None
    
    def process_multiple_files(self, file_paths: List[Union[str, Path]]) -> List[Document]:
        """
        批量处理多个文件
        
        Args:
            file_paths: 文件路径列表
            
        Returns:
            所有文档的分割片段列表
        """
        all_documents = []
        
        for file_path in file_paths:
            documents = self.process_file(file_path)
            if documents:
                all_documents.extend(documents)
        
        return all_documents
    
    def load_pdf(self, file_path: Path) -> List[Document]:
        """加载PDF文件"""
        pdf_loader = PyPDFLoader(str(file_path))
        documents = pdf_loader.load()
        
        # 为每个文档添加文件信息
        for doc in documents:
            doc.metadata['file_name'] = file_path.name
            doc.metadata['file_path'] = str(file_path)
            doc.metadata['file_type'] = 'pdf'
        
        # 使用通用分割方法
        document_chunks = self._split_documents(documents)
        return document_chunks
    
    def load_md(self, file_path: Path) -> List[Document]:
        """加载Markdown文件"""
        try:
            md_loader = UnstructuredMarkdownLoader(str(file_path))
            documents = md_loader.load()

            # 为每个文档添加文件信息
            for doc in documents:
                doc.metadata['file_name'] = file_path.name
                doc.metadata['file_path'] = str(file_path)
                doc.metadata['file_type'] = 'md'

            # 使用通用分割方法
            document_chunks = self._split_documents(documents)
            return document_chunks
            
        except Exception as e:
            print(f"加载Markdown文件失败: {e}")
            # 如果专用加载器失败，回退到文本加载器
            return self.load_text(file_path)

    def load_csv(self, file_path: Path) -> List[Document]:
        """加载CSV文件"""
        csv_loader = CSVLoader(file_path=str(file_path))
        documents = csv_loader.load()
        
        # 为每个文档添加文件信息
        for doc in documents:
            doc.metadata['file_name'] = file_path.name
            doc.metadata['file_path'] = str(file_path)
            doc.metadata['file_type'] = 'csv'
        
        # CSV通常不需要再分割，但如果内容很长可以分割
        if any(len(doc.page_content) > self.chunk_size for doc in documents):
            document_chunks = self._split_documents(documents)
            return document_chunks
        
        return documents
    
    def load_excel(self, file_path: Path) -> List[Document]:
        """加载Excel文件"""
        try:
            # 方法1: 使用pandas读取，然后转换为Document
            df = pd.read_excel(file_path)
            documents = []
            
            for index, row in df.iterrows():
                # 将每行转换为文本
                content = "\n".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                
                doc = Document(
                    page_content=content,
                    metadata={
                        'file_name': file_path.name,
                        'file_path': str(file_path),
                        'file_type': 'excel',
                        'row': index,
                        'sheet': 'Sheet1'  # 可以扩展支持多个sheet
                    }
                )
                documents.append(doc)
            
            # 如果内容很长，进行分割
            if any(len(doc.page_content) > self.chunk_size for doc in documents):
                document_chunks = self._split_documents(documents)
                return document_chunks
            
            return documents
            
        except Exception as e:
            print(f"使用pandas加载Excel失败: {e}")
            # 方法2: 使用UnstructuredExcelLoader作为备选
            try:
                excel_loader = UnstructuredExcelLoader(str(file_path))
                documents = excel_loader.load()
                
                for doc in documents:
                    doc.metadata['file_name'] = file_path.name
                    doc.metadata['file_path'] = str(file_path)
                    doc.metadata['file_type'] = 'excel'
                
                document_chunks = self._split_documents(documents)
                return document_chunks
                
            except Exception as e2:
                print(f"使用UnstructuredExcelLoader也失败: {e2}")
                return []
    
    def load_text(self, file_path: Path) -> List[Document]:
        """加载文本文件(.txt等)"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            document = Document(
                page_content=content,
                metadata={
                    'file_name': file_path.name,
                    'file_path': str(file_path),
                    'file_type': file_path.suffix[1:]  # 去掉点号
                }
            )
            
            # 使用通用分割方法
            document_chunks = self._split_documents([document])
            return document_chunks
            
        except Exception as e:
            print(f"加载文本文件失败: {e}")
            return []
    
    def get_supported_formats(self) -> List[str]:
        """获取支持的文件格式列表"""
        return list(self.supported_formats.keys())
    
    def update_chunk_settings(self, chunk_size: int = None, chunk_overlap: int = None, split_mode: str = None,
                             embedding_type: str = None, embedding_model: str = None, api_key: str = None):
        """更新分割设置"""
        if chunk_size is not None:
            self.chunk_size = chunk_size
        if chunk_overlap is not None:
            self.chunk_overlap = chunk_overlap
        if split_mode is not None:
            self.split_mode = split_mode
        if embedding_type is not None:
            self.embedding_type = embedding_type
        if embedding_model is not None:
            self.embedding_model = embedding_model
        if api_key is not None:
            self.api_key = api_key
            
        # 根据分割模式重新创建分割器
        if self.split_mode == "semantic":
            self.semantic_chunker = DynamicSemanticChunker(
                embedding_type=self.embedding_type,
                embedding_model=self.embedding_model,
                api_key=self.api_key,
                max_chunk_length=self.chunk_size,
                min_chunk_length=max(50, self.chunk_size // 10)
            )
            self.text_splitter = None
        else:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                add_start_index=True
            )
            self.semantic_chunker = None
        
        print(f"分割设置已更新: chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}, split_mode={self.split_mode}")


# 便捷函数
def load_documents_from_path(file_path: Union[str, Path], 
                           chunk_size: int = 500, 
                           chunk_overlap: int = 200,
                           split_mode: str = "recursive",
                           embedding_type: str = "siliconflow",
                           embedding_model: str = "BAAI/bge-large-zh-v1.5",
                           api_key: Optional[str] = None) -> List[Document]:
    """
    便捷函数：从单个文件加载文档
    
    Args:
        file_path: 文件路径
        chunk_size: 分割大小
        chunk_overlap: 分割重叠
        split_mode: 分割模式，"recursive" 或 "semantic"
        embedding_type: 嵌入服务类型（仅在semantic模式下使用）
        embedding_model: 嵌入模型名称（仅在semantic模式下使用）
        api_key: API密钥（仅在semantic模式下使用）
        
    Returns:
        文档片段列表
    """
    loader = DocumentLoader(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap, 
        split_mode=split_mode,
        embedding_type=embedding_type,
        embedding_model=embedding_model,
        api_key=api_key
    )
    return loader.process_file(file_path) or []


def load_documents_from_directory(directory_path: Union[str, Path],
                                file_extensions: Optional[List[str]] = None,
                                chunk_size: int = 500,
                                chunk_overlap: int = 200,
                                split_mode: str = "recursive",
                                embedding_type: str = "siliconflow",
                                embedding_model: str = "BAAI/bge-large-zh-v1.5",
                                api_key: Optional[str] = None) -> List[Document]:
    """
    便捷函数：从目录加载所有支持的文档
    
    Args:
        directory_path: 目录路径
        file_extensions: 指定文件扩展名列表，如['.pdf', '.txt']
        chunk_size: 分割大小
        chunk_overlap: 分割重叠
        split_mode: 分割模式，"recursive" 或 "semantic"
        embedding_type: 嵌入服务类型（仅在semantic模式下使用）
        embedding_model: 嵌入模型名称（仅在semantic模式下使用）
        api_key: API密钥（仅在semantic模式下使用）
        
    Returns:
        所有文档的片段列表
    """
    directory_path = Path(directory_path)
    if not directory_path.is_dir():
        print(f"目录不存在: {directory_path}")
        return []
    
    loader = DocumentLoader(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap, 
        split_mode=split_mode,
        embedding_type=embedding_type,
        embedding_model=embedding_model,
        api_key=api_key
    )
    
    # 如果没有指定扩展名，使用所有支持的格式
    if file_extensions is None:
        file_extensions = loader.get_supported_formats()
    
    all_documents = []
    
    # 遍历目录中的文件
    for file_path in directory_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in file_extensions:
            documents = loader.process_file(file_path)
            if documents:
                all_documents.extend(documents)
    
    return all_documents


# 使用示例
if __name__ == "__main__":
    # 示例1: 使用递归分割加载单个PDF文件
    loader = DocumentLoader(chunk_size=1000, chunk_overlap=200, split_mode="recursive")
    pdf_docs = loader.process_file("example.pdf")
    
    # 示例2: 使用语义分割加载文本文件（使用SiliconFlow API）
    semantic_loader = DocumentLoader(
        chunk_size=512, 
        split_mode="semantic",
        embedding_type="siliconflow",
        embedding_model="BAAI/bge-large-zh-v1.5",
        api_key=None  # 将使用环境变量中的API密钥
    )
    #text_docs = semantic_loader.process_file("example.txt")
    
    # 示例3: 批量加载多个文件（递归分割）
    file_paths = ["example.pdf"]
    all_docs = loader.process_multiple_files(file_paths)
    
    # 示例4: 从目录加载所有PDF和TXT文件（语义分割）
    #directory_docs = load_documents_from_directory(
    #    "./", 
    #    file_extensions=['.pdf', '.txt', '.md'],
    #    split_mode="semantic",
    #    embedding_type="siliconflow",
    #    api_key=None  # 将使用环境变量
    #)
    
    # 示例5: 动态切换分割模式
    loader.update_chunk_settings(
        chunk_size=800, 
        split_mode="semantic",
        embedding_type="siliconflow",
        embedding_model="BAAI/bge-large-zh-v1.5"
    )
    
    print(f"总共加载了 {len(all_docs)} 个文档片段")