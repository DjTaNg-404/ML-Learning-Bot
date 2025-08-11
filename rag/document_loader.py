from pathlib import Path
from typing import List, Optional, Union
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
import pandas as pd


class DocumentLoader:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 200):
        """
        初始化文档加载器
        
        Args:
            chunk_size: 文档分割大小
            chunk_overlap: 文档分割重叠
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True
        )
        
        # 支持的文件格式
        self.supported_formats = {
            '.pdf': self.load_pdf,
            '.csv': self.load_csv,
            '.xlsx': self.load_excel,
            '.xls': self.load_excel,
            '.txt': self.load_text,
            '.md': self.load_text,
        }
    
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
        
        # 分割文档
        document_chunks = self.text_splitter.split_documents(documents)
        return document_chunks
    
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
            document_chunks = self.text_splitter.split_documents(documents)
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
                document_chunks = self.text_splitter.split_documents(documents)
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
                
                document_chunks = self.text_splitter.split_documents(documents)
                return document_chunks
                
            except Exception as e2:
                print(f"使用UnstructuredExcelLoader也失败: {e2}")
                return []
    
    def load_text(self, file_path: Path) -> List[Document]:
        """加载文本文件(.txt, .md等)"""
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
            
            # 分割文档
            document_chunks = self.text_splitter.split_documents([document])
            return document_chunks
            
        except Exception as e:
            print(f"加载文本文件失败: {e}")
            return []
    
    def get_supported_formats(self) -> List[str]:
        """获取支持的文件格式列表"""
        return list(self.supported_formats.keys())
    
    def update_chunk_settings(self, chunk_size: int = None, chunk_overlap: int = None):
        """更新分割设置"""
        if chunk_size is not None:
            self.chunk_size = chunk_size
        if chunk_overlap is not None:
            self.chunk_overlap = chunk_overlap
            
        # 重新创建text_splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True
        )
        
        print(f"分割设置已更新: chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}")


# 便捷函数
def load_documents_from_path(file_path: Union[str, Path], 
                           chunk_size: int = 500, 
                           chunk_overlap: int = 200) -> List[Document]:
    """
    便捷函数：从单个文件加载文档
    
    Args:
        file_path: 文件路径
        chunk_size: 分割大小
        chunk_overlap: 分割重叠
        
    Returns:
        文档片段列表
    """
    loader = DocumentLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return loader.process_file(file_path) or []


def load_documents_from_directory(directory_path: Union[str, Path],
                                file_extensions: Optional[List[str]] = None,
                                chunk_size: int = 500,
                                chunk_overlap: int = 200) -> List[Document]:
    """
    便捷函数：从目录加载所有支持的文档
    
    Args:
        directory_path: 目录路径
        file_extensions: 指定文件扩展名列表，如['.pdf', '.txt']
        chunk_size: 分割大小
        chunk_overlap: 分割重叠
        
    Returns:
        所有文档的片段列表
    """
    directory_path = Path(directory_path)
    if not directory_path.is_dir():
        print(f"目录不存在: {directory_path}")
        return []
    
    loader = DocumentLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
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
    # 示例1: 加载单个PDF文件
    loader = DocumentLoader(chunk_size=1000, chunk_overlap=200)
    pdf_docs = loader.process_file("example.pdf")
    
    # 示例2: 批量加载多个文件
    file_paths = ["doc1.pdf", "data.csv", "notes.txt"]
    all_docs = loader.process_multiple_files(file_paths)
    
    # 示例3: 从目录加载所有PDF和TXT文件
    directory_docs = load_documents_from_directory(
        "./", 
        file_extensions=['.pdf', '.txt']
    )
    
    print(f"总共加载了 {len(all_docs)} 个文档片段")