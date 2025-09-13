# 机器学习智能助手 - Sugar-AI

一个基于 RAG（检索增强生成）技术的智能学习平台，集成了先进的文档处理、语义分块、向量检索和大语言模型对话功能，专为机器学习学习和研究而设计。

> 此项目为个人学习，开发并上传至 GitHub 的第一个项目，如果可以修改或提升的地方，欢迎提出。
> 目前还有许多仍未解决的地方：1，目前对话历史记录按 tokens 数来保留，但 tokens 数的计算是预估的；2，仅有普通对话 graph_app 中是有保留历史对话记忆的，而 rag_chain 并未做对话历史记录的处理；3，md类型文件的读写方式仍可改进，目前按照txt类型直接读入；4，机器学习部分仍不支持上传自定义数据集。
>
> 同时，也特别感谢 evi1boy 对本项目的支持和贡献。

## 🌟 核心特性

- **🤖 智能对话系统**：Sugar-AI 助手，专业解答机器学习、深度学习、数据科学相关问题
- **📚 高级 RAG 检索**：支持递归分割和动态语义分块的文档处理系统
- **🧠 多模型支持**：集成 SiliconFlow API，支持多种嵌入模型和重排序模型
- **⚡ 实时流式响应**：支持流式对话
- **🎛️ 智能参数调节**：动态调整检索条数、重排序候选数等参数
- **📊 直观 Web 界面**：基于 Streamlit 的现代化交互界面
- **🔧 灵活文档处理**：支持 PDF、Markdown、CSV、Excel 等多种格式
- **💾 持久化存储**：使用 FAISS 向量数据库，支持增量更新

## 🏗️ 项目架构

```
ML-Learning-Bot/
├── backend/                 # FastAPI 后端服务
│   └── api_server.py       # API 服务器，处理聊天请求和流式响应
├── frontend/               # Streamlit 前端界面
│   ├── Hello.py           # 主页和项目介绍
│   └── pages/
│       ├── 聊天bot.py      # 智能对话界面
│       └── 机器学习.py     # 机器学习专题页面
├── llm/                   # 大语言模型相关模块
│   ├── chain_app.py       # 基础聊天链和模型配置
│   ├── graph_app.py       # LangGraph 对话状态管理
│   └── rag_chain.py       # RAG 增强聊天链
├── rag/                   # RAG 检索系统核心模块
│   ├── document_loader.py  # 文档加载器（支持多格式，语义分块）
│   ├── embedding.py        # 嵌入服务（SiliconFlow API）
│   ├── vector_store.py     # 向量存储（FAISS）
│   └── rerank.py          # 重排序服务
├── data/                  # 数据存储目录
│   ├── ml_books/          # PDF 书籍存储
│   ├── hml-solutions/     # 机器学习习题解答（Markdown）
│   ├── pumpkin-book/      # 南瓜书文档（Markdown）
│   └── vector_db/         # FAISS 向量数据库文件
└── test_siliconflow_index/ # 测试向量索引
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <your-repo-url>
cd ML-Learning-Bot

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 环境配置

创建 `.env` 文件并配置 API 密钥：

```bash
# SiliconFlow API 配置
SILICONFLOW_API_KEY=your_siliconflow_api_key
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1

# LLM 配置
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5

# 重排序配置（可选）
RERANK_API_KEY=your_rerank_api_key
RERANK_BASE_URL=your_rerank_endpoint
```

### 3. 启动应用

```bash
# 启动后端服务
cd backend
python api_server.py

# 启动前端界面（新终端）
cd frontend
streamlit run Hello.py
```

访问 `http://localhost:8501` 开始使用！

## 💡 使用指南

### 文档导入
1. 在 `data/` 目录下放置您的文档文件
2. 支持批量导入多种格式文档
3. 系统会自动进行文档分块和向量化

### 对话模式
- **基础聊天**：通用对话，适合日常交流
- **RAG 增强**：基于文档知识库的专业问答
- **机器学习**：专注于 ML 算法和理论的深度讨论

### 高级功能
- 调整分块策略（递归/语义）以优化检索效果
- 使用重排序功能提升答案准确性
- 批量处理大型文档集合

## 📖 功能特性

### 🤖 智能对话
- **多模式聊天**：支持基础对话、RAG 增强对话、机器学习专题问答
- **流式响应**：实时生成回复，提升用户体验
- **上下文记忆**：基于 LangGraph 的对话状态管理

### 📚 文档处理
- **多格式支持**：PDF、Excel、CSV、TXT、Markdown 文档一键导入
- **智能分块**：
  - 递归字符分割（Recursive Character Splitter）
  - 语义分块（Semantic Chunking）：基于嵌入向量相似度的智能分割
- **批量处理**：支持大规模文档集合的高效处理

### 🔍 检索系统
- **向量检索**：基于 FAISS 的高效相似度搜索
- **重排序优化**：提升检索结果的相关性和准确性
- **多策略检索**：结合关键词和语义检索的混合策略

### 🧠 机器学习集成
- **专业知识库**：整合《统计学习方法》、《南瓜书》等经典教材
- **习题解答**：提供详细的机器学习习题解析
- **算法讲解**：涵盖从基础到进阶的机器学习算法

## 🛠️ 技术架构

### 核心技术栈
- **后端框架**：Flask + FastAPI
- **前端界面**：Streamlit
- **LLM 服务**：SiliconFlow API
- **向量数据库**：FAISS
- **文档处理**：LangChain + Unstructured
- **对话管理**：LangGraph 状态机
- **嵌入模型**：BAAI/bge-large-zh-v1.5
- **重排序**：BAAI/bge-reranker-v2-m3

## 🔧 配置说明

### 文档分块配置
```python
# 递归分块参数
CHUNK_SIZE = 500          # 分块大小
CHUNK_OVERLAP = 50        # 重叠长度

# 语义分块参数  
SEMANTIC_THRESHOLD = 0.3  # 语义相似度阈值
BREAKPOINT_PERCENTILE = 95 # 分割点百分位
```

## 🔗 相关资源

- [LangChain 文档](https://python.langchain.com/)
- [FAISS 向量检索](https://github.com/facebookresearch/faiss)
- [SiliconFlow API](https://docs.siliconflow.cn/)
- [Streamlit 框架](https://streamlit.io/)

## 📧 联系方式

如有问题或建议，欢迎提交 Issue 或联系项目维护者。

---

⭐ 如果这个项目对您有帮助，请给我们一个 Star！

- 向量库存储在 `data/vector_db/` 目录
- 支持增量更新和持久化存储
- 可通过API查询向量库状态

## 🔒 安全说明

- ✅ `.env` 文件已加入 `.gitignore`，API密钥不会被提交、
- ✅ 支持环境变量和配置文件多种配置方式
