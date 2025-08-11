# 机器学习可视化学习平台 - Sugar-AI

一个集成了RAG（检索增强生成）功能的智能聊天机器人平台，专为机器学习学习和研究而设计。

## 🌟 功能特性

- **🤖 智能聊天机器人**：Sugar-AI 助手，专业解答机器学习相关问题
- **📚 RAG知识检索**：基于本地文档库的智能检索和问答
- **🔄 多模型支持**：支持硅基流动(SiliconFlow)等多种AI服务
- **⚡ 流式响应**：支持实时流式对话体验
- **🎛️ 可调参数**：灵活的检索参数和重排序设置
- **📊 可视化界面**：基于Streamlit的友好Web界面

## 🏗️ 项目架构

```
project/
├── backend/           # FastAPI后端服务
│   └── api_server.py  # API服务器
├── frontend/          # Streamlit前端界面
│   ├── Hello.py       # 主页
│   └── pages/
│       └── 聊天bot.py  # 聊天界面
├── llm/              # LLM相关模块
│   ├── chain_app.py   # 基础聊天链
│   ├── graph_app.py   # 对话图结构
│   └── rag_chain.py   # RAG增强链
├── rag/              # RAG检索模块
│   ├── document_loader.py  # 文档加载器
│   ├── embedding.py        # 向量嵌入
│   ├── rerank.py          # 重排序服务
│   └── vector_store.py    # 向量存储
├── ml/               # 机器学习模块
├── data/             # 数据存储
│   ├── ml_books/     # 机器学习书籍
│   └── vector_db/    # 向量数据库
└── .env              # 环境变量配置
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install fastapi uvicorn streamlit langchain langchain-openai
pip install python-dotenv faiss-cpu requests
```

### 2. 配置环境变量

创建 `.env` 文件并配置API密钥：

```env
# SiliconFlow API 配置
SILICONFLOW_API_KEY=sk-your-siliconflow-api-key-here
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
SILICONFLOW_MODEL=deepseek-ai/DeepSeek-V2.5

# 其他配置（可选）
DEEPSEEK_API_KEY=sk-your-deepseek-api-key-here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

### 3. 启动服务

**步骤1：启动后端服务**

```bash
cd d:\demo\project
uvicorn backend.api_server:app --reload
```

**步骤2：启动前端界面**

```bash
streamlit run Hello.py
```

### 4. 访问应用

浏览器访问 `http://localhost:8501` 即可使用。

## 📖 使用指南

### 基础聊天

- 直接输入问题与Sugar-AI对话
- 支持机器学习、数据分析等专业问题

### RAG知识检索

1. **启用RAG**：在侧边栏开启"启用知识检索(RAG)"
2. **调整参数**：
   - **检索条数**：控制返回的文档片段数量(1-20)
   - **重排序候选数**：重排序时的候选文档数量
3. **重排序**：开启"启用重排序"获得更精准的结果

### 参数说明

- **检索条数(retrieval_k)**：最终返回给用户的文档片段数量
- **重排序候选数(rerank_top_k)**：重排序阶段的候选文档数量，通常是检索条数的2-10倍
- **流式响应**：实时显示AI思考和回答过程

## 🛠️ 技术栈

- **后端框架**：FastAPI
- **前端框架**：Streamlit
- **AI模型**：SiliconFlow API
- **向量数据库**：FAISS
- **嵌入模型**：BAAI/bge-large-zh-v1.5
- **重排序模型**：BAAI/bge-reranker-v2-m3
- **对话管理**：LangGraph
- **文档处理**：LangChain

## 📁 数据管理

### 添加文档

1. 将PDF文档放入 `data/ml_books/` 目录
2. 重启服务，系统会自动处理新文档
3. 向量化处理完成后即可进行RAG检索

### 向量库管理

- 向量库存储在 `data/vector_db/` 目录
- 支持增量更新和持久化存储
- 可通过API查询向量库状态

## 🔒 安全说明

- ✅ `.env` 文件已加入 `.gitignore`，API密钥不会被提交、
- ✅ 支持环境变量和配置文件多种配置方式
