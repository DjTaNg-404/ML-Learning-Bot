import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai.chat_models import ChatOpenAI

load_dotenv()

SILICONFLOW_API_KEY = os.getenv('SILICONFLOW_API_KEY')
SILICONFLOW_BASE_URL = os.getenv('SILICONFLOW_BASE_URL')
MODEL_NAME = os.getenv('SILICONFLOW_MODEL', 'Qwen/Qwen3-8B')  # 默认值

chat_model = ChatOpenAI(
    model=MODEL_NAME,
    openai_api_key=SILICONFLOW_API_KEY,
    openai_api_base=SILICONFLOW_BASE_URL,
    temperature=0.7
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一个智能管家，名字叫Sugar-AI，服务于机器学习可视化学习平台。"
        "你的职责包括：解答用户关于机器学习、数据分析、模型训练、平台使用等相关问题。"
        "请用简洁、准确的语言回复，必要时可结合平台功能进行说明。"
    ),
    MessagesPlaceholder(variable_name="messages"),
])

# 创建输出解析器
output_parser = StrOutputParser()

# 创建链：提示 -> 模型 -> 解析器
chain = prompt | chat_model | output_parser