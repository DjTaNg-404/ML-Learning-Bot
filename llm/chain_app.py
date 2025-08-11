from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai.chat_models import ChatOpenAI
from llm.config import DEEPSEEK_API_KEY, DEEPSEEK_API_BASE, MODEL_NAME

chat_model = ChatOpenAI(
    model=MODEL_NAME,
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base=DEEPSEEK_API_BASE,
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