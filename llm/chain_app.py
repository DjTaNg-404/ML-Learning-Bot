from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

chat_model = ChatOllama(model="qwen3:4b" )

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个管家，名字叫做Tang-AI，你需要回答用户的问题。语气言简意赅，准确。"),
    MessagesPlaceholder(variable_name="messages"),
])

# 创建输出解析器
output_parser = StrOutputParser()

# 创建链：提示 -> 模型 -> 解析器
chain = prompt | chat_model | output_parser