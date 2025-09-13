from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from llm.chain_app import chain
from langchain_core.messages import AIMessage, trim_messages



workflow = StateGraph(state_schema=MessagesState)

def simple_token_counter(messages):
    """
    智能token计数估算
    - 中文字符: 1.5 tokens
    - 英文字符: 0.25 tokens  
    - 考虑格式化标记的额外开销
    """
    total_tokens = 0
    for msg in messages:
        if hasattr(msg, 'content'):
            content = str(msg.content)
            # 计算基础字符token
            chinese_chars = len([c for c in content if '\u4e00' <= c <= '\u9fff'])
            other_chars = len(content) - chinese_chars
            base_tokens = chinese_chars * 1.5 + other_chars * 0.25
            
            # 为消息结构添加开销 (role, metadata等)
            structure_overhead = 10
            total_tokens += int(base_tokens + structure_overhead)
    return total_tokens

# 优化trimmer配置 - 适应更长的购物对话
trimmer = trim_messages(
    max_tokens=30000,  # 提高到30000，支持更多轮对话
    strategy="last",   # 保留最新的对话
    token_counter=simple_token_counter,
    include_system=True,  # 保留系统prompt
)

def call_model(state: MessagesState):
    # 使用智能trimmer处理消息历史
    trimmed_history = trimmer.invoke(state["messages"])
    response = chain.invoke({"messages": trimmed_history})
    new_history = trimmed_history + [AIMessage(content=response)]
    print(f"Trimmed history length: {len(trimmed_history)}")
    return {"messages": new_history}

workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app_graph = workflow.compile(checkpointer=memory)