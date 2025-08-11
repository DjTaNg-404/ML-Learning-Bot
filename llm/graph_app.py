from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from llm.chain_app import chain
from langchain_core.messages import AIMessage, trim_messages
from langchain_ollama import ChatOllama


workflow = StateGraph(state_schema=MessagesState)

'''
trimmer = trim_messages(
    max_tokens=1024,
    strategy="last",
    token_counter=ChatOllama(model="qwen3:0.6b" ),
    include_system=True,
    allow_partial=False,
    start_on="human",
)
'''

def call_model(state: MessagesState):
    
    '''trimmed_history = trimmer.invoke(state["messages"])
    response = chain.invoke({"messages": trimmed_history})
    new_history = trimmed_history + [AIMessage(content=response)]
    '''
    max_history = 10
    trimmed_history = state["messages"][-max_history:]
    response = chain.invoke({"messages": trimmed_history})
    new_history = trimmed_history + [AIMessage(content=response)]
    print(new_history)
    return {"messages": new_history}

workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app_graph = workflow.compile(checkpointer=memory)