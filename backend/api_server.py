from fastapi import FastAPI
from llm.graph_app import app_graph
from llm.chain_app import chain
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import json

app = FastAPI()

class ChatRequest(BaseModel):
    session_id: str
    question: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    config = {"configurable": {"thread_id": request.session_id}}
    input_messages = [HumanMessage(content=request.question)]
    output = app_graph.invoke({"messages": input_messages}, config)
    # output["messages"] 是历史消息列表，取最后一条 AI 回复
    return {"response": output["messages"][-1].content}
