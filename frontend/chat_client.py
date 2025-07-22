import streamlit as st
import requests
import json
import uuid

st.title("LLM 问答 Demo")
if "question" not in st.session_state:
    st.session_state.question = ""
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

user_question = st.text_input("请输入你的问题：", key="question")
if st.button("发送"):
    response = requests.post(
        "http://localhost:8000/chat",
        json={
            "session_id": st.session_state.session_id,
            "question": user_question
        }
    )
    st.write(response.json()["response"])