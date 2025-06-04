import os
import uuid
from typing import Annotated, List
from dotenv import load_dotenv
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatOpenAI(temperature=0.7, model="gpt-4.1-mini")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant."),
        ("human", "{input}"),
    ]
)

class ChatState(TypedDict):
    messages: Annotated[List[dict], add_messages]

def to_dict(msg):
    if isinstance(msg, dict) and "role" in msg and "content" in msg:
        return {"role": msg["role"], "content": msg["content"]}
    if isinstance(msg, tuple) and len(msg) == 2:
        return {"role": msg[0], "content": msg[1]}
    if hasattr(msg, "role") and hasattr(msg, "content"):
        return {"role": getattr(msg, "role"), "content": getattr(msg, "content")}
    if hasattr(msg, "type") and hasattr(msg, "content"):
        return {"role": getattr(msg, "type"), "content": getattr(msg, "content")}
    print(f"Warning: Could not convert message to dict: {msg} (Type: {type(msg)})")
    content = str(msg)
    if hasattr(msg, 'type'):
         return {"role": str(getattr(msg, 'type')), "content": content}
    elif hasattr(msg, 'role'):
         return {"role": str(getattr(msg, 'role')), "content": content}
    return {"role": "unknown", "content": content}

def chatbot_agent(state: ChatState) -> ChatState:
    print("\n---Invoking Chatbot Agent---")
    messages_as_dicts = [to_dict(m) for m in state.get("messages", [])]
    if not messages_as_dicts:
         print("Warning: Chatbot agent received empty or unconvertible messages state.")
         return state
    last_message = messages_as_dicts[-1]
    print(f"Agent received message (as dict): {last_message}")
    if not isinstance(last_message, dict) or "content" not in last_message:
         print(f"Warning: Last message in state is still malformed after to_dict: {last_message}")
         return state
    chain = prompt | llm
    response = chain.invoke({"input": last_message["content"]})
    print(f"Agent sending response: {response.content}")
    return {"messages": [{"role": "assistant", "content": response.content}]}

chat_graph = StateGraph(ChatState)
chat_graph.add_node("chatbot_agent", chatbot_agent)
chat_graph.add_edge(START, "chatbot_agent")
chat_graph.add_edge("chatbot_agent", END)
memory = MemorySaver()
graph_app = chat_graph.compile(checkpointer=memory)

st.title("OpenAI LangGraph Chatbot 🤖")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "history" not in st.session_state:
    st.session_state.history = []

if st.button("Reset Conversation"):
    st.session_state.thread_id = None
    st.session_state.history = []
    st.rerun()

for msg in st.session_state.history:
    msg_dict = to_dict(msg)
    with st.chat_message(msg_dict["role"]):
        st.markdown(msg_dict["content"])

user_input = st.chat_input("Say something...")

if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    if st.session_state.thread_id is None:
        st.session_state.thread_id = uuid.uuid4().hex
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    state_input = {"messages": st.session_state.history}
    try:
        updated_state = graph_app.invoke(state_input, config=config)
        st.session_state.history = updated_state["messages"]
    except Exception as e:
        st.error(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    st.rerun()
