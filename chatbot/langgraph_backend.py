from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
import os

load_dotenv()

# print(os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"))

llm = HuggingFaceEndpoint(
    # repo_id="MiniMaxAI/MiniMax-M2.1",
    # repo_id="openai/gpt-oss-20b",
    # repo_id="deepseek-ai/DeepSeek-R1",
    repo_id="deepseek-ai/DeepSeek-V4-Flash",
    # repo_id="deepseek-ai/DeepSeek-V4-Pro",
    # repo_id="google/gemma-3-12b",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm=llm)

# print(model.invoke("hi"))

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = model.invoke(messages)
    return {"messages": [response]}

# Checkpointer
checkpointer = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)