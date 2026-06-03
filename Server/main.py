from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
# from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
import os
load_dotenv()
from Server.llm import call_LLM

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    writer = get_stream_writer()
    full_response = ""
    for chunk in call_LLM(messages):
        writer(chunk)
        full_response += chunk
    return {"messages": [AIMessage(content=full_response)]}

# --- Postgres checkpointer ---
DATABASE_URI = os.getenv('DATABASE_URI')

pool = ConnectionPool(
    conninfo=DATABASE_URI,
    max_size=10,
    kwargs={
        "autocommit": True,        # PostgresSaver requires this
        "prepare_threshold": 0,    # recommended for the checkpointer's query patterns
    },
)

checkpointer = PostgresSaver(pool)
checkpointer.setup()   # creates the checkpoint tables on first run; idempotent

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)
chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)
