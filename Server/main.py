from datetime import date

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
# from langgraph.checkpoint.memory import InMemorySaver
# from langgraph.config import get_stream_writer
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import os

load_dotenv()
# from Server.llm import call_LLM
OLLAMA_URL = os.getenv("OLLAMA_URL")
LLM_MODEL = os.getenv("LLM_MODEL")

SYSTEM_PROMPT = """You are a helpful assistant.

Today's date is {today}.

Rules:
- Always respond in English only. Never use other languages.
- Keep answers concise. Do not show step-by-step work unless the user asks.
- For ANY arithmetic (including simple expressions like 5+4), always call the calculator tool. Never compute in your head or pretend to use a tool.
- For expressions with multiple operations, call calculator repeatedly — one operation at a time — following standard order of operations.
- For current events, recent news, sports results, or anything time-sensitive, always call duckduckgo_search. Do not rely on training data for those topics.
- Do not say you searched or calculated unless you actually called the corresponding tool."""

#initialize the LLM
llm = ChatOllama(model=LLM_MODEL, temperature=0)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# def chat_node(state: ChatState):
#     messages = state['messages']
#     writer = get_stream_writer()
#     full_response = ""
#     for chunk in call_LLM(messages):
#         writer(chunk)
#         full_response += chunk
#     return {"messages": [AIMessage(content=full_response)]}


# Tools
search_tool = DuckDuckGoSearchRun(region="us-en")

# Calculator tool
@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Perform one arithmetic operation on two numbers. Must be used for all math questions.

    Supported operations: add, sub, mul, div.
    For multi-step expressions, call this tool once per operation (e.g. 5*10, then +5, then -5, etc.).
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}

tools = [calculator, search_tool]
llm_with_tools = llm.bind_tools(tools)

def chat_node(state: ChatState):
    messages = state['messages']
    msgs_with_system = [
        SystemMessage(content=SYSTEM_PROMPT.format(today=date.today().isoformat())),
        *messages,
    ]
    response = llm_with_tools.invoke(msgs_with_system)
    return {"messages": [response]}

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
graph.add_node("tools", ToolNode(tools=tools))
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
#conditional edges are used to determine which tool to use based on the response from the LLM
#tools_condition returns tools if the response from the LLM is a tool call, otherwise it returns __end__ (meaning the conversation is over) thats why we do not add the end edge by ourself
graph.add_conditional_edges("chat_node", tools_condition) 
graph.add_edge("tools", "chat_node")
# graph.add_edge("chat_node", END)
chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)
