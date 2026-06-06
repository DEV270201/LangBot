from datetime import date

from langchain_core.messages import SystemMessage

from Server.model import llm_with_tools
from Server.prompts import SYSTEM_PROMPT
from Server.state import ChatState


def chat_node(state: ChatState):
    messages = state["messages"]
    msgs_with_system = [
        SystemMessage(content=SYSTEM_PROMPT.format(today=date.today().isoformat())),
        *messages,
    ]
    response = llm_with_tools.invoke(msgs_with_system)
    return {"messages": [response]}
