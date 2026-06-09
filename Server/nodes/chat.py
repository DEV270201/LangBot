from datetime import date

from langchain_core.messages import SystemMessage

from Server.model import llm_with_tools
from Server.state import ChatState


def chat_node(state: ChatState):
    messages = state["messages"]
    print("length of messages: ", len(messages))
    print("=======================\n\n")
    summary = state.get("summary", "")
    SYSTEM_PROMPT = """You are a helpful assistant.

Today's date is {today}.

Rules:
- Always respond in English only. Never use other languages.
- Keep answers concise. Do not show step-by-step work unless the user asks.
- For ANY arithmetic (including simple expressions like 5+4), always call the calculator tool. Never compute in your head or pretend to use a tool.
- For expressions with multiple operations, call calculator repeatedly — one operation at a time — following standard order of operations.
- For current events, recent news, sports results, or anything time-sensitive, always call duckduckgo_search. Do not rely on training data for those topics.
- Do not say you searched or calculated unless you actually called the corresponding tool.

  Summary of the earlier conversation: {summary}
"""


    msgs_with_system = [
        SystemMessage(content=SYSTEM_PROMPT.format(today=date.today().isoformat(), summary=summary)),
        *messages,
    ]
    response = llm_with_tools.invoke(msgs_with_system)
    return {"messages": [response]}
