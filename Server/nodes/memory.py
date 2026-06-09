from langchain_core.messages import RemoveMessage, HumanMessage
from Server.state import ChatState
from Server.config import TRIM_THRESHOLD, KEEP_TARGET
from Server.utils import count_messages_tokens, summarize_messages

def manage_memory(state: ChatState):

    """Runs FIRST, before chat_node, when the message list is on a clean
    boundary (ends on the previous complete answer + the new user message).
    That position is what makes trimming cluster-safe."""

    messages = state["messages"]
    summary = state.get("summary", "")

    for message in messages:
        print("==============")
        print(f"Message: {message.type}")
        print(f"Message Content: {message.content}")
        print(f"Message ID: {message.id}")
        print("==============")
    
    tokens = count_messages_tokens(messages)
    if tokens < TRIM_THRESHOLD:
        return {}

    cluster_starts = [
        i for i, m in enumerate(messages) if isinstance(m, HumanMessage) and m.content.strip()
    ]

    cut_index = -1

    for i in reversed(cluster_starts):
        messages_tokens = count_messages_tokens(messages[i:])
        if messages_tokens > KEEP_TARGET:
            cut_index = i
            break

    if cut_index == -1:
        return {}

    print(f"Cut Index: {cut_index}")

    evicted = messages[:cut_index]

    print(f"Summarizing at => TRIM THRESHOLD: {TRIM_THRESHOLD} | TOKENS: {tokens}")

    summary = summarize_messages(summary, evicted)
    print(f"Summary: {summary}")

    #using the removemessage api from langgraph for trimming the earlier messages
    removals = [RemoveMessage(id=m.id) for m in evicted if getattr(m, "id", None) is not None]

    return {"messages": removals, "summary": summary}


