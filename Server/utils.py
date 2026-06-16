from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from Server.model import llm
from Server.config import TOKENIZER_NAME

import logging
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
from transformers import AutoTokenizer

def _format_messages_for_summary(messages: list[BaseMessage]) -> str:
    role_map = {"human": "User", "ai": "Assistant", "tool": "Tool", "system": "System"}
    lines = []
    for message in messages:
        role = role_map.get(message.type, message.type)
        content = message.content if isinstance(message.content, str) else str(message.content)
        if message.type == "tool" and len(content) > 400:
            content = content[:400] + "..."
        lines.append(f"{role}: {content}")
    return "\n".join(lines)

def summarize_messages(summary: str, messages: list[BaseMessage]) -> str:

    """Fold a freshly-evicted batch into the existing running summary.
    This is incremental on purpose: the raw messages are about to be deleted,
    so the next time we summarize, `previous_summary` is the only record of
    them. We never re-summarize from scratch.
    """

    formatted_messages = _format_messages_for_summary(messages)

    system_prompt = f"""You maintain a running summary of an ongoing conversation.
    Merge the EXISTING SUMMARY with the NEW MESSAGES into one updated summary.
    Prioritize (in order):
    1. User identity, preferences, constraints, and stated goals
    2. Decisions the user made and conclusions reached
    3. Important facts established about the task
    4. Open or unresolved threads

    De-emphasize or omit: greetings, filler, repeated tool call mechanics, and raw tool JSON unless a specific result mattered to the user.
    Write the summary only — no preamble or headings.
    Write the summary in English only, regardless of the language used in the conversation.

    EXISTING SUMMARY:

    {summary or '(none yet)'}

    NEW MESSAGES:
    {formatted_messages}

    """
    resp = llm.invoke([SystemMessage(content=system_prompt)])
    return resp.content if isinstance(resp.content, str) else str(resp.content)


def _to_chat_dicts(messages: list[BaseMessage]) -> list[dict]:
    role_map = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}
    return [
        {"role": role_map.get(m.type, "user"), "content": m.content}
        for m in messages
    ]

def count_messages_tokens(messages: list[BaseMessage]) -> int:

    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        chat = _to_chat_dicts(messages)

        chat_text = tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )
        chat_tokens = tokenizer.tokenize(chat_text)
        print(f"Chat Tokens Length: {len(chat_tokens)}")
        return len(chat_tokens)

    except Exception as e:
        print(f"Error running AutoTokenizer. Switching back to fallback method --- {e}")
        total_tokens = 0
        per_message_overhead = 4
        for message in messages:
            total_tokens += per_message_overhead + len(message.content.strip()) // 3
        return total_tokens


