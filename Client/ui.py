import os
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from Server.main import chatbot, delete_thread, retrieve_all_threads

# ---------------------------------------------------------------------------
# Page + assets
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Devly's Chatbot",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded",
)

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
USER_AVATAR = os.path.join(ASSETS_DIR, "user.svg")
ASSISTANT_AVATAR = os.path.join(ASSETS_DIR, "assistant.svg")

# Friendly, human-readable labels for the tool status indicator.
TOOL_LABELS = {
    "duckduckgo_search": "Searching the web",
    "search": "Searching the web",
    "calculator": "Calculating",
}

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* ---- Sidebar shell ---- */
section[data-testid="stSidebar"] {
    background: #1c1f26;
    border-right: 1px solid rgba(255, 255, 255, 0.06);
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

/* ---- Sidebar brand ---- */
.sidebar-brand {
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: 0.2px;
    color: #f4f6fb;
    margin: 0 0 1rem 2px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.sidebar-section-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #8a91a0;
    margin: 0.4rem 0 0.4rem 4px;
}

/* ---- All sidebar buttons (conversation list) ---- */
section[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    text-align: left;
    justify-content: flex-start;
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.09);
    border-radius: 10px;
    padding: 0.5rem 0.7rem;
    color: #c7cdda;
    font-weight: 400;
    font-size: 0.9rem;
    line-height: 1.2;
    transition: background 0.12s ease, color 0.12s ease, border-color 0.12s ease;
    box-shadow: none;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
section[data-testid="stSidebar"] .stButton > button p {
    text-align: left;
    width: 100%;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255, 255, 255, 0.06);
    color: #ffffff;
    border-color: rgba(255, 255, 255, 0.18);
}

/* ---- Conversation row: chat button + actions menu ---- */
section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
    gap: 0.35rem;
    margin-bottom: 0.4rem;
}

/* Active conversation gets a highlighted card with an accent edge */
section[data-testid="stSidebar"] .st-key-chatrow-active [data-testid="stColumn"]:first-of-type .stButton > button {
    background: rgba(99, 102, 241, 0.18);
    border-color: rgba(99, 102, 241, 0.65);
    border-left: 3px solid #6366f1;
    color: #ffffff;
    font-weight: 600;
}

/* ---- Per-row actions (⋯) popover trigger ---- */
section[data-testid="stSidebar"] [data-testid="stPopover"] > div > button {
    width: 100%;
    border: 1px solid rgba(255, 255, 255, 0.09);
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.02);
    color: #c7cdda;
    padding: 0.5rem 0;
}
section[data-testid="stSidebar"] [data-testid="stPopover"] > div > button:hover {
    background: rgba(255, 255, 255, 0.06);
    border-color: rgba(255, 255, 255, 0.18);
    color: #ffffff;
}

/* ---- New Chat (primary) button ---- */
section[data-testid="stSidebar"] .stButton > button[kind="primary"],
section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #6366f1 0%, #06b6d4 100%);
    color: #ffffff;
    font-weight: 600;
    justify-content: center;
    text-align: center;
    border: none;
    box-shadow: 0 2px 10px rgba(99, 102, 241, 0.35);
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"] p,
section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] p {
    text-align: center;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover,
section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"]:hover {
    filter: brightness(1.07);
    border: none;
}

/* ---- Chat messages ---- */
[data-testid="stChatMessage"] {
    background: transparent;
    padding: 0.35rem 0.2rem;
}
[data-testid="stChatMessageAvatarUser"] img,
[data-testid="stChatMessageAvatarAssistant"] img,
[data-testid="stChatMessage"] img[alt="user avatar"],
[data-testid="stChatMessage"] img[alt="assistant avatar"] {
    border-radius: 50%;
}

/* ---- Tool status box: keep it quiet and grown-up ---- */
[data-testid="stStatus"] {
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.02);
    font-size: 0.85rem;
}

/* ---- Chat input ---- */
[data-testid="stChatInput"] {
    border-radius: 12px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Thread helpers
# ---------------------------------------------------------------------------
def generate_unique_thread_id():
    return str(uuid.uuid4())


def reset_chat():
    """New chat: fresh thread id + clean message panel.

    The new thread is intentionally NOT added to the sidebar here — an empty
    conversation shouldn't show up until the user sends its first message.
    """
    st.session_state.messages = []
    st.session_state.thread_id = generate_unique_thread_id()


def add_thread_id(thread_id):
    # Streamlit reruns the script frequently; avoid duplicating ids in the sidebar list.
    # New threads go to the front so the most recent conversation sits at the top.
    if thread_id not in st.session_state.chat_thread_ids:
        st.session_state.chat_thread_ids.insert(0, thread_id)


def move_to_top(thread_id):
    """Re-order the sidebar so the just-opened conversation jumps to the top."""
    ids = st.session_state.chat_thread_ids
    if thread_id in ids:
        ids.remove(thread_id)
    ids.insert(0, thread_id)


def select_thread(thread_id):
    """Open an existing conversation and surface it at the top of the list."""
    st.session_state.thread_id = thread_id
    move_to_top(thread_id)
    messages = load_conversation(thread_id)
    rebuilt = []
    for msg in messages:
        if isinstance(msg, HumanMessage) and msg.content.strip():
            rebuilt.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage) and msg.content.strip():
            rebuilt.append({"role": "assistant", "content": msg.content})
    st.session_state.messages = rebuilt


def rename_thread(thread_id, new_title):
    """Override a conversation's displayed title (UI-side only)."""
    new_title = " ".join(new_title.strip().split())
    if new_title:
        st.session_state.thread_titles[thread_id] = new_title


def remove_thread(thread_id):
    """Delete a conversation from Postgres and the sidebar."""
    delete_thread(thread_id)
    if thread_id in st.session_state.chat_thread_ids:
        st.session_state.chat_thread_ids.remove(thread_id)
    st.session_state.thread_titles.pop(thread_id, None)

    # If we just deleted the open conversation, fall back to the next one
    # (or spin up a fresh chat when nothing is left).
    if thread_id == st.session_state.thread_id:
        if st.session_state.chat_thread_ids:
            select_thread(st.session_state.chat_thread_ids[0])
        else:
            reset_chat()


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


def get_thread_title(thread_id):
    """Derive a readable title from the first user message of a thread.

    Real titles are cached so we don't re-hit the checkpointer every rerun;
    the placeholder is intentionally NOT cached so a brand-new thread picks up
    its title as soon as the first message lands.
    """
    cached = st.session_state.thread_titles.get(thread_id)
    if cached:
        return cached

    for msg in load_conversation(thread_id):
        if isinstance(msg, HumanMessage) and msg.content.strip():
            title = " ".join(msg.content.strip().split())
            title = title[:34] + "…" if len(title) > 34 else title
            st.session_state.thread_titles[thread_id] = title
            return title

    return "New conversation"


# ---------------------------------------------------------------------------
# Session state bootstrap
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_unique_thread_id()

if "chat_thread_ids" not in st.session_state:
    st.session_state.chat_thread_ids = retrieve_all_threads()

if "thread_titles" not in st.session_state:
    st.session_state.thread_titles = {}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-brand">💬 Devly\'s Chatbot</div>', unsafe_allow_html=True)

    if st.button("✦  New Chat", type="primary", width="stretch"):
        # Don't stack empty chats: only start fresh if the current one has content.
        if st.session_state.messages:
            reset_chat()
        st.rerun()

    st.markdown('<div class="sidebar-section-label">Conversations</div>', unsafe_allow_html=True)

    for thread_id in st.session_state.chat_thread_ids:
        title = get_thread_title(thread_id)
        is_active = thread_id == st.session_state.thread_id

        # A fixed key on the active row lets the stylesheet highlight it.
        row_key = "chatrow-active" if is_active else f"chatrow-{thread_id}"
        with st.container(key=row_key):
            chat_col, menu_col = st.columns([0.82, 0.18], gap="small")

            with chat_col:
                if st.button(title, key=f"thread-{thread_id}", width="stretch"):
                    select_thread(thread_id)
                    st.rerun()

            with menu_col:
                with st.popover("⋯", width="stretch"):
                    new_title = st.text_input(
                        "Rename conversation",
                        value=title,
                        key=f"rename-input-{thread_id}",
                    )
                    if st.button("Save", key=f"rename-save-{thread_id}", width="stretch"):
                        rename_thread(thread_id, new_title)
                        st.rerun()
                    if st.button("🗑  Delete", key=f"delete-{thread_id}", width="stretch"):
                        remove_thread(thread_id)
                        st.rerun()


# ---------------------------------------------------------------------------
# Main chat panel
# ---------------------------------------------------------------------------
CONFIG = {"configurable": {"thread_id": st.session_state.thread_id}}

for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else ASSISTANT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

user_input = st.chat_input("Type your message here…")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    # Register the thread in the sidebar now that it has real content. A fresh
    # conversation stays hidden until its first message lands here.
    add_thread_id(st.session_state.thread_id)
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        # Mutable holder so the generator can lazily create/update the status box.
        status_holder = {"box": None}

        def stream_ai_message():
            try:
                for message_chunk, metadata in chatbot.stream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=CONFIG,
                    stream_mode="messages",
                ):
                    # Lazily create & update a single, understated status box per tool run.
                    if isinstance(message_chunk, ToolMessage):
                        tool_name = getattr(message_chunk, "name", "tool")
                        label = TOOL_LABELS.get(tool_name, f"Running {tool_name}")
                        if status_holder["box"] is None:
                            status_holder["box"] = st.status(label, expanded=False)
                        else:
                            status_holder["box"].update(label=label, state="running")

                    # Stream only the chat reply — skip hidden summarization output.
                    if (
                        isinstance(message_chunk, AIMessage)
                        and message_chunk.content
                        and metadata.get("langgraph_node") == "chat_node"
                    ):
                        yield message_chunk.content
            finally:
                if status_holder["box"] is not None:
                    status_holder["box"].update(label="Done", state="complete", expanded=False)

        ai_message = st.write_stream(stream_ai_message())
        st.session_state.messages.append({"role": "assistant", "content": ai_message})

    # Refresh the sidebar so a brand-new thread shows its derived title immediately.
    st.rerun()
