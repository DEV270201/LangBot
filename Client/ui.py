import streamlit as st
from Server.main import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk
import uuid

def generate_unique_thread_id():
    thread_id = str(uuid.uuid4())
    return thread_id

# whenever user clicks on new chat then we need to create a new thread id for that conversation as well as clean the UI for new messages
def reset_chat():
    st.session_state.messages = []
    st.session_state.thread_id = generate_unique_thread_id()
    add_thread_id(st.session_state.thread_id)

def add_thread_id(thread_id):
    # Streamlit reruns the script frequently; avoid duplicating ids in the sidebar list.
    if thread_id not in st.session_state.chat_thread_ids:
        st.session_state.chat_thread_ids.append(thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    # Check if messages key exists in state values, return empty list if not
    return state.values.get('messages', [])

# ---------
# CODE FOR STATE MANAGEMENT WHEN THE APP STARTS 
# we would be using session state for maintaining the state of the chat, the current session id and list of previous session ids of the chats
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_unique_thread_id()

if "chat_thread_ids" not in st.session_state:
    st.session_state.chat_thread_ids = retrieve_all_threads()

add_thread_id(st.session_state.thread_id)
# -------


# SIDEBAR 
st.sidebar.title("Devly's Chatbot")
if st.sidebar.button("New Chat"):
    reset_chat()
st.sidebar.header("My Conversations")

CONFIG = {"configurable": {"thread_id": st.session_state.thread_id}}

for thread_id in st.session_state.chat_thread_ids[::-1]:
    if st.sidebar.button(thread_id):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)
        # print("MESSAGES:", messages)
        temp_messages = []

        for msg in messages:
            # print("MSG:", msg.type)
            if isinstance(msg, HumanMessage) and msg.content.strip():
                temp_messages.append({'role': 'user', 'content': msg.content})
            elif isinstance(msg, AIMessage) and msg.content.strip():
                temp_messages.append({'role': 'assistant', 'content': msg.content})

        st.session_state['messages'] = temp_messages

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Type your message here...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    #invoke the chatbot
    # this will be the code for invoking the chatbot without streaming 
    # """
    # response = chatbot.invoke({"messages": [HumanMessage(content=user_input)]}, config=CONFIG)
    # chatbot_message = response.get("messages", {})[-1].content
    # st.session_state.messages.append({"role": "assistant", "content": chatbot_message})
    # """
    with st.chat_message("assistant"):

        def stream_ai_message():
            for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode='messages' #when you switch the stream mode to messages then a callback is passed down to langgraph which will stream the tokens from the LLM to the UI
                #when you switch the stream mode to custom then you have to handle the streaming yourself 
            ):
                # Stream only assistant tokens (skip ToolMessage JSON from the tools node)
                if isinstance(message_chunk, (AIMessage)) and message_chunk.content:
                    yield message_chunk.content


        ai_message = st.write_stream(stream_ai_message())
        st.session_state.messages.append({'role': 'assistant', 'content': ai_message})
        print("AI MESSAGE:", ai_message)