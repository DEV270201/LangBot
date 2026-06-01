import streamlit as st
from Server.main import chatbot
from langchain_core.messages import HumanMessage

CONFIG = {"configurable": {"thread_id": "123"}}

# we would be using session state for maintaining the state of the chat 
if "messages" not in st.session_state:
    st.session_state.messages = []

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
            for chunk in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode='custom'
            ):
                yield chunk

        ai_message = st.write_stream(stream_ai_message())
        st.session_state.messages.append({'role': 'assistant', 'content': ai_message})
        print("AI MESSAGE:", ai_message)