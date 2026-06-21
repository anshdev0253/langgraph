import streamlit as st
from langgraph_backend import chatbot, model
from langchain_core.messages import HumanMessage, SystemMessage
import uuid


# utility functions
def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(thread_id)
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'][thread_id] = "New Chat"


def generate_title(user_message, ai_message):
    prompt = [
        SystemMessage(content="Generate a very short title (4-6 words max) for a chat conversation based on the first exchange. Reply with only the title, no quotes or punctuation."),
        HumanMessage(content=f"User: {user_message}\nAssistant: {ai_message}")
    ]
    response = model.invoke(prompt)
    return response.content.strip()

def load_conversation(thread_id):
    return chatbot.get_state(config={'configurable': {'thread_id': thread_id}}).values['messages']


# Session setup
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = {}

add_thread(st.session_state['thread_id'])



# Sidebar UI
st.sidebar.title("Langgraph Chatbot")

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversations')

for thread_id, title in reversed(st.session_state['chat_threads'].items()):
    if st.sidebar.button(title, key=str(thread_id)):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []

        for message in messages:
            if isinstance(message, HumanMessage):
                role = 'user'
            else:
                role = 'assistant'
            temp_messages.append({'role': role, 'content': message.content})

        st.session_state['message_history'] = temp_messages
        



# Main UI

# Loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])


user_input = st.chat_input("Type here")

if user_input:

    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)
    
    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

    with st.chat_message('assistant'):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config = CONFIG,
                stream_mode = 'messages'
            )
        )

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})

    # Generate title after the first message exchange
    thread_id = st.session_state['thread_id']
    if st.session_state['chat_threads'].get(thread_id) == "New Chat":
        title = generate_title(user_input, ai_message)
        st.session_state['chat_threads'][thread_id] = title
        st.rerun()  # Re-render the page so the sidebar shows the updated title
