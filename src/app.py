import streamlit as st
from brain import get_brind_ai_response

# Page Configuration
st.set_page_config(page_title="OpenMentor-Brind", page_icon="💬", layout="centered")

# Inject custom CSS to mimic WeChat's clean grey background style
st.markdown("""
    <style>
    .stApp { background-color: #f2f2f2; }
    </style>
""", unsafe_allow_html=True)

st.title("💬 Brind 智库 (OpenMentor)")
st.caption("Powered by Gemini 3.5 Flash & Google Drive Automation")

# Initialize chat history inside session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "你好。今天想探讨什么？"}]

# Render chat messages with left/right speech bubbles
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input from the chat box
if user_input := st.chat_input("向 Brind 提问..."):
    # Append and display user message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
    # Generate and display assistant response using the brain pipeline
    with st.chat_message("assistant"):
        with st.spinner("正在检索Brind思维笔记..."):
            try:
                response = get_brind_ai_response(user_input)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error connecting to brain. Please check your Secrets or Google Drive permissions. Details: {e}")
