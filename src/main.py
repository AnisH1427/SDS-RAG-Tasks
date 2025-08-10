"""
Streamlit AI Chatbot App
Built with LangChain + Groq + Streamlit
"""

import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import yaml
from retrieval import get_top_k_chunks

import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


# Load environment variables from .env in current directory
load_dotenv()


# Initialize the LLM with caching for performance
@st.cache_resource
def load_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("❌ GROQ_API_KEY not found in environment variables!")
        st.error("Please make sure you have a .env file with your Groq API key.")
        st.stop()
    return ChatGroq(
        model=os.getenv("MODEL"),
        temperature=0.7,
        api_key=os.getenv("GROQ_API_KEY")
    )


def load_system_prompt():
    with open(os.path.join(os.path.dirname(__file__), '../config/prompt_config.yaml'), 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    sys_prompt = config['system_prompt']['description'].strip()
    constraints = '\n'.join(['- ' + c for c in config['system_prompt'].get('constraints', [])])
    tone = config['system_prompt'].get('tone', '')
    full_prompt = f"{sys_prompt}\n\nTone: {tone}\n\nConstraints:\n{constraints}\n"
    return full_prompt


def get_ai_response(user_message: str, llm) -> str:
    # Retrieve context
    top_chunks = get_top_k_chunks(user_message, k=3)
    context = '\n---\n'.join([f"Page {c['page']}: {c['text']}" for c in top_chunks if c['text']])
    system_prompt = load_system_prompt()
    full_prompt = f"{system_prompt}\n\nContext:\n{context}\n\nUser Question: {user_message}"
    messages = [
        SystemMessage(content=full_prompt),
        HumanMessage(content=user_message),
    ]
    response = llm.invoke(messages)
    return response.content


def main():
    # Page configuration
    st.set_page_config(
        page_title="AI Chatbot Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Main title and description
    st.title("🤖 RAG based AI Chatbot Assistant")
    st.markdown("Ask me about APPLE INC. (AAPL) filings, financials, and more!")

    # Load the LLM
    llm = load_llm()

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What would you like to know?"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_ai_response(prompt, llm)
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

    # Sidebar for additional features
    with st.sidebar:
        st.markdown("## Smart Data Solution Technical Tasks")
        st.markdown("---")
        st.markdown("### Chat Controls")

        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")
        st.markdown("### Stats")
        st.metric("Messages in Chat", len(st.session_state.messages))

        if st.session_state.messages:
            user_messages = len([msg for msg in st.session_state.messages if msg["role"] == "user"])
            st.metric("Questions Asked", user_messages)


if __name__ == "__main__":
    main()


