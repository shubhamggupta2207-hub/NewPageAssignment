import streamlit as st
from uuid import uuid4

from langchain_core.messages import HumanMessage, AIMessage
from rag_backend import RAGLangGraphApp


class RAGChatUI:
    """
    Streamlit UI for persistent LangGraph-based RAG chatbot.
    """

    def __init__(self, rag_app: RAGLangGraphApp):
        self.rag_app = rag_app

        if "thread_id" not in st.session_state:
            st.session_state.thread_id = str(uuid4())

        if "messages" not in st.session_state:
            st.session_state.messages = []

    def render_messages(self):
        for msg in st.session_state.messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            with st.chat_message(role):
                st.write(msg.content)

    def run(self):
        st.title("📄 Persistent RAG Chatbot")

        st.caption(f"Conversation ID: `{st.session_state.thread_id}`")

        self.render_messages()

        user_input = st.chat_input("Ask a question about your PDFs...")
        if user_input:
            user_msg = HumanMessage(content=user_input)
            st.session_state.messages.append(user_msg)

            updated_messages = self.rag_app.invoke(
                st.session_state.messages,
                thread_id=st.session_state.thread_id,
            )

            st.session_state.messages = updated_messages

            with st.chat_message("assistant"):
                st.write(updated_messages[-1].content)