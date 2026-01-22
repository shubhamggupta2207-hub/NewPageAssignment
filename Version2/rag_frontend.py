import streamlit as st
from rag_backend import PDFIngestor, VectorStoreManager, RAGPipeline


class RAGChatUI:
    """
    Streamlit-based chat UI for RAG application.
    """

    def __init__(self):
        """
        Initializes UI state.
        """
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

    def render_chat(self):
        """
        Renders previous chat messages.
        """
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    def run(self, rag_pipeline: RAGPipeline):
        """
        Runs the Streamlit chat loop.

        Args:
            rag_pipeline: Initialized RAGPipeline instance.
        """
        st.title("📄 RAG PDF Chatbot")

        self.render_chat()

        user_input = st.chat_input("Ask a question about the documents...")
        if user_input:
            st.session_state.chat_history.append(
                {"role": "user", "content": user_input}
            )

            with st.chat_message("assistant"):
                answer = rag_pipeline.answer(user_input)
                st.write(answer)

            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer}
            )
