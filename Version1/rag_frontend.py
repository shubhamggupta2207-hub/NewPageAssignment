import streamlit as st
import rag_backend as rag

PDF_PATH = "ISLR.pdf"
st.title("RAG PDF Question Answering")
user_input = st.chat_input("Type Here")

if user_input:
    answer = rag.setup_pipeline_and_query(PDF_PATH, user_input)
    with st.chat_message("user"):
        st.text(user_input)

    with st.chat_message("assistant"):
        st.text(answer)