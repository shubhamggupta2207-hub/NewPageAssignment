from rag_backend import RAGVectorStore, RAGLangGraphApp
from rag_frontend import RAGChatUI


def main():
    """
    Main application bootstrap.
    """
    PDF_DIR = "./backend/pdfs"
    CHROMA_DIR = "./chroma_store"
    SQLITE_DB = "./rag_state.db"

    vector_store = RAGVectorStore(
        pdf_dir=PDF_DIR,
        persist_dir=CHROMA_DIR,
        embedding_model="intfloat/e5-small",
        chunk_size=1000,
        chunk_overlap=200,
    )

    vector_store.ingest()

    retriever = vector_store.get_retriever(k=4)

    rag_app = RAGLangGraphApp(
        retriever=retriever,
        db_path=SQLITE_DB,
    )

    ui = RAGChatUI(rag_app)
    ui.run()


if __name__ == "__main__":
    main()