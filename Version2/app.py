from rag_backend import PDFIngestor, VectorStoreManager, RAGPipeline
from rag_frontend import RAGChatUI

# 1. Ingest PDFs
ingestor = PDFIngestor(
    pdf_dir="./backend/pdfs",
    chunk_size=1000,
    chunk_overlap=200
)
documents = ingestor.load_and_split()

# 2. Build / load vector store
vs_manager = VectorStoreManager(
    persist_directory="./chroma_store"
)
vs_manager.index_documents(documents)

retriever = vs_manager.get_retriever(k=4)

# 3. Create RAG pipeline (THIS is what you were missing)
rag_pipeline = RAGPipeline(
    retriever=retriever,
    llm_model="gemma3:4b-it-qat"
)

# 4. Start UI and inject pipeline
ui = RAGChatUI()
ui.run(rag_pipeline)