from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


class PDFIngestor:
    """
    Handles loading and chunking of PDF documents.
    """

    def __init__(
        self,
        pdf_dir: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Args:
            pdf_dir: Directory containing PDF files.
            chunk_size: Size of text chunks.
            chunk_overlap: Overlap between consecutive chunks.
        """
        self.pdf_dir = Path(pdf_dir)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def load_and_split(self) -> List[Document]:
        """
        Loads PDFs from disk and splits them into chunks.

        Returns:
            List of chunked LangChain Document objects.
        """
        documents: List[Document] = []

        for pdf_path in self.pdf_dir.glob("*.pdf"):
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()
            documents.extend(docs)

        return self.splitter.split_documents(documents)


class VectorStoreManager:
    """
    Manages embedding generation and vector store persistence.
    """

    def __init__(
        self,
        persist_directory: str,
        embedding_model_name: str = "intfloat/e5-small",
        collection_name: str = "pdf_chunks",
    ):
        """
        Args:
            persist_directory: Directory where Chroma DB is stored.
            embedding_model_name: HuggingFace embedding model name.
            collection_name: Chroma collection name.
        """
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name
        )

        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
        )

    def index_documents(self, documents: List[Document]) -> None:
        """
        Embeds and stores documents in Chroma.

        Args:
            documents: List of Document objects.
        """
        self.vectorstore.add_documents(documents)
        self.vectorstore.persist()

    def get_retriever(self, k: int = 4):
        """
        Returns a retriever for similarity search.

        Args:
            k: Number of documents to retrieve.

        Returns:
            Chroma retriever instance.
        """
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )


class RAGPipeline:
    """
    End-to-end Retrieval Augmented Generation pipeline.
    """

    def __init__(
        self,
        retriever,
        llm_model: str = "gemma3:4b-it-qat",
    ):
        """
        Args:
            retriever: Vector store retriever.
            llm_model: Ollama LLM model name.
        """
        self.retriever = retriever
        self.llm = ChatOllama(model=llm_model)

        self.prompt = ChatPromptTemplate.from_template(
            """
            Use the following context to answer the question.
            If the answer is not present, say you don't know.

            Context:
            {context}

            Question:
            {question}
            """
        )

    def answer(self, question: str) -> str:
        """
        Answers a user query using retrieved context.

        Args:
            question: User question.

        Returns:
            LLM generated answer.
        """
        docs = self.retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)

        messages = self.prompt.invoke(
            {"context": context, "question": question}
        )

        return self.llm.invoke(messages).content
