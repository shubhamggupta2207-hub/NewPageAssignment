# Chat With Your Docs – RAG Assistant

## Overview
This project implements **Option 1: Chat With Your Docs** using a Retrieval-Augmented Generation (RAG) approach.
Users can ask questions about a PDF document, and the system answers strictly based on the document content.

The focus is on clean engineering, clear RAG decisions, and simplicity rather than UI polish or over-engineering.

---

## Quick Setup

### Prerequisites
- Python 3.10+
- Ollama running locally

Pull required models:
```
ollama pull nomic-embed-text
ollama pull gemma3:4b-it-qat
```

### Install & Run
```
pip install -r requirements.txt
streamlit run rag_frontend.py
```

---

## Architecture Overview

```
PDF
 → Loader
 → Chunking
 → Embeddings
 → FAISS Vector Store
 → Retriever
 → Prompt + Context
 → LLM
 → Answer
```

**Tech Stack**
- LLM: Gemma (via Ollama)
- Embeddings: nomic-embed-text
- Vector DB: FAISS (disk-cached)
- UI: Streamlit
- Observability: LangChain tracing (LangSmith-ready)

---

## RAG & LLM Decisions

- Chunking: Recursive character splitter for balanced context and recall
- Retrieval: Similarity search (top-k) for predictable and debuggable behavior
- Prompting: Context-only answers with explicit “I don’t know” fallback
- Guardrails: Strict grounding via prompt and limited retrieved context
- Caching: FAISS index persisted using a deterministic PDF fingerprint

The goal was clarity and reliability over complexity.

---

## Key Engineering Decisions

- Local-first setup using Ollama to avoid external dependencies
- FAISS chosen for lightweight and fast local retrieval
- Functional RAG pipeline for readability
- Minimal UI to keep focus on backend correctness

---

## Engineering Standards

- Clear separation between frontend and backend
- Deterministic indexing and retrieval
- Readable, modular code
- Built-in observability hooks

Intentionally skipped (documented):
- Streaming responsess
- Advanced retrieval strategies

---

## Use of AI Tools

AI tools were used for boilerplate generation and refactoring assistance.
They were not used for architectural decisions, RAG strategy, or prompt guardrails.
All generated code was reviewed and adapted.

---

## What I’d Do Differently With More Time

- Refactor backend into a class-based architecture
- Persist chat history using Streamlit session state
- Add evaluation metrics for retrieval and answer quality
- Experiment with MMR and metadata-based retrieval
- Containerize and add production-grade logging and monitoring

---

## Productionization

To deploy on AWS / GCP / Azure:
- Dockerize services
- Use a managed vector database
- Add async ingestion workers
- Centralized logging, metrics, and alerts
- Authentication and rate limiting

---

## Final Note
This project prioritizes sound engineering judgment and a solid RAG foundation over feature completeness and is designed to be easily extensible.
