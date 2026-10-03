# Hybrid RAG Knowledge Engine (Backend)

An asynchronous backend service built with **FastAPI** providing high-performance document ingestion, vector retrieval, and contextual query processing for educational datasets.

## Key Capabilities
- **Sub-180ms p95 Retrieval Latency:** Optimized chunk indexing and vector database queries.
- **Hybrid Retrieval Pipeline:** Evaluates semantic similarity alongside sparse keyword matching to boost retrieval recall.
- **Strict Data Integrity:** Request & response validation enforced via Pydantic schemas.
- **Automated Test Coverage:** Includes unit and integration tests via `pytest`.

## Architecture Flow
