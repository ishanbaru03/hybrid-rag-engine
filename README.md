# PrepIQ Study Engine Backend: 

An asynchronous, high-throughput RAG (Retrieval-Augmented Generation) backend built with **FastAPI**, **SQLAlchemy (AsyncPG)**, and **Google Gemini GenAI**. Designed to parse complex academic documents, generate semantic embeddings, and answer queries with verifiable source citations.
---
##  Architecture & Features
- **Asynchronous Pipeline**: Built on FastAPI and `asyncpg` for non-blocking I/O operations across LLM requests and database queries.
- **Document Preprocessing**: Ingestion pipeline using `PyMuPDF` with adaptive overlap chunking.
- **Semantic Retrieval**: Vector embeddings and cosine similarity search powered by Google's Gemini models.
- **Isolated Design Pattern**: Strict separation between API routers, domain services, Pydantic schemas, and SQLAlchemy ORM models.
- **Rate Throttling & Protection**: Built-in request controls and environmental secrets decoupling.
---
##  Tech Stack

- **Framework**: FastAPI (Python 3.11+)
- **Database**: PostgreSQL 16 with AsyncPG
- **ORM**: SQLAlchemy 2.0 (Async)
- **AI/LLM**: Google Gemini API (`google-genai`)
- **PDF Extraction**: PyMuPDF (`fitz`)
- **Server**: Uvicorn with uvloop
```bash
git clone [https://github.com/ishanbaru03/study-engine-backend.git](https://github.com/ishanbaru03/study-engine-backend.git)
cd study-engine-backend
