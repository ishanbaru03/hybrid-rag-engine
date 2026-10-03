from app.api.v1.endpoints.documents import router as documents_router
from app.api.v1.endpoints.query import router as query_router
from app.core.config import settings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Documents management router
app.include_router(
    documents_router,
    prefix=f"{settings.API_V1_STR}/documents",
    tags=["documents"],
)

# Semantic search and RAG Q&A router
app.include_router(
    query_router,
    prefix=f"{settings.API_V1_STR}/query",
    tags=["query & rag"],
)


@app.get("/health")
def health_check():
    return {"status": "healthy", "project": settings.PROJECT_NAME}