import json
from typing import List, Optional
import uuid
from app.core.database import get_db
from app.models.document import DocumentChunkModel, DocumentModel
from app.services.genai_service import GenAIService
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()


class SearchRequest(BaseModel):
  query: str
  document_id: Optional[uuid.UUID] = None
  top_k: int = 5


class AskRequest(BaseModel):
  question: str
  document_id: Optional[uuid.UUID] = None
  top_k: int = 5


class QuizRequest(BaseModel):
  topic_or_query: str
  document_id: Optional[uuid.UUID] = None
  num_questions: int = 5
  top_k: int = 6


class FlashcardRequest(BaseModel):
  topic_or_query: str
  document_id: Optional[uuid.UUID] = None
  count: int = 6
  top_k: int = 6


@router.post("/search")
async def semantic_search(
    request: SearchRequest, db: AsyncSession = Depends(get_db)
):
  if not request.query.strip():
    raise HTTPException(status_code=400, detail="Query cannot be empty.")

  query_vector = GenAIService.generate_embedding(request.query)

  stmt = (
      select(DocumentChunkModel, DocumentModel.filename)
      .join(DocumentModel, DocumentChunkModel.document_id == DocumentModel.id)
      .where(DocumentChunkModel.embedding.isnot(None))
  )

  if request.document_id:
    stmt = stmt.where(DocumentChunkModel.document_id == request.document_id)

  result = await db.execute(stmt)
  records = result.all()

  if not records:
    return {"query": request.query, "results": []}

  scored_chunks = []
  for chunk, filename in records:
    if chunk.embedding:
      similarity = GenAIService.cosine_similarity(query_vector, chunk.embedding)
      scored_chunks.append({
          "chunk_id": str(chunk.id),
          "document_id": str(chunk.document_id),
          "filename": filename,
          "page_number": chunk.page_number,
          "section_title": chunk.section_title,
          "content": chunk.content,
          "similarity_score": round(similarity, 4),
      })

  scored_chunks.sort(key=lambda x: x["similarity_score"], reverse=True)
  return {
      "query": request.query,
      "results": scored_chunks[: request.top_k],
  }


@router.post("/ask")
async def ask_question(request: AskRequest, db: AsyncSession = Depends(get_db)):
  if not request.question.strip():
    raise HTTPException(status_code=400, detail="Question cannot be empty.")

  query_vector = GenAIService.generate_embedding(request.question)

  stmt = (
      select(DocumentChunkModel, DocumentModel.filename)
      .join(DocumentModel, DocumentChunkModel.document_id == DocumentModel.id)
      .where(DocumentChunkModel.embedding.isnot(None))
  )

  if request.document_id:
    stmt = stmt.where(DocumentChunkModel.document_id == request.document_id)

  result = await db.execute(stmt)
  records = result.all()

  if not records:
    return {
        "question": request.question,
        "answer": "No indexed notes with embeddings found.",
        "citations": [],
    }

  scored_chunks = []
  for chunk, filename in records:
    if chunk.embedding:
      sim = GenAIService.cosine_similarity(query_vector, chunk.embedding)
      scored_chunks.append((sim, chunk, filename))

  scored_chunks.sort(key=lambda x: x[0], reverse=True)
  top_matches = scored_chunks[: request.top_k]

  context_blocks = []
  citations = []
  for sim, chunk, filename in top_matches:
    context_blocks.append(
        f"[File: {filename} | Page {chunk.page_number}]:\n{chunk.content}"
    )
    citations.append({
        "filename": filename,
        "page_number": chunk.page_number,
        "similarity": round(sim, 4),
    })

  full_context = "\n\n---\n\n".join(context_blocks)
  ai_answer = GenAIService.answer_question(
      question=request.question, context=full_context
  )

  return {
      "question": request.question,
      "answer": ai_answer,
      "citations": citations,
  }


@router.post("/generate-quiz")
async def generate_quiz_endpoint(
    request: QuizRequest, db: AsyncSession = Depends(get_db)
):
  query_vector = GenAIService.generate_embedding(request.topic_or_query)

  stmt = (
      select(DocumentChunkModel, DocumentModel.filename)
      .join(DocumentModel, DocumentChunkModel.document_id == DocumentModel.id)
      .where(DocumentChunkModel.embedding.isnot(None))
  )

  if request.document_id:
    stmt = stmt.where(DocumentChunkModel.document_id == request.document_id)

  result = await db.execute(stmt)
  records = result.all()

  if not records:
    raise HTTPException(status_code=404, detail="No study notes available.")

  scored_chunks = []
  for chunk, _ in records:
    if chunk.embedding:
      sim = GenAIService.cosine_similarity(query_vector, chunk.embedding)
      scored_chunks.append((sim, chunk))

  scored_chunks.sort(key=lambda x: x[0], reverse=True)
  top_matches = scored_chunks[: request.top_k]

  context = "\n\n".join(
      [f"Page {c.page_number}:\n{c.content}" for _, c in top_matches]
  )

  quiz_json_raw = GenAIService.generate_quiz(
      context=context, num_questions=request.num_questions
  )

  try:
    quiz_data = json.loads(quiz_json_raw)
  except Exception:
    quiz_data = quiz_json_raw

  return {
      "topic": request.topic_or_query,
      "total_questions": (
          len(quiz_data)
          if isinstance(quiz_data, list)
          else request.num_questions
      ),
      "quiz": quiz_data,
  }


@router.post("/generate-flashcards")
async def generate_flashcards_endpoint(
    request: FlashcardRequest, db: AsyncSession = Depends(get_db)
):
  query_vector = GenAIService.generate_embedding(request.topic_or_query)

  stmt = (
      select(DocumentChunkModel, DocumentModel.filename)
      .join(DocumentModel, DocumentChunkModel.document_id == DocumentModel.id)
      .where(DocumentChunkModel.embedding.isnot(None))
  )

  if request.document_id:
    stmt = stmt.where(DocumentChunkModel.document_id == request.document_id)

  result = await db.execute(stmt)
  records = result.all()

  if not records:
    raise HTTPException(status_code=404, detail="No study notes available.")

  scored_chunks = []
  for chunk, _ in records:
    if chunk.embedding:
      sim = GenAIService.cosine_similarity(query_vector, chunk.embedding)
      scored_chunks.append((sim, chunk))

  scored_chunks.sort(key=lambda x: x[0], reverse=True)
  top_matches = scored_chunks[: request.top_k]

  context = "\n\n".join(
      [f"Page {c.page_number}:\n{c.content}" for _, c in top_matches]
  )

  flashcards_raw = GenAIService.generate_flashcards(
      context=context, count=request.count
  )

  try:
    flashcards_data = json.loads(flashcards_raw)
  except Exception:
    flashcards_data = flashcards_raw

  return {
      "topic": request.topic_or_query,
      "total_cards": (
          len(flashcards_data)
          if isinstance(flashcards_data, list)
          else request.count
      ),
      "flashcards": flashcards_data,
  }