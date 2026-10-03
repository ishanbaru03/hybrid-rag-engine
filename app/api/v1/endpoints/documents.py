from pathlib import Path
import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings
from app.core.database import get_db
from app.models.document import DocumentChunkModel, DocumentModel
from app.services.chunking_service import ChunkingService
from app.services.pdf_service import PDFService
from app.services.genai_service import GenAIService

router = APIRouter()


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_pdf(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported."
        )

    doc_id = uuid.uuid4()
    saved_filename = f"{doc_id}_{file.filename}"
    file_path = Path(settings.UPLOAD_DIR) / saved_filename

    # Ensure upload directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    try:
        parsed_result = PDFService.extract_text_by_pages(
            file_path=file_path,
            document_id=str(doc_id),
            filename=file.filename
        )

        raw_chunks = ChunkingService.chunk_document(parsed_result.pages)
        if not raw_chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No readable text found in the PDF. Scanned images require OCR."
            )

        new_doc = DocumentModel(
            id=doc_id,
            filename=file.filename,
            file_path=str(file_path),
            total_pages=parsed_result.total_pages,
            status="processing",
        )
        db.add(new_doc)
        await db.flush()

        # Let GenAIService handle rate-safe batching (batches of 25 with 429 backoff)
        texts_to_embed = [chunk.content for chunk in raw_chunks]
        embeddings = GenAIService.generate_embeddings_batch(texts_to_embed)

        chunk_models = [
            DocumentChunkModel(
                id=uuid.uuid4(),
                document_id=doc_id,
                chunk_index=chunk.chunk_index,
                page_number=chunk.page_number,
                section_title=chunk.section_title,
                content=chunk.content,
                embedding=embeddings[idx] if idx < len(embeddings) else None,
            )
            for idx, chunk in enumerate(raw_chunks)
        ]
        db.add_all(chunk_models)

        new_doc.status = "indexed"
        await db.commit()

        return {
            "document_id": str(doc_id),
            "filename": file.filename,
            "total_pages": parsed_result.total_pages,
            "total_chunks": len(raw_chunks),
            "status": "indexed",
            "message": "File uploaded, chunked, and embedded successfully.",
        }

    except HTTPException:
        # Cleanup uploaded file on validation errors
        if file_path.exists():
            file_path.unlink()
        await db.rollback()
        raise

    except Exception as e:
        # Ensure database rollback and delete partial file on unexpected errors/429s
        await db.rollback()
        if file_path.exists():
            file_path.unlink()
        
        error_text = str(e)
        if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Google API quota temporarily exceeded. Please wait 45 seconds and try again."
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload processing failed: {error_text}"
        )


@router.get("", response_model=List[dict])
async def list_documents(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(DocumentModel).order_by(DocumentModel.created_at.desc()))
    docs = result.scalars().all()
    return [
        {
            "id": str(doc.id),
            "filename": doc.filename,
            "total_pages": doc.total_pages,
            "status": doc.status,
            "created_at": doc.created_at.isoformat() if doc.created_at else None,
        }
        for doc in docs
    ]


@router.get("/{document_id}/chunks")
async def get_document_chunks(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
):
    doc_res = await db.execute(select(DocumentModel).where(DocumentModel.id == document_id))
    doc = doc_res.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    chunks_res = await db.execute(
        select(DocumentChunkModel)
        .where(DocumentChunkModel.document_id == document_id)
        .order_by(DocumentChunkModel.chunk_index.asc())
    )
    chunks = chunks_res.scalars().all()

    return {
        "document_id": str(document_id),
        "filename": doc.filename,
        "total_chunks": len(chunks),
        "chunks": [
            {
                "id": str(c.id),
                "chunk_index": c.chunk_index,
                "page_number": c.page_number,
                "section_title": c.section_title,
                "content": c.content,
                "has_embedding": c.embedding is not None,
            }
            for c in chunks
        ],
    }


@router.delete("/{document_id}", status_code=status.HTTP_200_OK)
async def delete_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
):
    doc_res = await db.execute(select(DocumentModel).where(DocumentModel.id == document_id))
    doc = doc_res.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    file_path = Path(doc.file_path)
    if file_path.exists():
        try:
            file_path.unlink()
        except OSError:
            pass

    # SQLAlchemy cascade will automatically delete associated document_chunks
    await db.delete(doc)
    await db.commit()

    return {"message": f"Document {doc.filename} and all associated chunks deleted successfully."}