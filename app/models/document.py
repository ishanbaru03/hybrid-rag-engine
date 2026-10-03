import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from app.core.database import Base


class DocumentModel(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(255), nullable=False)
    file_path = Column(Text, nullable=False)
    total_pages = Column(Integer, default=0)
    status = Column(String(50), default="uploaded")
    created_at = Column(DateTime, default=datetime.utcnow)


class DocumentChunkModel(Base):
    __tablename__ = "document_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False
    )
    chunk_index = Column(Integer, nullable=False)
    page_number = Column(Integer, nullable=False)
    section_title = Column(Text, nullable=True)
    content = Column(Text, nullable=False)
    embedding = Column(JSONB, nullable=True)  # <-- This line must be present
    created_at = Column(DateTime, default=datetime.utcnow)