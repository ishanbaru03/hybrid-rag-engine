from dataclasses import dataclass
from typing import List
from app.schemas.document import DocumentPage

@dataclass
class ChunkData:
    chunk_index: int
    page_number: int
    content: str
    section_title: str | None = None

class ChunkingService:
    @staticmethod
    def chunk_document(
        pages: List[DocumentPage],
        chunk_size: int = 700,
        chunk_overlap: int = 100
    ) -> List[ChunkData]:
        chunks: List[ChunkData] = []
        chunk_counter = 0

        for page in pages:
            text = page.text.strip()
            if not text:
                continue

            start = 0
            text_len = len(text)

            while start < text_len:
                end = start + chunk_size
                chunk_text = text[start:end].strip()

                if chunk_text:
                    chunks.append(
                        ChunkData(
                            chunk_index=chunk_counter,
                            page_number=page.page_number,
                            content=chunk_text,
                            section_title=None
                        )
                    )
                    chunk_counter += 1

                # Advance window with overlap
                start += chunk_size - chunk_overlap

        return chunks