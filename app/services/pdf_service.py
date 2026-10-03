from pathlib import Path
import pymupdf  # Modern import
from app.schemas.document import DocumentPage, DocumentParseResult


class PDFService:

  @staticmethod
  def extract_text_by_pages(
      file_path: Path, document_id: str, filename: str
  ) -> DocumentParseResult:
    doc = pymupdf.open(str(file_path))
    pages_data: list[DocumentPage] = []
    total_chars = 0

    for page_idx in range(len(doc)):
      page = doc.load_page(page_idx)
      text = page.get_text("text").strip()
      char_count = len(text)
      total_chars += char_count

      pages_data.append(
          DocumentPage(
              page_number=page_idx + 1,
              char_count=char_count,
              text=text,
          )
      )

    doc.close()

    return DocumentParseResult(
        document_id=document_id,
        filename=filename,
        total_pages=len(pages_data),
        total_chars=total_chars,
        pages=pages_data,
    )