from pydantic import BaseModel


class DocumentPage(BaseModel):
  page_number: int
  char_count: int
  text: str


class DocumentParseResult(BaseModel):
  document_id: str
  filename: str
  total_pages: int
  total_chars: int
  pages: list[DocumentPage]


class DocumentUploadResponse(BaseModel):
  document_id: str
  filename: str
  total_pages: int
  message: str