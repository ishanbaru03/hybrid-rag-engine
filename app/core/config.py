from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
  PROJECT_NAME: str = "PrepIQ Study Engine"
  API_V1_STR: str = "/api/v1"
  UPLOAD_DIR: str = "./uploads"
  MAX_UPLOAD_SIZE_MB: int = 25
  DATABASE_URL: str

  # AI Keys
  GEMINI_API_KEY: Optional[str] = None
  GROQ_API_KEY: Optional[str] = None

  model_config = SettingsConfigDict(
      env_file=".env", env_file_encoding="utf-8", extra="allow"
  )


settings = Settings()