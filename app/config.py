from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_", env_file=".env", extra="ignore"
    )

    data_dir: Path = Path("./data")
    collection: str = "documents"
    chunk_size: int = Field(512, gt=0)
    chunk_overlap: int = Field(64, ge=0)
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = Field(5, gt=0)
        # ---- LLM (Grok) ----
    grok_api_key: str | None = None
    grok_base_url: str = "https://api.x.ai/v1"
    grok_model: str = "grok-2-latest"
    llm_max_tokens: int = Field(512, gt=0)
    llm_temperature: float = 0.0

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    def ensure_dirs(self) -> None:
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s