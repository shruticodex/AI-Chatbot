"""Application configuration loaded from environment variables."""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    nvidia_api_key: str
    pinecone_api_key: str
    pinecone_index_name: str = "agentic-ai-ebook"

    huggingface_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    nvidia_chat_model: str = "openai/gpt-oss-20b"

    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5

    embedding_dimension: int = 384

@lru_cache
def get_settings() -> Settings:
    return Settings()