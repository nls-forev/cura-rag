from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


class LLMProvider(str, Enum):
    deepseek = "deepseek"
    openai = "openai"
    anthropic = "anthropic"


class ChunkingStrategy(str, Enum):
    fixed = "fixed"
    recursive = "recursive"
    semantic = "semantic"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: LLMProvider = LLMProvider.deepseek
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.1

    deepseek_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    embedding_dim: int = 384

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "curarag"

    chunking_strategy: ChunkingStrategy = ChunkingStrategy.recursive
    chunk_size: int = 900
    chunk_overlap: int = 150
    dense_top_k: int = 30
    sparse_top_k: int = 30
    rrf_k: int = 60
    rerank_top_k: int = 5

    # Below this fused retrieval strength the system refuses to answer rather than
    # risk a fabricated dose/contraindication. Tuned on the golden eval set: 0.35
    # keeps the unanswerable cases abstaining without suppressing valid answers.
    retrieval_confidence_threshold: float = 0.35

    # A near-duplicate above this cosine similarity is dropped before upsert.
    dedupe_threshold: float = 0.95

    openfda_base_url: str = "https://api.fda.gov/drug/label.json"
    medlineplus_base_url: str = "https://wsearch.nlm.nih.gov/ws/query"

    request_timeout: float = Field(default=30.0)

    @property
    def active_api_key(self) -> str:
        return {
            LLMProvider.deepseek: self.deepseek_api_key,
            LLMProvider.openai: self.openai_api_key,
            LLMProvider.anthropic: self.anthropic_api_key,
        }[self.llm_provider]


@lru_cache
def get_settings() -> Settings:
    return Settings()
