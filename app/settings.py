"""Typed application settings loaded from environment variables.

No credentials are hardcoded. All configuration flows through this module so that
the rest of the codebase has a single, typed entry point for environment-derived
values.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4o-mini", alias="OPENAI_MODEL")
    openai_base_url: str = Field("https://api.openai.com/v1", alias="OPENAI_BASE_URL")

    # Postgres
    postgres_user: str = Field("mao", alias="POSTGRES_USER")
    postgres_password: str = Field(..., alias="POSTGRES_PASSWORD")
    postgres_db: str = Field("mao", alias="POSTGRES_DB")
    postgres_host: str = Field("postgres", alias="POSTGRES_HOST")
    postgres_port: int = Field(5432, alias="POSTGRES_PORT")

    # Redis
    redis_url: str = Field("redis://redis:6379/0", alias="REDIS_URL")

    # Vector store
    chroma_persist_dir: str = Field("/data/chroma", alias="CHROMA_PERSIST_DIR")

    # Context budgets
    default_agent_budget: int = Field(3000, alias="DEFAULT_AGENT_BUDGET")
    orchestrator_budget: int = Field(4000, alias="ORCHESTRATOR_BUDGET")
    compression_threshold: float = Field(0.85, alias="COMPRESSION_THRESHOLD")

    # RAG corpus
    arxiv_corpus_path: str = Field("/data/arxiv_corpus", alias="ARXIV_CORPUS_PATH")
    arxiv_category: str = Field("cs.CL", alias="ARXIV_CATEGORY")
    arxiv_max_papers: int = Field(50, alias="ARXIV_MAX_PAPERS")

    # Logging
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    log_json: bool = Field(True, alias="LOG_JSON")

    # API
    api_host: str = Field("0.0.0.0", alias="API_HOST")
    api_port: int = Field(8000, alias="API_PORT")

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
