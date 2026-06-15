from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    llm_provider: Literal["anthropic", "openrouter"] = "anthropic"
    anthropic_api_key: str = ""
    openrouter_api_key: str = ""
    openrouter_model: str = "anthropic/claude-3-haiku"
    claude_scoring_model: str = "claude-3-5-sonnet-20241022"
    claude_fast_model: str = "claude-3-haiku-20240307"

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"

    # Data
    dataset_path: str = "data/candidates.csv"

    # Redis
    redis_url: str = "redis://localhost:6379"
    redis_enabled: bool = False

    # Pipeline knobs
    retrieval_top_k: int = 50
    advance_percentile_normal: float = 0.60
    advance_percentile_high: float = 0.60
    advance_percentile_extra_high: float = 0.60
    rescue_floor_normal: float = 0.80
    rescue_floor_high: float = 0.90
    rescue_floor_extra_high: float = 0.95
    finalist_count_normal: int = 15
    finalist_count_extra_high: int = 25

    # LLM
    llm_temperature: float = 0.1

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:3000"


settings = Settings()
