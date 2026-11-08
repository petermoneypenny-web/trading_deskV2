from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    ENVIRONMENT: str = "production"
    PORT: int = 8000
    HOST: str = "0.0.0.0"
    DATABASE_URL: str = "sqlite:///./data/trading_v2.db"
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-flash-latest"
    # Measured Gemini Flash latency on this setup: 6-11s per call (the original
    # 12s client timeout was already marginal -- one call took 12.2s). 20s leaves
    # real headroom; the *deadline* below is what protects the trading loop.
    GEMINI_HTTP_TIMEOUT: float = 20.0
    # Outer deadline for an LLM evaluation. The call now runs in a background
    # task, so this no longer blocks risk monitoring -- it can be generous.
    LLM_DEADLINE_SECONDS: float = 25.0
    NTFY_TOPIC: str = "m0ney6944"
    NTFY_SERVER_URL: str = "https://ntfy.sh"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()