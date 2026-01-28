import os

# Railway-safe DB location (container writable path)
if os.getenv("RAILWAY_ENVIRONMENT"):
    os.environ["DATABASE_URL"] = os.getenv("DATABASE_URL", "sqlite:////app/data/quantumtron.db")

"""Application configuration using pydantic-settings."""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=(".env.local", ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    APP_NAME: str = "QuantumTron Intelligence Platform"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = Field(default=False, description="Debug mode")
    API_V1_PREFIX: str = "/api/v1"

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./quantumtron.db")
    DB_ECHO: bool = Field(default=False, description="Echo SQL queries")

    # Security
    SECRET_KEY: str = Field(
        default="your-secret-key-change-in-production",
        description="Secret key for JWT token signing"
    )
    ALGORITHM: str = Field(default="HS256", description="JWT algorithm")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30,
        description="Access token expiration time in minutes"
    )

    # Redis
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )

    # CORS
    CORS_ORIGINS: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="Allowed CORS origins"
    )

    # Logging
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )

    # Statistical Analysis
    STATISTICAL_SEED: Optional[int] = Field(
        default=42,
        description="Random seed for reproducible statistical analysis"
    )

    # DeepSeek AI Configuration
    DEEPSEEK_API_KEY: str = Field(
        default="",
        description="API Key for DeepSeek AI service"
    )
    DEEPSEEK_MODEL: str = Field(
        default="deepseek-chat",
        description="DeepSeek model to use"
    )
    DEEPSEEK_BASE_URL: str = Field(
        default="https://api.deepseek.com/v1",
        description="DeepSeek API base URL"
    )
    DEEPSEEK_MAX_TOKENS: int = Field(
        default=2000,
        description="Maximum tokens for DeepSeek responses"
    )

settings = Settings()

