"""Database configuration with async SQLAlchemy."""

import logging
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
    AsyncEngine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass


# Create async engine - use SQLite in-memory for Railway
# This gets overridden in init_db() but we need a default
engine: AsyncEngine = create_async_engine(
    "sqlite+aiosqlite:///:memory:",
    echo=settings.DB_ECHO,
    future=True,
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for getting database session.

    Yields:
        AsyncSession: Database session
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db() -> None:
    """
    Initialize database connection and create tables.
    
    For Railway deployment, always use SQLite in-memory.
    """
    global engine
    
    try:
        # Always use SQLite in-memory for Railway deployment
        logger.info("Using SQLite in-memory database for Railway deployment")
        
        # Recreate engine to ensure fresh connection
        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            echo=settings.DB_ECHO,
            future=True,
        )
        
        # Update session factory with new engine
        global AsyncSessionLocal
        AsyncSessionLocal = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
        
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("SQLite in-memory database initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        raise


async def close_db() -> None:
    """Close database connections."""
    await engine.dispose()