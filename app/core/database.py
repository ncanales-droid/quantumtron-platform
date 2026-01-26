"""Database configuration with async SQLAlchemy."""

from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
    AsyncEngine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass


# Create async engine
engine: AsyncEngine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DB_ECHO,
    future=True,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
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
    
    This function handles connection gracefully for deployment environments.
    """
    global engine
    original_engine = engine
    
    try:
        logger.info(f"Attempting database connection to: {settings.DATABASE_URL}")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.warning(f"Database connection failed: {str(e)}")
        logger.info("Falling back to SQLite in-memory database")
        
        # Create in-memory SQLite engine
        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            echo=settings.DEBUG,
            future=True,
        )
        
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("SQLite in-memory database initialized successfully")
        except Exception as inner_e:
            logger.error(f"Failed to initialize fallback database: {str(inner_e)}")
            # Keep original engine even if it failed
            engine = original_engine
            raise


async def close_db() -> None:
    """Close database connections."""
    await engine.dispose()
