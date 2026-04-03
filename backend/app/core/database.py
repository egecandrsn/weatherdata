"""Database engine and session management."""

from typing import Optional

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


# Lazy initialization — avoids import-time dependency on asyncpg
_engine = None
_async_session = None


def _init_engine():
    global _engine, _async_session
    if _engine is None:
        from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
        from app.core.config import settings
        _engine = create_async_engine(settings.database_url, echo=settings.debug)
        _async_session = async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)


async def get_db():
    _init_engine()
    async with _async_session() as session:
        yield session
