"""SQLAlchemy models and engine helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    event,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class PostRecord(Base):
    __tablename__ = "posts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    content_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    post_urn: Mapped[str | None] = mapped_column(String(255), unique=True, nullable=True)
    post_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    author: Mapped[str | None] = mapped_column(String(255), nullable=True)
    posted_at_text: Mapped[str | None] = mapped_column(String(100), nullable=True)
    raw_text: Mapped[str] = mapped_column(Text)
    image_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_interview_related: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    extraction_status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    extraction_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    qa_pairs: Mapped[list[QAPairRecord]] = relationship(back_populates="post")


class QAPairRecord(Base):
    __tablename__ = "qa_pairs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    post_id: Mapped[int] = mapped_column(ForeignKey("posts.id"), index=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_answered: Mapped[bool] = mapped_column(Boolean, default=False)
    category: Mapped[str] = mapped_column(String(64), index=True)
    question_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    pair_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    source_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    post: Mapped[PostRecord] = relationship(back_populates="qa_pairs")


@event.listens_for(Engine, "connect")
def _set_sqlite_pragma(dbapi_connection: Any, _connection_record: Any) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def make_engine(database_path: Path | str):
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        f"sqlite:///{path}",
        echo=False,
        future=True,
    )
    return engine


def init_db(engine) -> None:
    Base.metadata.create_all(engine)
    _ensure_sqlite_columns(engine)


def _ensure_sqlite_columns(engine) -> None:
    with engine.begin() as conn:
        rows = conn.execute(text("PRAGMA table_info(posts)")).fetchall()
        names = {row[1] for row in rows}
        if "image_text" not in names:
            conn.execute(text("ALTER TABLE posts ADD COLUMN image_text TEXT"))


def make_session_factory(engine) -> sessionmaker:
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
