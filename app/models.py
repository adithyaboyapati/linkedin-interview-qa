"""Pydantic models for posts, extraction results, and PDF rendering."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class Category(StrEnum):
    PYTHON = "Python"
    SQL = "SQL"
    DSA = "DSA"
    SYSTEM_DESIGN = "System Design"
    ML = "ML"
    GENAI = "GenAI"
    RAG = "RAG"
    AWS = "AWS"
    JAVA = "Java"
    JAVASCRIPT = "JavaScript"
    DEVOPS = "DevOps"
    BEHAVIORAL = "Behavioral"
    OTHER = "Other"


CATEGORY_ORDER: tuple[str, ...] = tuple(c.value for c in Category)

_CATEGORY_ALIASES: dict[str, str] = {
    "python": Category.PYTHON,
    "py": Category.PYTHON,
    "sql": Category.SQL,
    "mysql": Category.SQL,
    "postgres": Category.SQL,
    "postgresql": Category.SQL,
    "dsa": Category.DSA,
    "data structures": Category.DSA,
    "algorithms": Category.DSA,
    "coding": Category.DSA,
    "system design": Category.SYSTEM_DESIGN,
    "systemdesign": Category.SYSTEM_DESIGN,
    "distributed systems": Category.SYSTEM_DESIGN,
    "ml": Category.ML,
    "machine learning": Category.ML,
    "machinelearning": Category.ML,
    "genai": Category.GENAI,
    "gen ai": Category.GENAI,
    "generative ai": Category.GENAI,
    "llm": Category.GENAI,
    "rag": Category.RAG,
    "retrieval": Category.RAG,
    "aws": Category.AWS,
    "amazon web services": Category.AWS,
    "java": Category.JAVA,
    "javascript": Category.JAVASCRIPT,
    "js": Category.JAVASCRIPT,
    "typescript": Category.JAVASCRIPT,
    "devops": Category.DEVOPS,
    "kubernetes": Category.DEVOPS,
    "docker": Category.DEVOPS,
    "behavioral": Category.BEHAVIORAL,
    "hr": Category.BEHAVIORAL,
    "other": Category.OTHER,
}


def normalize_category(value: str | None) -> str:
    if not value:
        return Category.OTHER
    key = " ".join(value.strip().lower().replace("_", " ").replace("-", " ").split())
    if key in _CATEGORY_ALIASES:
        return _CATEGORY_ALIASES[key]
    for alias, category in _CATEGORY_ALIASES.items():
        if alias in key:
            return category
    return Category.OTHER


class CollectedPost(BaseModel):
    """A single accessible LinkedIn post captured from the activity feed."""

    post_urn: str | None = None
    post_url: str | None = None
    author: str | None = None
    posted_at_text: str | None = None
    raw_text: str
    content_hash: str
    image_text: str = ""

    @field_validator("raw_text")
    @classmethod
    def _text_must_exist(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Post text is empty.")
        return value.strip()


class QAPairDraft(BaseModel):
    """One Q&A pair returned by the LLM. Answers must come from the post."""

    question: str
    answer: str | None = None
    answered: bool = False
    category: str = Category.OTHER

    @field_validator("question")
    @classmethod
    def _question_required(cls, value: str) -> str:
        cleaned = (value or "").strip()
        if not cleaned:
            raise ValueError("Question is empty.")
        return cleaned

    @field_validator("answer")
    @classmethod
    def _empty_answer_to_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @field_validator("category")
    @classmethod
    def _normalize_category(cls, value: str) -> str:
        return normalize_category(value)


class PostClassification(BaseModel):
    """Structured LLM output for interview-related classification."""

    is_interview_related: bool
    reason: str = ""


class ExtractionResult(BaseModel):
    """Structured LLM output for a single post."""

    is_interview_related: bool
    qa_pairs: list[QAPairDraft] = Field(default_factory=list)
    reason: str = ""

    def answered_pairs(self) -> list[QAPairDraft]:
        pairs: list[QAPairDraft] = []
        for pair in self.qa_pairs:
            has_answer = bool(pair.answer) and pair.answered
            if has_answer:
                pairs.append(
                    pair.model_copy(update={"answered": True, "answer": pair.answer})
                )
        return pairs


class StoredQA(BaseModel):
    """A persisted, answered Q&A pair ready for the PDF."""

    id: int
    question: str
    answer: str
    category: str
    source_url: str | None = None
    author: str | None = None
    posted_at_text: str | None = None


class CollectorStats(BaseModel):
    posts: int = 0
    interview_related: int = 0
    not_interview_related: int = 0
    pending_extraction: int = 0
    extraction_failed: int = 0
    qa_pairs: int = 0
    answered_qa_pairs: int = 0
    unanswered_qa_pairs: int = 0
    by_category: dict[str, int] = Field(default_factory=dict)


class PdfDocumentData(BaseModel):
    creator: str
    generated_at: datetime
    categories: dict[str, list[StoredQA]]
