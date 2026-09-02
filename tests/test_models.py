"""Pydantic model tests."""

from __future__ import annotations

import pytest
from app.models import (
    CATEGORY_ORDER,
    Category,
    CollectedPost,
    ExtractionResult,
    QAPairDraft,
    normalize_category,
)
from pydantic import ValidationError


def test_collected_post_rejects_empty_text() -> None:
    with pytest.raises(ValidationError):
        CollectedPost(raw_text="   ", content_hash="abc")


def test_qa_pair_normalizes_category_aliases() -> None:
    pair = QAPairDraft(
        question="What is RAG?",
        answer="Retrieval-Augmented Generation",
        answered=True,
        category="gen ai",
    )
    assert pair.category == Category.GENAI


def test_normalize_category_known_values() -> None:
    assert normalize_category("python") == Category.PYTHON
    assert normalize_category("System-Design") == Category.SYSTEM_DESIGN
    assert normalize_category("machine learning") == Category.ML
    assert normalize_category("unknown-topic") == Category.OTHER


def test_extraction_result_answered_pairs_excludes_unanswered() -> None:
    result = ExtractionResult(
        is_interview_related=True,
        qa_pairs=[
            QAPairDraft(question="Q1", answer="A1", answered=True, category="Python"),
            QAPairDraft(question="Q2", answer=None, answered=False, category="Python"),
            QAPairDraft(question="Q3", answer="  ", answered=True, category="SQL"),
        ],
    )
    answered = result.answered_pairs()
    assert len(answered) == 1
    assert answered[0].question == "Q1"


def test_category_order_covers_all_categories() -> None:
    assert set(CATEGORY_ORDER) == {c.value for c in Category}
