"""Tests for LLM extraction and anti-hallucination grounding."""

from __future__ import annotations

import json

from app.extraction.qa_extractor import (
    QAExtractor,
    answer_is_grounded,
    looks_interview_related,
)
from app.models import ExtractionResult, PostClassification


POST = (
    "Interview questions I was asked:\n"
    "Q: What is the GIL?\n"
    "A: Global Interpreter Lock in CPython.\n"
    "Q: What is a primary key?\n"
)


def test_looks_interview_related() -> None:
    assert looks_interview_related(POST)
    assert looks_interview_related("Swipe through 👉 Scenario based questions asked in Razorpay")
    assert not looks_interview_related("Excited to start my new role next week!")


def test_answer_is_grounded_requires_post_text() -> None:
    assert answer_is_grounded("Global Interpreter Lock in CPython.", POST)
    assert not answer_is_grounded("A mutex that serializes bytecode execution in CPython and Jython.", POST)


def test_extractor_parses_structured_json_and_drops_hallucinations() -> None:
    payload = {
        "is_interview_related": True,
        "reason": "contains interview Q&A",
        "qa_pairs": [
            {
                "question": "What is the GIL?",
                "answer": "Global Interpreter Lock in CPython.",
                "answered": True,
                "category": "Python",
            },
            {
                "question": "What is a primary key?",
                "answer": "A unique identifier invented by the model.",
                "answered": True,
                "category": "SQL",
            },
            {
                "question": "Explain CAP theorem",
                "answer": None,
                "answered": False,
                "category": "System Design",
            },
        ],
    }

    def completer(_messages: list[dict[str, str]]) -> str:
        return json.dumps(payload)

    extractor = QAExtractor(api_key="x", base_url="http://localhost", model="test", completer=completer)
    result = extractor.extract(POST)
    assert isinstance(result, ExtractionResult)
    assert result.is_interview_related
    assert len(result.qa_pairs) == 1
    assert result.qa_pairs[0].question == "What is the GIL?"
    assert result.qa_pairs[0].answer == "Global Interpreter Lock in CPython."


def test_extractor_accepts_markdown_fenced_json() -> None:
    def completer(_messages: list[dict[str, str]]) -> str:
        return """```json
{"is_interview_related": false, "reason": "job update", "qa_pairs": []}
```"""

    extractor = QAExtractor(api_key="x", base_url="http://localhost", model="test", completer=completer)
    result = extractor.extract("Happy to announce I joined Acme.")
    assert result.is_interview_related is False
    assert result.qa_pairs == []


def test_classify_uses_structured_output() -> None:
    def completer(_messages: list[dict[str, str]]) -> str:
        return '{"is_interview_related": true, "reason": "shares interview questions"}'

    extractor = QAExtractor(api_key="x", base_url="http://localhost", model="test", completer=completer)
    result = extractor.classify(POST)
    assert isinstance(result, PostClassification)
    assert result.is_interview_related is True
