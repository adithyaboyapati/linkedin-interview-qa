"""Tests for text normalization and hashing."""

from __future__ import annotations

from app.extraction.normalizer import content_hash, normalize_text, pair_hash, question_hash


def test_normalize_text_collapses_whitespace_and_see_more() -> None:
    raw = "Hello   world\nSee more\n  extra"
    assert normalize_text(raw) == "Hello world extra"


def test_content_hash_is_stable_across_cosmetic_differences() -> None:
    a = "What is Python? See more"
    b = "what is python?"
    assert content_hash(a) == content_hash(b)


def test_question_hash_ignores_punctuation() -> None:
    assert question_hash("What is GIL?") == question_hash("what is gil")


def test_pair_hash_changes_when_answer_changes() -> None:
    q = "What is GIL?"
    assert pair_hash(q, "A lock") != pair_hash(q, "A different lock")
