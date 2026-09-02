"""Normalize LinkedIn post text and produce stable hashes for deduplication."""

from __future__ import annotations

import hashlib
import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")
_SEE_MORE_RE = re.compile(r"(?i)\bsee more\b|\bshow more\b|\b…more\b|\b\.\.\.more\b")
_PUNCT_RE = re.compile(r"[^\w\s]")


def normalize_text(text: str) -> str:
    """Collapse whitespace and strip LinkedIn UI artifacts without rewriting content."""
    value = unicodedata.normalize("NFKC", text or "")
    value = value.replace("\u00a0", " ").replace("\u200b", "")
    value = _SEE_MORE_RE.sub("", value)
    value = _WHITESPACE_RE.sub(" ", value).strip()
    return value


def normalize_for_hash(text: str) -> str:
    value = normalize_text(text).lower()
    value = _PUNCT_RE.sub(" ", value)
    return _WHITESPACE_RE.sub(" ", value).strip()


def content_hash(text: str) -> str:
    payload = normalize_for_hash(text).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def question_hash(question: str) -> str:
    return content_hash(question)


def pair_hash(question: str, answer: str | None) -> str:
    combined = f"{normalize_for_hash(question)}\n{normalize_for_hash(answer or '')}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()
