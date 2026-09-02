"""Shared LangGraph state for the interview Q&A workflow."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class CollectorState(TypedDict, total=False):
    """State passed between graph nodes.

    SQLite records and PDF bytes are not stored here. Nodes orchestrate those
    side effects through the existing repository and PDF modules.
    """

    posts: list[dict[str, Any]]
    interview_posts: list[dict[str, Any]]
    qa_pairs: list[dict[str, Any]]
    validation_results: list[dict[str, Any]]
    extract_attempts: int
    skipped_posts: list[str]
    pdf_path: str
    skip_scrape: bool
    require_login_interrupt: bool
    logs: Annotated[list[str], operator.add]
