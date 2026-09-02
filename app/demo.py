"""Offline sample posts so the pipeline can be tested without LinkedIn or an LLM."""

from __future__ import annotations

from app.extraction.normalizer import content_hash
from app.models import CollectedPost, ExtractionResult, PostClassification, QAPairDraft

INTERVIEW_TEXT = """Interview questions I was asked yesterday:

Q: What is the GIL?
A: Global Interpreter Lock in CPython.

Q: What is a primary key?
A: A column that uniquely identifies a row.
"""

JOB_UPDATE_TEXT = "Excited to start my new role next week at Acme!"


def is_sample_post(post) -> bool:
    """True for offline demo fixtures that must never appear in the LinkedIn PDF."""
    urn = (getattr(post, "post_urn", None) or "").lower()
    url = (getattr(post, "post_url", None) or "").lower()
    author = (getattr(post, "author", None) or "").strip()
    return (
        "demo-interview" in urn
        or "demo-job" in urn
        or "demo-interview" in url
        or "demo-job" in url
        or author == "Demo Author"
    )


def sample_posts() -> list[CollectedPost]:
    posts = [
        ("urn:li:activity:demo-interview", INTERVIEW_TEXT),
        ("urn:li:activity:demo-job", JOB_UPDATE_TEXT),
    ]
    result: list[CollectedPost] = []
    for urn, text in posts:
        result.append(
            CollectedPost(
                post_urn=urn,
                post_url=f"https://www.linkedin.com/feed/update/{urn}",
                author="Demo Author",
                posted_at_text="1d",
                raw_text=text,
                content_hash=content_hash(text),
            )
        )
    return result


class DemoExtractor:
    """Deterministic stand-in for the LLM, used only by `python -m app.main demo`."""

    def classify(self, post_text: str) -> PostClassification:
        related = "interview" in post_text.lower()
        return PostClassification(
            is_interview_related=related,
            reason="demo: interview keyword" if related else "demo: not interview-related",
        )

    def extract(self, post_text: str, *, ground: bool = True) -> ExtractionResult:
        if "GIL" not in post_text:
            return ExtractionResult(is_interview_related=False, qa_pairs=[], reason="demo")
        return ExtractionResult(
            is_interview_related=True,
            reason="demo sample post",
            qa_pairs=[
                QAPairDraft(
                    question="What is the GIL?",
                    answer="Global Interpreter Lock in CPython.",
                    answered=True,
                    category="Python",
                ),
                QAPairDraft(
                    question="What is a primary key?",
                    answer="A column that uniquely identifies a row.",
                    answered=True,
                    category="SQL",
                ),
            ],
        )
