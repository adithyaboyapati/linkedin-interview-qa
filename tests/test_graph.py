"""LangGraph workflow tests. Uses fake LLM + injected posts (no LinkedIn)."""

from __future__ import annotations

from pathlib import Path

from langgraph.types import Command

from app.config import Settings
from app.extraction.normalizer import content_hash
from app.graph.workflow import _pending_interrupts, build_graph, open_repository, run_workflow
from app.models import CollectedPost, ExtractionResult, PostClassification, QAPairDraft

GIL_POST = (
    "Interview questions I was asked:\n"
    "Q: What is the GIL?\n"
    "A: Global Interpreter Lock in CPython."
)
JOB_POST = "Excited to start my new role next week at Acme!"


def _post(text: str, urn: str) -> CollectedPost:
    return CollectedPost(
        post_urn=urn,
        post_url=f"https://www.linkedin.com/feed/update/{urn}",
        author="Ada",
        posted_at_text="1d",
        raw_text=text,
        content_hash=content_hash(text),
    )


class FakeExtractor:
    def __init__(self, *, extract_results: list[ExtractionResult] | None = None) -> None:
        self.classify_calls: list[str] = []
        self.extract_calls: list[str] = []
        self.extract_results = list(extract_results or [])

    def classify(self, post_text: str) -> PostClassification:
        self.classify_calls.append(post_text)
        related = "interview" in post_text.lower()
        return PostClassification(
            is_interview_related=related,
            reason="contains interview Q&A" if related else "not an interview post",
        )

    def extract(self, post_text: str, *, ground: bool = True) -> ExtractionResult:
        self.extract_calls.append(post_text)
        if self.extract_results:
            return self.extract_results.pop(0)
        return ExtractionResult(
            is_interview_related=True,
            qa_pairs=[
                QAPairDraft(
                    question="What is the GIL?",
                    answer="Global Interpreter Lock in CPython.",
                    answered=True,
                    category="Python",
                )
            ],
        )


def _invoke(graph, state: dict, thread_id: str = "test-thread") -> dict:
    return graph.invoke(state, {"configurable": {"thread_id": thread_id}})


def _initial(posts: list[CollectedPost]) -> dict:
    return {
        "posts": [post.model_dump() for post in posts],
        "interview_posts": [],
        "qa_pairs": [],
        "validation_results": [],
        "extract_attempts": 0,
        "skipped_posts": [],
        "pdf_path": "",
        "skip_scrape": True,
        "logs": [],
    }


def test_graph_happy_path_writes_pdf(tmp_settings: Settings) -> None:
    repo = open_repository(tmp_settings)
    extractor = FakeExtractor()
    graph = build_graph(settings=tmp_settings, extractor=extractor, repo=repo)
    result = _invoke(graph, _initial([_post(GIL_POST, "urn:li:activity:1"), _post(JOB_POST, "urn:li:activity:2")]))

    assert len(extractor.classify_calls) == 2
    assert len(extractor.extract_calls) == 1
    assert result["extract_attempts"] == 1
    assert len(result["qa_pairs"]) == 1
    assert result["qa_pairs"][0]["question"] == "What is the GIL?"
    assert result["pdf_path"].endswith("linkedin_interview_qa.pdf")
    assert Path(result["pdf_path"]).exists()
    node_names = [log.split(": ", 1)[1] for log in result["logs"] if log.startswith("[graph] running node:")]
    assert node_names == [
        "collect_posts",
        "classify_posts",
        "extract_qa",
        "validate_qa",
        "deduplicate",
        "save_to_db",
        "generate_pdf",
    ]
    stats = repo.stats()
    assert stats.posts == 2
    assert stats.interview_related == 1
    assert stats.not_interview_related == 1
    assert stats.answered_qa_pairs == 1
    repo.session.close()


def test_graph_retries_invalid_qa_once(tmp_settings: Settings) -> None:
    repo = open_repository(tmp_settings)
    extractor = FakeExtractor(
        extract_results=[
            ExtractionResult(
                is_interview_related=True,
                qa_pairs=[
                    QAPairDraft(
                        question="What is the GIL?",
                        answer="A hallucinated answer that is not in the post.",
                        answered=True,
                        category="Python",
                    )
                ],
            ),
            ExtractionResult(
                is_interview_related=True,
                qa_pairs=[
                    QAPairDraft(
                        question="What is the GIL?",
                        answer="Global Interpreter Lock in CPython.",
                        answered=True,
                        category="Python",
                    )
                ],
            ),
        ]
    )
    graph = build_graph(settings=tmp_settings, extractor=extractor, repo=repo)
    result = _invoke(graph, _initial([_post(GIL_POST, "urn:li:activity:1")]), thread_id="retry")
    assert result["extract_attempts"] == 2
    assert result["qa_pairs"][0]["answer"] == "Global Interpreter Lock in CPython."
    assert "retry extract_qa" in " ".join(result["logs"]) or result["extract_attempts"] == 2
    repo.session.close()


def test_graph_skips_post_after_failed_retry(tmp_settings: Settings) -> None:
    repo = open_repository(tmp_settings)
    bad = ExtractionResult(
        is_interview_related=True,
        qa_pairs=[
            QAPairDraft(
                question="What is the GIL?",
                answer="Made up answer",
                answered=True,
                category="Python",
            )
        ],
    )
    extractor = FakeExtractor(extract_results=[bad, bad])
    graph = build_graph(settings=tmp_settings, extractor=extractor, repo=repo)
    result = _invoke(graph, _initial([_post(GIL_POST, "urn:li:activity:1")]), thread_id="skip")
    assert result["extract_attempts"] == 2
    assert result["qa_pairs"] == []
    assert result["skipped_posts"]
    assert repo.stats().answered_qa_pairs == 0
    repo.session.close()


def test_graph_deduplicates_questions(tmp_settings: Settings) -> None:
    repo = open_repository(tmp_settings)
    extractor = FakeExtractor()
    graph = build_graph(settings=tmp_settings, extractor=extractor, repo=repo)
    first = _post(GIL_POST, "urn:li:activity:1")
    second = _post(GIL_POST + "\nThanks for reading.", "urn:li:activity:2")
    result = _invoke(graph, _initial([first, second]))
    assert len(result["qa_pairs"]) == 1
    repo.session.close()


def test_graph_login_interrupt_then_resume(tmp_settings: Settings) -> None:
    repo = open_repository(tmp_settings)
    collected = [_post(GIL_POST, "urn:li:activity:1")]
    extractor = FakeExtractor()
    graph = build_graph(
        settings=tmp_settings,
        extractor=extractor,
        repo=repo,
        collect_fn=lambda: collected,
    )
    config = {"configurable": {"thread_id": "hitl-test"}}
    graph.invoke(
        {
            "posts": [],
            "skip_scrape": False,
            "require_login_interrupt": True,
            "logs": [],
            "extract_attempts": 0,
            "qa_pairs": [],
            "interview_posts": [],
            "validation_results": [],
            "skipped_posts": [],
            "pdf_path": "",
        },
        config,
    )
    interrupts = _pending_interrupts(graph, config)
    assert interrupts
    payload = getattr(interrupts[0], "value", interrupts[0])
    assert "Log in to LinkedIn" in str(payload)
    result = graph.invoke(Command(resume=True), config)
    assert result["posts"][0]["post_urn"] == "urn:li:activity:1"
    assert result["qa_pairs"][0]["question"] == "What is the GIL?"
    repo.session.close()


def test_run_workflow_helper(tmp_settings: Settings) -> None:
    result = run_workflow(
        tmp_settings,
        skip_scrape=True,
        posts=[_post(GIL_POST, "urn:li:activity:1")],
        extractor=FakeExtractor(),
        handle_login=False,
    )
    assert Path(result["pdf_path"]).is_file()


def test_graph_extracts_from_image_text(tmp_settings: Settings) -> None:
    repo = open_repository(tmp_settings)
    extractor = FakeExtractor()
    graph = build_graph(settings=tmp_settings, extractor=extractor, repo=repo)
    post = CollectedPost(
        post_urn="urn:li:activity:carousel",
        post_url="https://www.linkedin.com/feed/update/urn:li:activity:carousel",
        author="Ada",
        posted_at_text="1d",
        raw_text="Swipe through the slides.",
        content_hash=content_hash("Swipe through the slides."),
        image_text=(
            "Interview questions I was asked:\n"
            "Q: What is the GIL?\n"
            "A: Global Interpreter Lock in CPython."
        ),
    )
    result = _invoke(graph, _initial([post]), thread_id="carousel")
    assert "interview" in extractor.classify_calls[0].lower()
    assert "TEXT FROM POST IMAGES" in extractor.extract_calls[0]
    assert result["qa_pairs"][0]["question"] == "What is the GIL?"
    stored = repo.get_posts_for_extraction(force=True)[0]
    assert "GIL" in (stored.image_text or "")
    repo.session.close()
