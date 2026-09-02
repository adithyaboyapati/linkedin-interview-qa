"""SQLite repository tests."""

from __future__ import annotations

from pathlib import Path

from app.extraction.normalizer import content_hash
from app.models import CollectedPost, QAPairDraft
from app.storage.database import init_db, make_engine, make_session_factory
from app.storage.repository import Repository


def _repo(tmp_path: Path) -> Repository:
    engine = make_engine(tmp_path / "test.db")
    init_db(engine)
    session = make_session_factory(engine)()
    return Repository(session, raw_dump_dir=tmp_path / "raw")


def _post(text: str, urn: str | None = None, url: str | None = None) -> CollectedPost:
    return CollectedPost(
        post_urn=urn,
        post_url=url,
        author="Ada",
        posted_at_text="2d",
        raw_text=text,
        content_hash=content_hash(text),
    )


def test_upsert_post_deduplicates_by_hash(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    created, skipped = repo.upsert_posts(
        [
            _post("Interview question: what is GIL?", urn="urn:li:activity:1"),
            _post("Interview question: what is GIL?"),
        ]
    )
    assert created == 1
    assert skipped == 1
    assert repo.stats().posts == 1
    dump = tmp_path / "raw" / f"{content_hash('Interview question: what is GIL?')}.json"
    assert dump.exists()


def test_upsert_post_deduplicates_by_urn(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    repo.upsert_posts(
        [
            _post("First wording", urn="urn:li:activity:99"),
            _post("Different wording", urn="urn:li:activity:99"),
        ]
    )
    assert repo.stats().posts == 1


def test_save_extraction_keeps_only_answered_pairs(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    record, _ = repo.upsert_post(_post("Q: What is GIL? A: Global Interpreter Lock", url="https://lnkd.in/x"))
    repo.session.commit()
    created, skipped = repo.save_extraction(
        record,
        [
            QAPairDraft(
                question="What is GIL?",
                answer="Global Interpreter Lock",
                answered=True,
                category="Python",
            ),
            QAPairDraft(question="What is Python?", answer=None, answered=False, category="Python"),
        ],
        is_interview_related=True,
        reason="contains interview Q&A",
    )
    assert created == 1
    assert skipped == 1
    grouped = repo.answered_qa_by_category()
    assert "Python" in grouped
    assert grouped["Python"][0].source_url == "https://lnkd.in/x"
    assert grouped["Python"][0].answer == "Global Interpreter Lock"


def test_pdf_categories_follow_feed_order(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    later, _ = repo.upsert_post(_post("later SQL post Q: What is a join? A: Combine rows.", urn="urn:li:activity:2"))
    first, _ = repo.upsert_post(_post("first ML post Q: What is overfitting? A: Fit train too well.", urn="urn:li:activity:1"))
    repo.session.commit()
    # Insert later post first, then the feed-first post, so id order matches collection order.
    repo.save_extraction(
        later,
        [QAPairDraft(question="What is a join?", answer="Combine rows.", answered=True, category="SQL")],
        is_interview_related=True,
    )
    repo.save_extraction(
        first,
        [QAPairDraft(question="What is overfitting?", answer="Fit train too well.", answered=True, category="ML")],
        is_interview_related=True,
    )
    grouped = repo.answered_qa_by_category()
    assert list(grouped)[0] == "SQL"


def test_question_dedup_prefers_answered_version(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    first, _ = repo.upsert_post(_post("post one", urn="urn:li:activity:1"))
    second, _ = repo.upsert_post(_post("post two", urn="urn:li:activity:2"))
    repo.session.commit()
    repo.save_extraction(
        first,
        [QAPairDraft(question="What is GIL?", answer="short", answered=True, category="Python")],
        is_interview_related=True,
    )
    created, skipped = repo.save_extraction(
        second,
        [QAPairDraft(question="What is GIL?", answer="short", answered=True, category="Python")],
        is_interview_related=True,
    )
    assert created == 0
    assert skipped == 1
    assert repo.stats().answered_qa_pairs == 1


def test_pending_extraction_query(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    record, _ = repo.upsert_post(_post("hello interview world"))
    repo.session.commit()
    pending = repo.get_posts_for_extraction()
    assert len(pending) == 1
    repo.mark_not_interview_related(record, "not interview")
    assert repo.get_posts_for_extraction() == []
    assert repo.stats().not_interview_related == 1


def test_purge_sample_posts_are_excluded_from_pdf_data(tmp_path: Path) -> None:
    from app.demo import sample_posts

    repo = _repo(tmp_path)
    demo, _ = repo.upsert_post(sample_posts()[0])
    real, _ = repo.upsert_post(_post("Q: What is RAG? A: Retrieval-Augmented Generation", urn="urn:li:activity:real"))
    repo.session.commit()
    repo.save_extraction(
        demo,
        [QAPairDraft(question="What is the GIL?", answer="Global Interpreter Lock in CPython.", answered=True, category="Python")],
        is_interview_related=True,
    )
    repo.save_extraction(
        real,
        [QAPairDraft(question="What is RAG?", answer="Retrieval-Augmented Generation", answered=True, category="RAG")],
        is_interview_related=True,
    )
    grouped = repo.answered_qa_by_category()
    assert "Python" not in grouped
    assert grouped["RAG"][0].question == "What is RAG?"
    assert repo.purge_sample_posts() == 1
    assert repo.stats().posts == 1


def test_image_text_is_persisted(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    post = _post("Swipe through the carousel", urn="urn:li:activity:img")
    post.image_text = "Q: How would you design financial guardrails?"
    record, created = repo.upsert_post(post)
    repo.session.commit()
    assert created
    assert record.image_text.startswith("Q:")
    repo.set_image_text(record, "Updated slide text")
    assert record.image_text == "Updated slide text"
