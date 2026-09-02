"""CLI wiring tests."""

from __future__ import annotations

import pytest

from app.config import Settings
from app.extraction.normalizer import content_hash
from app.main import main
from app.models import CollectedPost, QAPairDraft
from app.storage.database import init_db, make_engine, make_session_factory
from app.storage.repository import Repository


def test_stats_and_pdf_cli(tmp_settings: Settings, capsys) -> None:
    engine = make_engine(tmp_settings.database_path)
    init_db(engine)
    repo = Repository(make_session_factory(engine)(), raw_dump_dir=tmp_settings.raw_dump_dir)
    post, _ = repo.upsert_post(
        CollectedPost(
            post_urn="urn:li:activity:1",
            post_url="https://www.linkedin.com/feed/update/urn:li:activity:1",
            author="Ada",
            posted_at_text="1d",
            raw_text="Q: What is GIL? A: Global Interpreter Lock",
            content_hash=content_hash("Q: What is GIL? A: Global Interpreter Lock"),
        )
    )
    repo.session.commit()
    repo.save_extraction(
        post,
        [
            QAPairDraft(
                question="What is GIL?",
                answer="Global Interpreter Lock",
                answered=True,
                category="Python",
            )
        ],
        is_interview_related=True,
    )

    assert main(["stats"]) == 0
    out = capsys.readouterr().out
    assert "Posts stored:          1" in out
    assert "Python: 1" in out

    assert main(["generate-pdf"]) == 0
    pdf_out = capsys.readouterr().out
    assert "Wrote 1 answered Q&A pair(s)" in pdf_out
    assert tmp_settings.pdf_output_path.exists()
    assert tmp_settings.pdf_output_path.read_bytes()[:4] == b"%PDF"


def test_cli_requires_command() -> None:
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_run_from_db_without_posts(tmp_settings: Settings, capsys) -> None:
    assert main(["run", "--from-db"]) == 1
    assert "No stored posts" in capsys.readouterr().out


def test_demo_writes_pdf(tmp_settings: Settings, capsys) -> None:
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "Demo finished" in out
    demo_pdf = tmp_settings.pdf_output_path.with_name("demo_" + tmp_settings.pdf_output_path.name)
    assert demo_pdf.exists()
    assert demo_pdf.read_bytes()[:4] == b"%PDF"
    assert not tmp_settings.pdf_output_path.exists()


def test_check_command(tmp_settings: Settings, capsys) -> None:
    assert main(["check"]) == 0
    out = capsys.readouterr().out
    assert "Readiness check" in out
    assert "Offline demo" in out
