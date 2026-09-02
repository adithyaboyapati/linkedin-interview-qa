"""PDF generation tests."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from app.models import PdfDocumentData, StoredQA
from app.pdf.generator import generate_pdf, render_html


def _sample_data() -> PdfDocumentData:
    return PdfDocumentData(
        creator="https://www.linkedin.com/in/example-profile",
        generated_at=datetime(2026, 9, 2),
        categories={
            "Python": [
                StoredQA(
                    id=1,
                    question="What is the GIL?",
                    answer="Global Interpreter Lock in CPython.",
                    category="Python",
                    source_url="https://www.linkedin.com/feed/update/urn:li:activity:1",
                )
            ],
            "SQL": [
                StoredQA(
                    id=2,
                    question="What is a primary key?",
                    answer="A column that uniquely identifies a row.",
                    category="SQL",
                    source_url="https://www.linkedin.com/feed/update/urn:li:activity:2",
                )
            ],
        },
    )


def test_render_html_contains_sections_and_sources() -> None:
    html = render_html(_sample_data())
    assert "LinkedIn Interview Questions" in html
    assert "Python" in html
    assert "SQL" in html
    assert "Q1. What is the GIL?" in html
    assert "Global Interpreter Lock in CPython." in html
    assert "urn:li:activity:1" in html
    assert "2026-09-02" in html


def test_generate_pdf_writes_file(tmp_path: Path) -> None:
    output = tmp_path / "linkedin_interview_qa.pdf"
    path = generate_pdf(_sample_data(), output)
    assert path.exists()
    assert path.stat().st_size > 500
    assert path.read_bytes()[:4] == b"%PDF"
