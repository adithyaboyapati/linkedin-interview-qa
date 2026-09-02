"""Vision transcription helpers (no live LLM)."""

from __future__ import annotations

from pathlib import Path

from app.extraction.qa_extractor import answer_is_grounded, looks_interview_related
from app.extraction.vision import (
    SlideTranscriber,
    combined_source_text,
    filter_interview_image_text,
    looks_like_interview_qa,
    source_text_for_post,
)


def test_combined_source_text_includes_slide_section() -> None:
    text = combined_source_text("Swipe through", "Financial guardrails for autonomous agents?")
    assert "Swipe through" in text
    assert "TEXT FROM POST IMAGES" in text
    assert "Financial guardrails" in text
    assert looks_interview_related(text)
    assert answer_is_grounded("Financial guardrails for autonomous agents?", text)


def test_looks_like_interview_qa_skips_title_cards() -> None:
    assert looks_like_interview_qa("Financial guardrails for autonomous agents?")
    assert looks_like_interview_qa("How would you design transaction processing?")
    assert looks_like_interview_qa("1. Explain RAG evaluation\n2. What is a guardrail")
    assert not looks_like_interview_qa("SCENARIO BASED INTERVIEW Check caption for the bundle link")
    assert not looks_like_interview_qa("Swipe")
    assert filter_interview_image_text("Check caption for the AI Interview Bundle link.") == ""


def test_filter_keeps_only_qa_slides() -> None:
    blob = (
        "[Slide slide_00.png]\nSCENARIO BASED INTERVIEW Check caption for the bundle\n\n"
        "[Slide slide_01.png]\nHow would you add financial guardrails for autonomous agents?\n"
        "Require human approval above a threshold."
    )
    kept = filter_interview_image_text(blob)
    assert "How would you add financial guardrails" in kept
    assert "SCENARIO BASED INTERVIEW" not in kept


def test_combined_source_text_drops_non_qa_images() -> None:
    text = combined_source_text(
        "Swipe through",
        "SCENARIO BASED INTERVIEW Check caption for the bundle link",
    )
    assert "TEXT FROM POST IMAGES" not in text
    assert text == "Swipe through"


def test_source_text_without_slides_is_caption_only(tmp_path: Path) -> None:
    combined, image_text = source_text_for_post(
        caption="Caption only",
        dump_dir=tmp_path,
        content_hash="abc",
    )
    assert combined == "Caption only"
    assert image_text == ""


def test_transcriber_uses_injected_fn(tmp_path: Path) -> None:
    folder = tmp_path / "hash" / "slides"
    folder.mkdir(parents=True)
    (folder / "slide_00.png").write_bytes(b"png-a")
    (folder / "slide_01.png").write_bytes(b"png-b")

    transcriber = SlideTranscriber(
        api_key="x",
        base_url="http://localhost",
        model="test",
        transcribe_fn=lambda path: (
            "How would you add financial guardrails?"
            if path.name.endswith("00.png")
            else "Check caption for the AI Interview Bundle link."
        ),
    )
    combined, image_text = source_text_for_post(
        caption="Swipe through",
        content_hash="hash",
        dump_dir=tmp_path,
        transcriber=transcriber,
    )
    assert "financial guardrails" in image_text.lower()
    assert "Interview Bundle" not in image_text
    assert "TEXT FROM POST IMAGES" in combined
    assert combined.startswith("Swipe through")


def test_source_text_skips_ocr_when_image_text_already_present(tmp_path: Path) -> None:
    folder = tmp_path / "hash" / "slides"
    folder.mkdir(parents=True)
    (folder / "slide_00.png").write_bytes(b"png")
    transcriber = SlideTranscriber(
        api_key="x",
        base_url="http://localhost",
        model="test",
        transcribe_fn=lambda _path: "should not run",
    )
    combined, image_text = source_text_for_post(
        caption="Caption",
        image_text="Q: What is the GIL? A: Global Interpreter Lock in CPython.",
        content_hash="hash",
        dump_dir=tmp_path,
        transcriber=transcriber,
    )
    assert "What is the GIL?" in image_text
    assert "should not run" not in combined
