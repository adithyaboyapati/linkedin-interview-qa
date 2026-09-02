"""LangGraph nodes. Each function is one step in the Q&A workflow."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from langchain_core.tools import BaseTool
from langgraph.types import interrupt

from app.config import Settings
from app.extraction.normalizer import question_hash
from app.extraction.qa_extractor import QAExtractor, answer_is_grounded
from app.extraction.vision import SlideTranscriber, combined_source_text, source_text_for_post
from app.graph.state import CollectorState
from app.models import CollectedPost, PdfDocumentData, QAPairDraft
from app.pdf.generator import generate_pdf
from app.storage.repository import Repository

logger = logging.getLogger("app.graph")


def log_node(name: str) -> dict[str, list[str]]:
    message = f"[graph] running node: {name}"
    print(message, flush=True)
    logger.info(message)
    return {"logs": [message]}


def _post_key(post: dict[str, Any]) -> str:
    return post.get("content_hash") or post.get("post_urn") or ""


def _source_text(payload: dict[str, Any]) -> str:
    return combined_source_text(payload.get("raw_text") or "", payload.get("image_text") or "")


def _enrich_payload(
    payload: dict[str, Any],
    *,
    transcriber: SlideTranscriber | None,
    raw_dump_dir,
) -> dict[str, Any]:
    _, image_text = source_text_for_post(
        caption=payload.get("raw_text") or "",
        image_text=payload.get("image_text") or "",
        content_hash=payload.get("content_hash") or "",
        dump_dir=raw_dump_dir,
        transcriber=transcriber,
    )
    return {**payload, "image_text": image_text}


def make_collect_posts_node(collect_tool: BaseTool) -> Callable[[CollectorState], dict]:
    def collect_posts(state: CollectorState) -> dict:
        updates = log_node("collect_posts")
        if state.get("skip_scrape"):
            logger.info("skip_scrape=True; using %s post(s) already in state", len(state.get("posts") or []))
            return updates

        # Optional teaching interrupt. The live CLI skips this so Chromium opens immediately.
        if state.get("require_login_interrupt"):
            interrupt(
                {
                    "type": "linkedin_login",
                    "message": (
                        "Human-in-the-loop: a Chromium window will open (or reuse "
                        "data/browser_profile). Log in to LinkedIn manually if asked. "
                        "This graph will not type your password or solve CAPTCHA. "
                        "Resume when you are logged in."
                    ),
                }
            )

        print("Opening Chromium and loading LinkedIn...", flush=True)
        posts = collect_tool.invoke({})
        print(f"Collected {len(posts)} accessible post(s).", flush=True)
        return {**updates, "posts": posts}

    return collect_posts


def make_classify_posts_node(
    extractor: QAExtractor,
    *,
    transcriber: SlideTranscriber | None = None,
    raw_dump_dir=None,
) -> Callable[[CollectorState], dict]:
    def classify_posts(state: CollectorState) -> dict:
        updates = log_node("classify_posts")
        interview: list[dict[str, Any]] = []
        enriched: list[dict[str, Any]] = []
        for payload in state.get("posts") or []:
            payload = _enrich_payload(
                payload, transcriber=transcriber, raw_dump_dir=raw_dump_dir
            )
            enriched.append(payload)
            text = _source_text(payload)
            try:
                result = extractor.classify(text)
            except Exception as exc:  # noqa: BLE001 - keep classifying remaining posts
                logger.warning("classify failed for %s: %s", _post_key(payload), exc)
                continue
            if result.is_interview_related:
                interview.append(payload)
                logger.info("interview-related: %s (%s)", _post_key(payload), result.reason)
            else:
                logger.info("skipped non-interview: %s (%s)", _post_key(payload), result.reason)
        return {**updates, "posts": enriched, "interview_posts": interview}

    return classify_posts


def make_extract_qa_node(extractor: QAExtractor) -> Callable[[CollectorState], dict]:
    def extract_qa(state: CollectorState) -> dict:
        updates = log_node("extract_qa")
        attempts = int(state.get("extract_attempts") or 0) + 1
        invalid_hashes = {
            item["post_hash"]
            for item in state.get("validation_results") or []
            if not item.get("valid")
        }
        targets = list(state.get("interview_posts") or [])
        if attempts > 1:
            targets = [post for post in targets if _post_key(post) in invalid_hashes]
            logger.info("retry extract_qa for %s invalid post(s)", len(targets))

        kept_valid = [pair for pair in state.get("qa_pairs") or [] if pair.get("valid")]
        new_pairs: list[dict[str, Any]] = []
        for payload in targets:
            try:
                result = extractor.extract(_source_text(payload), ground=False)
            except Exception as exc:  # noqa: BLE001
                logger.warning("extract failed for %s: %s", _post_key(payload), exc)
                continue
            for pair in result.qa_pairs:
                new_pairs.append(
                    {
                        "post_hash": _post_key(payload),
                        "source_url": payload.get("post_url"),
                        "question": pair.question,
                        "answer": pair.answer,
                        "answered": pair.answered,
                        "category": pair.category,
                        "valid": False,
                    }
                )
        logger.info("extracted %s draft Q&A pair(s) on attempt %s", len(new_pairs), attempts)
        return {
            **updates,
            "qa_pairs": kept_valid + new_pairs,
            "extract_attempts": attempts,
        }

    return extract_qa


def validate_qa(state: CollectorState) -> dict:
    updates = log_node("validate_qa")
    posts = {_post_key(post): post for post in state.get("interview_posts") or []}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in state.get("qa_pairs") or []:
        grouped[pair.get("post_hash")].append(pair)

    validated: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    attempts = int(state.get("extract_attempts") or 0)
    skipped: list[str] = []

    for post_hash, post in posts.items():
        post_text = _source_text(post)
        errors: list[str] = []
        valid_for_post = 0
        for pair in grouped.get(post_hash, []):
            answer = pair.get("answer")
            if not pair.get("answered") or not answer:
                errors.append(f"unanswered: {pair.get('question', '')[:60]}")
                continue
            if not answer_is_grounded(str(answer), post_text):
                errors.append(f"not grounded in post: {pair.get('question', '')[:60]}")
                continue
            validated.append({**pair, "valid": True, "answered": True})
            valid_for_post += 1
        is_valid = valid_for_post > 0
        if not is_valid:
            if not errors:
                errors.append("no answered Q&A pairs")
            if attempts >= 2:
                skipped.append(post_hash)
        results.append(
            {
                "post_hash": post_hash,
                "valid": is_valid,
                "errors": errors,
            }
        )
        logger.info(
            "validate %s: %s (%s valid pair(s))",
            post_hash,
            "ok" if is_valid else "invalid",
            valid_for_post,
        )

    return {
        **updates,
        "qa_pairs": validated,
        "validation_results": results,
        "skipped_posts": skipped,
    }


def route_after_validate(state: CollectorState) -> str:
    invalid = [item for item in state.get("validation_results") or [] if not item.get("valid")]
    attempts = int(state.get("extract_attempts") or 0)
    if invalid and attempts < 2:
        logger.info("conditional edge: invalid Q&A -> retry extract_qa")
        return "extract_qa"
    if invalid:
        logger.info("conditional edge: still invalid after retry -> skip post(s) and deduplicate")
    else:
        logger.info("conditional edge: valid -> deduplicate")
    return "deduplicate"


def deduplicate(state: CollectorState) -> dict:
    updates = log_node("deduplicate")
    unique: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    for pair in state.get("qa_pairs") or []:
        if not pair.get("valid") or not pair.get("answer"):
            continue
        key = question_hash(str(pair.get("question") or ""))
        if key in seen_questions:
            continue
        seen_questions.add(key)
        unique.append(pair)
    logger.info("deduplicated to %s Q&A pair(s)", len(unique))
    return {**updates, "qa_pairs": unique}


def make_save_to_db_node(repo: Repository) -> Callable[[CollectorState], dict]:
    def save_to_db(state: CollectorState) -> dict:
        updates = log_node("save_to_db")
        records = {}
        for payload in state.get("posts") or []:
            post = CollectedPost.model_validate(payload)
            record, _ = repo.upsert_post(post)
            records[_post_key(payload)] = record
        repo.session.commit()

        pairs_by_post: dict[str, list[QAPairDraft]] = defaultdict(list)
        for pair in state.get("qa_pairs") or []:
            pairs_by_post[pair["post_hash"]].append(
                QAPairDraft(
                    question=pair["question"],
                    answer=pair.get("answer"),
                    answered=True,
                    category=pair.get("category") or "Other",
                )
            )

        interview_hashes = {_post_key(post) for post in state.get("interview_posts") or []}
        skipped = set(state.get("skipped_posts") or [])

        for post_hash, record in records.items():
            if post_hash in skipped:
                repo.mark_extraction_failed(record, "validation failed after one retry")
                continue
            if post_hash not in interview_hashes:
                repo.mark_not_interview_related(record, "classified as not interview-related")
                continue
            repo.save_extraction(
                record,
                pairs_by_post.get(post_hash, []),
                is_interview_related=True,
                reason="graph extract/validate",
            )
        logger.info("saved graph results to sqlite")
        return updates

    return save_to_db


def make_generate_pdf_node(settings: Settings, repo: Repository) -> Callable[[CollectorState], dict]:
    def generate_pdf_node(state: CollectorState) -> dict:
        updates = log_node("generate_pdf")
        repo.purge_sample_posts()
        grouped = repo.answered_qa_by_category()
        data = PdfDocumentData(
            creator=settings.linkedin_profile_url or "LinkedIn profile",
            generated_at=datetime.now(timezone.utc).replace(tzinfo=None),
            categories=grouped,
        )
        path = generate_pdf(data, settings.pdf_output_path)
        logger.info("wrote PDF to %s", path)
        return {**updates, "pdf_path": str(path)}

    return generate_pdf_node
