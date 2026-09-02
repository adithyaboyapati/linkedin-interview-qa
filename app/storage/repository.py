"""CRUD helpers for posts and Q&A pairs."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.demo import is_sample_post
from app.extraction.normalizer import normalize_text, pair_hash, question_hash
from app.models import CATEGORY_ORDER, CollectedPost, CollectorStats, QAPairDraft, StoredQA
from app.storage.database import PostRecord, QAPairRecord, utcnow


class Repository:
    def __init__(self, session: Session, raw_dump_dir: Path | None = None) -> None:
        self.session = session
        self.raw_dump_dir = raw_dump_dir

    def upsert_post(self, post: CollectedPost) -> tuple[PostRecord, bool]:
        """Insert a post if it is new. Returns (record, created)."""
        existing = self._find_existing_post(post)
        if existing is not None:
            self._refresh_metadata(existing, post)
            return existing, False

        record = PostRecord(
            content_hash=post.content_hash,
            post_urn=post.post_urn,
            post_url=post.post_url,
            author=post.author,
            posted_at_text=post.posted_at_text,
            raw_text=normalize_text(post.raw_text),
            image_text=post.image_text or None,
            extraction_status="pending",
        )
        self.session.add(record)
        self.session.flush()
        self._dump_raw(post)
        return record, True

    def upsert_posts(self, posts: Iterable[CollectedPost]) -> tuple[int, int]:
        created = 0
        skipped = 0
        for post in posts:
            _, is_new = self.upsert_post(post)
            if is_new:
                created += 1
            else:
                skipped += 1
        self.session.commit()
        return created, skipped

    def get_posts_for_extraction(self, *, force: bool = False, limit: int | None = None) -> list[PostRecord]:
        stmt = select(PostRecord)
        if not force:
            stmt = stmt.where(PostRecord.extraction_status == "pending")
        stmt = stmt.order_by(PostRecord.id.asc())
        if limit is not None:
            stmt = stmt.limit(limit)
        return list(self.session.scalars(stmt))

    def mark_not_interview_related(self, post: PostRecord, reason: str) -> None:
        post.is_interview_related = False
        post.extraction_status = "skipped"
        post.extraction_reason = reason
        post.updated_at = utcnow()
        self.session.commit()

    def mark_extraction_failed(self, post: PostRecord, reason: str) -> None:
        post.extraction_status = "failed"
        post.extraction_reason = reason
        post.updated_at = utcnow()
        self.session.commit()

    def save_extraction(
        self,
        post: PostRecord,
        pairs: list[QAPairDraft],
        *,
        is_interview_related: bool,
        reason: str = "",
    ) -> tuple[int, int]:
        """Persist answered Q&A pairs. Unanswered pairs are stored only as skipped."""
        post.is_interview_related = is_interview_related
        post.extraction_status = "done"
        post.extraction_reason = reason
        post.updated_at = utcnow()

        created = 0
        skipped = 0
        for draft in pairs:
            if not draft.answered or not draft.answer:
                skipped += 1
                continue
            inserted = self._insert_qa_pair(post, draft)
            if inserted:
                created += 1
            else:
                skipped += 1
        self.session.commit()
        return created, skipped

    def answered_qa_by_category(self) -> dict[str, list[StoredQA]]:
        stmt = (
            select(QAPairRecord, PostRecord)
            .join(PostRecord, QAPairRecord.post_id == PostRecord.id)
            .where(QAPairRecord.is_answered.is_(True))
            .where(QAPairRecord.answer.is_not(None))
            .order_by(PostRecord.id.asc(), QAPairRecord.id.asc())
        )
        grouped: dict[str, list[StoredQA]] = {name: [] for name in CATEGORY_ORDER}
        extra: dict[str, list[StoredQA]] = {}
        first_post_id: dict[str, int] = {}

        for pair, post in self.session.execute(stmt):
            if is_sample_post(post):
                continue
            item = StoredQA(
                id=pair.id,
                question=pair.question,
                answer=pair.answer or "",
                category=pair.category,
                source_url=pair.source_url or post.post_url,
                author=post.author,
                posted_at_text=post.posted_at_text,
            )
            if pair.category in grouped:
                grouped[pair.category].append(item)
            else:
                extra.setdefault(pair.category, []).append(item)
            first_post_id.setdefault(pair.category, post.id)

        combined = {k: v for k, v in grouped.items() if v}
        combined.update(extra)
        return dict(sorted(combined.items(), key=lambda item: first_post_id.get(item[0], 10**9)))

    def stats(self) -> CollectorStats:
        posts = self.session.scalar(select(func.count(PostRecord.id))) or 0
        interview = (
            self.session.scalar(
                select(func.count(PostRecord.id)).where(PostRecord.is_interview_related.is_(True))
            )
            or 0
        )
        not_interview = (
            self.session.scalar(
                select(func.count(PostRecord.id)).where(PostRecord.is_interview_related.is_(False))
            )
            or 0
        )
        pending = (
            self.session.scalar(
                select(func.count(PostRecord.id)).where(PostRecord.extraction_status == "pending")
            )
            or 0
        )
        failed = (
            self.session.scalar(
                select(func.count(PostRecord.id)).where(PostRecord.extraction_status == "failed")
            )
            or 0
        )
        qa_total = self.session.scalar(select(func.count(QAPairRecord.id))) or 0
        answered = (
            self.session.scalar(
                select(func.count(QAPairRecord.id)).where(QAPairRecord.is_answered.is_(True))
            )
            or 0
        )
        by_category_rows = self.session.execute(
            select(QAPairRecord.category, func.count(QAPairRecord.id))
            .where(QAPairRecord.is_answered.is_(True))
            .group_by(QAPairRecord.category)
        )
        return CollectorStats(
            posts=posts,
            interview_related=interview,
            not_interview_related=not_interview,
            pending_extraction=pending,
            extraction_failed=failed,
            qa_pairs=qa_total,
            answered_qa_pairs=answered,
            unanswered_qa_pairs=qa_total - answered,
            by_category={row[0]: row[1] for row in by_category_rows},
        )

    def _find_existing_post(self, post: CollectedPost) -> PostRecord | None:
        if post.post_urn:
            found = self.session.scalar(
                select(PostRecord).where(PostRecord.post_urn == post.post_urn)
            )
            if found is not None:
                return found
        return self.session.scalar(
            select(PostRecord).where(PostRecord.content_hash == post.content_hash)
        )

    def _refresh_metadata(self, existing: PostRecord, post: CollectedPost) -> None:
        if post.post_url and not existing.post_url:
            existing.post_url = post.post_url
        if post.author and not existing.author:
            existing.author = post.author
        if post.posted_at_text and not existing.posted_at_text:
            existing.posted_at_text = post.posted_at_text
        if post.post_urn and not existing.post_urn:
            existing.post_urn = post.post_urn
        if post.image_text and not existing.image_text:
            existing.image_text = post.image_text
        existing.updated_at = utcnow()

    def set_image_text(self, post: PostRecord, image_text: str) -> None:
        post.image_text = image_text or None
        post.updated_at = utcnow()
        self.session.commit()

    def _insert_qa_pair(self, post: PostRecord, draft: QAPairDraft) -> bool:
        q_hash = question_hash(draft.question)
        p_hash = pair_hash(draft.question, draft.answer)

        existing_pair = self.session.scalar(
            select(QAPairRecord).where(QAPairRecord.pair_hash == p_hash)
        )
        if existing_pair is not None:
            return False

        existing_question = self.session.scalar(
            select(QAPairRecord).where(QAPairRecord.question_hash == q_hash)
        )
        if existing_question is not None:
            if existing_question.is_answered:
                return False
            existing_question.answer = draft.answer
            existing_question.is_answered = True
            existing_question.category = draft.category
            existing_question.pair_hash = p_hash
            existing_question.source_url = post.post_url
            existing_question.post_id = post.id
            return True

        self.session.add(
            QAPairRecord(
                post_id=post.id,
                question=draft.question.strip(),
                answer=draft.answer,
                is_answered=True,
                category=draft.category,
                question_hash=q_hash,
                pair_hash=p_hash,
                source_url=post.post_url,
            )
        )
        self.session.flush()
        return True

    def purge_sample_posts(self) -> int:
        """Remove offline demo fixtures so they cannot leak into the LinkedIn PDF."""
        removed = 0
        posts = list(self.session.scalars(select(PostRecord)))
        for post in posts:
            if not is_sample_post(post):
                continue
            self.session.execute(delete(QAPairRecord).where(QAPairRecord.post_id == post.id))
            self.session.delete(post)
            removed += 1
        if removed:
            self.session.commit()
        return removed

    def _dump_raw(self, post: CollectedPost) -> None:
        if self.raw_dump_dir is None:
            return
        self.raw_dump_dir.mkdir(parents=True, exist_ok=True)
        path = self.raw_dump_dir / f"{post.content_hash}.json"
        if path.exists():
            return
        path.write_text(post.model_dump_json(indent=2), encoding="utf-8")
