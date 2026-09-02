"""CLI for collecting LinkedIn posts, extracting Q&A, and generating a PDF."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime

from app.config import Settings, get_settings
from app.models import PdfDocumentData
from app.storage.database import init_db, make_engine, make_session_factory
from app.storage.repository import Repository


def _open_repo(settings: Settings) -> Repository:
    settings.ensure_directories()
    engine = make_engine(settings.database_path)
    init_db(engine)
    session = make_session_factory(engine)()
    return Repository(session, raw_dump_dir=settings.raw_dump_dir)


def cmd_collect(settings: Settings, args: argparse.Namespace) -> int:
    from app.linkedin.browser import open_linkedin_session
    from app.linkedin.scraper import collect_posts

    profile_url = settings.require_profile_url()
    repo = _open_repo(settings)
    max_scrolls = args.max_scrolls or settings.max_scrolls
    max_posts = args.max_posts or settings.max_posts

    created_total = 0
    skipped_total = 0

    def on_batch(posts) -> None:
        nonlocal created_total, skipped_total
        created, skipped = repo.upsert_posts(posts)
        created_total += created
        skipped_total += skipped
        print(f"  stored {created} new post(s), skipped {skipped} duplicate(s)")

    print(f"Opening LinkedIn session for {profile_url}")
    print(f"Collecting up to {max_posts} post(s).", flush=True)
    with open_linkedin_session(settings) as page:
        posts = collect_posts(
            page,
            profile_url,
            max_scrolls=max_scrolls,
            scroll_pause_ms=settings.scroll_pause_ms,
            max_idle_scrolls=settings.max_idle_scrolls,
            max_posts=max_posts,
            on_batch=on_batch,
            raw_dump_dir=settings.raw_dump_dir,
        )

    print(
        f"Collection finished. Visible posts: {len(posts)}. "
        f"New: {created_total}. Duplicates: {skipped_total}."
    )
    repo.session.close()
    return 0


def cmd_extract(settings: Settings, args: argparse.Namespace) -> int:
    from app.extraction.qa_extractor import QAExtractor, looks_interview_related
    from app.extraction.vision import SlideTranscriber, slide_paths, source_text_for_post

    settings.require_openai_key()
    repo = _open_repo(settings)
    removed = repo.purge_sample_posts()
    if removed:
        print(f"Removed {removed} offline demo post(s) from the LinkedIn database.")
    posts = repo.get_posts_for_extraction(force=args.force, limit=args.limit)
    if not posts:
        print("No posts waiting for extraction.")
        repo.session.close()
        return 0

    extractor = QAExtractor(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.openai_model,
    )
    transcriber = SlideTranscriber(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.openai_model,
    )
    created_total = 0
    skipped_total = 0
    print(f"Extracting Q&A from {len(posts)} post(s) using {settings.openai_model}")

    for index, post in enumerate(posts, start=1):
        preview = post.raw_text[:80].replace("\n", " ")
        print(f"[{index}/{len(posts)}] {preview}")
        try:
            paths = slide_paths(settings.raw_dump_dir, post.content_hash)
            if paths:
                print(f"  transcribing {len(paths)} post image(s)")
            source, image_text = source_text_for_post(
                caption=post.raw_text,
                image_text=post.image_text or "",
                content_hash=post.content_hash,
                dump_dir=settings.raw_dump_dir,
                transcriber=transcriber,
                force=args.force,
            )
            if image_text != (post.image_text or ""):
                repo.set_image_text(post, image_text)
            if not args.force and not looks_interview_related(source):
                repo.mark_not_interview_related(post, "no interview keywords")
                print("  skipped (not interview-related by keyword filter)")
                continue
            result = extractor.extract(source)
            if not result.is_interview_related:
                repo.mark_not_interview_related(post, result.reason or "LLM: not interview-related")
                print("  skipped (LLM: not interview-related)")
                continue
            created, skipped = repo.save_extraction(
                post,
                result.qa_pairs,
                is_interview_related=True,
                reason=result.reason,
            )
            created_total += created
            skipped_total += skipped
            print(f"  saved {created} Q&A pair(s), skipped {skipped}")
        except Exception as exc:
            repo.mark_extraction_failed(post, str(exc))
            print(f"  failed: {exc}")

    print(f"Extraction finished. New Q&A: {created_total}. Duplicates/unanswered: {skipped_total}.")
    repo.session.close()
    return 0


def cmd_generate_pdf(settings: Settings, _args: argparse.Namespace) -> int:
    from app.pdf.generator import generate_pdf

    repo = _open_repo(settings)
    removed = repo.purge_sample_posts()
    if removed:
        print(f"Removed {removed} offline demo post(s) from the LinkedIn database.")
    grouped = repo.answered_qa_by_category()
    creator = settings.linkedin_profile_url or "LinkedIn profile"
    data = PdfDocumentData(
        creator=creator,
        generated_at=datetime.now(UTC).replace(tzinfo=None),
        categories=grouped,
    )
    path = generate_pdf(data, settings.pdf_output_path)
    answered = sum(len(items) for items in grouped.values())
    print(f"Wrote {answered} answered Q&A pair(s) to {path}")
    repo.session.close()
    return 0


def cmd_run(settings: Settings, args: argparse.Namespace) -> int:
    from app.graph.workflow import open_repository, run_workflow
    from app.models import CollectedPost

    posts = None
    skip_scrape = bool(args.from_db)
    if skip_scrape:
        repo = open_repository(settings)
        records = repo.get_posts_for_extraction(force=args.force, limit=args.limit)
        posts = [
            CollectedPost(
                post_urn=record.post_urn,
                post_url=record.post_url,
                author=record.author,
                posted_at_text=record.posted_at_text,
                raw_text=record.raw_text,
                content_hash=record.content_hash,
                image_text=record.image_text or "",
            )
            for record in records
        ]
        repo.session.close()
        if not posts:
            print("No stored posts to process. Run collect first or omit --from-db.")
            return 1
    else:
        settings.require_profile_url()

    settings.require_openai_key()
    result = run_workflow(settings, skip_scrape=skip_scrape, posts=posts)
    pdf_path = result.get("pdf_path")
    qa_count = len(result.get("qa_pairs") or [])
    print(f"Graph finished. Answered Q&A: {qa_count}. PDF: {pdf_path}")
    return 0


def cmd_stats(settings: Settings, _args: argparse.Namespace) -> int:
    repo = _open_repo(settings)
    repo.purge_sample_posts()
    stats = repo.stats()
    print("LinkedIn interview Q&A collector")
    print(f"  Profile:               {settings.linkedin_profile_url or '(not set)'}")
    print(f"  Posts stored:          {stats.posts}")
    print(f"  Interview-related:     {stats.interview_related}")
    print(f"  Not interview-related: {stats.not_interview_related}")
    print(f"  Pending extraction:    {stats.pending_extraction}")
    print(f"  Extraction failed:     {stats.extraction_failed}")
    print(f"  Q&A pairs stored:      {stats.qa_pairs}")
    print(f"  Answered Q&A:          {stats.answered_qa_pairs}")
    print(f"  Unanswered Q&A:        {stats.unanswered_qa_pairs}")
    if stats.by_category:
        print("  By category:")
        for name, count in sorted(stats.by_category.items(), key=lambda item: (-item[1], item[0])):
            print(f"    {name}: {count}")
    repo.session.close()
    return 0


def cmd_demo(settings: Settings, _args: argparse.Namespace) -> int:
    """Run the graph on sample posts. No LinkedIn login or API key required."""
    from app.demo import DemoExtractor, sample_posts
    from app.graph.workflow import run_workflow

    demo_settings = settings.model_copy(
        update={
            "database_path": settings.database_path.with_name("demo_" + settings.database_path.name),
            "pdf_output_path": settings.pdf_output_path.with_name("demo_" + settings.pdf_output_path.name),
            "raw_dump_dir": settings.raw_dump_dir.with_name(settings.raw_dump_dir.name + "_demo"),
        }
    )
    demo_settings.ensure_directories()
    print("Running offline demo (sample posts, no LinkedIn login).")
    print("Demo data is stored separately and will not appear in the LinkedIn PDF.")
    result = run_workflow(
        demo_settings,
        skip_scrape=True,
        posts=sample_posts(),
        extractor=DemoExtractor(),
        handle_login=False,
    )
    pdf_path = result.get("pdf_path")
    qa_count = len(result.get("qa_pairs") or [])
    print()
    print(f"Demo finished. Answered Q&A: {qa_count}")
    print(f"PDF: {pdf_path}")
    print("Open that file to review the output.")
    print("For a real LinkedIn run, fill LINKEDIN_PROFILE_URL and OPENAI_API_KEY in .env,")
    print("then:  .venv/bin/python -m app.main run")
    return 0


def cmd_check(settings: Settings, _args: argparse.Namespace) -> int:
    """Print whether the machine is ready for a real LinkedIn run."""
    from app.config import configure_runtime_env

    configure_runtime_env()
    settings.ensure_directories()
    print("Readiness check")
    print(f"  Python:                {sys.version.split()[0]}")
    print(f"  Profile URL:           {settings.linkedin_profile_url or 'MISSING — set LINKEDIN_PROFILE_URL in .env'}")
    print(f"  OpenAI API key:        {'set' if settings.openai_api_key else 'MISSING — set OPENAI_API_KEY in .env'}")
    print(f"  Model:                 {settings.openai_model}")
    print(f"  Database:              {settings.database_path}")
    print(f"  PDF output:            {settings.pdf_output_path}")
    print(f"  Browser profile:       {settings.browser_user_data_dir}")

    chromium_ok = False
    chromium_path = ""
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            chromium_path = playwright.chromium.executable_path
            chromium_ok = bool(chromium_path)
    except Exception as exc:
        print(f"  Chromium:              MISSING ({exc})")
    else:
        print(f"  Chromium:              {'ok — ' + chromium_path if chromium_ok else 'MISSING'}")

    ready_for_demo = True
    ready_for_linkedin = bool(settings.linkedin_profile_url and settings.openai_api_key and chromium_ok)
    print()
    print("  Offline demo:          ready  (.venv/bin/python -m app.main demo)")
    print(
        "  LinkedIn + LLM run:    "
        + ("ready  (.venv/bin/python -m app.main run)" if ready_for_linkedin else "not ready until .env is filled")
    )
    return 0 if ready_for_demo else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.main",
        description="Collect LinkedIn interview Q&A and generate a PDF.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect", help="Scrape accessible posts from the configured profile.")
    collect.add_argument("--max-posts", type=int, default=None, help="Stop after this many posts (default: 100).")
    collect.add_argument("--max-scrolls", type=int, default=None)

    extract = sub.add_parser("extract", help="Extract Q&A from stored posts using the LLM.")
    extract.add_argument("--force", action="store_true", help="Re-run extraction for all posts.")
    extract.add_argument("--limit", type=int, default=None, help="Process at most N posts.")

    sub.add_parser("generate-pdf", help="Write data/output/linkedin_interview_qa.pdf")
    sub.add_parser("stats", help="Show collection and extraction counts.")

    run = sub.add_parser("run", help="Run the LangGraph collect → extract → PDF workflow.")
    run.add_argument(
        "--from-db",
        action="store_true",
        help="Skip LinkedIn scraping and process posts already stored in SQLite.",
    )
    run.add_argument("--force", action="store_true", help="With --from-db, include already processed posts.")
    run.add_argument("--limit", type=int, default=None, help="With --from-db, process at most N posts.")
    sub.add_parser("demo", help="Generate a sample PDF from fixture posts (no LinkedIn or API key).")
    sub.add_parser("check", help="Show whether .env, Chromium, and output paths are ready.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = get_settings()
    handlers = {
        "collect": cmd_collect,
        "extract": cmd_extract,
        "generate-pdf": cmd_generate_pdf,
        "stats": cmd_stats,
        "run": cmd_run,
        "demo": cmd_demo,
        "check": cmd_check,
    }
    try:
        return handlers[args.command](settings, args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
