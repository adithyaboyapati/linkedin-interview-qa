"""Assemble the LangGraph workflow.

Concepts used here:
  State, Nodes, Edges, Conditional Edges, Tools, Structured LLM output,
  Retry, Human-in-the-loop (interrupt), Checkpointing.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from app.config import Settings
from app.extraction.qa_extractor import QAExtractor
from app.extraction.vision import SlideTranscriber
from app.graph.nodes import (
    deduplicate,
    make_classify_posts_node,
    make_collect_posts_node,
    make_extract_qa_node,
    make_generate_pdf_node,
    make_save_to_db_node,
    route_after_validate,
    validate_qa,
)
from app.graph.state import CollectorState
from app.graph.tools import make_collect_posts_tool
from app.models import CollectedPost
from app.storage.database import init_db, make_engine, make_session_factory
from app.storage.repository import Repository

logger = logging.getLogger("app.graph")


def _checkpointer():
    try:
        from langgraph.checkpoint.memory import InMemorySaver as Saver
    except ImportError:  # pragma: no cover - older langgraph
        from langgraph.checkpoint.memory import MemorySaver as Saver
    return Saver()


def build_graph(
    *,
    settings: Settings,
    extractor: QAExtractor,
    repo: Repository,
    collect_fn: Callable[[], list[CollectedPost]] | None = None,
    checkpointer=None,
    transcriber: SlideTranscriber | None = None,
):
    """Compile the Q&A collector graph. SQLite and PDF stay in their modules."""
    collect_tool = make_collect_posts_tool(settings, collect_fn=collect_fn)
    builder = StateGraph(CollectorState)
    builder.add_node("collect_posts", make_collect_posts_node(collect_tool))
    builder.add_node(
        "classify_posts",
        make_classify_posts_node(
            extractor,
            transcriber=transcriber,
            raw_dump_dir=settings.raw_dump_dir,
        ),
    )
    builder.add_node("extract_qa", make_extract_qa_node(extractor))
    builder.add_node("validate_qa", validate_qa)
    builder.add_node("deduplicate", deduplicate)
    builder.add_node("save_to_db", make_save_to_db_node(repo))
    builder.add_node("generate_pdf", make_generate_pdf_node(settings, repo))

    builder.add_edge(START, "collect_posts")
    builder.add_edge("collect_posts", "classify_posts")
    builder.add_edge("classify_posts", "extract_qa")
    builder.add_edge("extract_qa", "validate_qa")
    builder.add_conditional_edges(
        "validate_qa",
        route_after_validate,
        {
            "extract_qa": "extract_qa",
            "deduplicate": "deduplicate",
        },
    )
    builder.add_edge("deduplicate", "save_to_db")
    builder.add_edge("save_to_db", "generate_pdf")
    builder.add_edge("generate_pdf", END)

    return builder.compile(checkpointer=checkpointer or _checkpointer())


def _pending_interrupts(graph, config) -> list:
    snapshot = graph.get_state(config)
    interrupts = list(getattr(snapshot, "interrupts", None) or [])
    for task in getattr(snapshot, "tasks", None) or []:
        interrupts.extend(getattr(task, "interrupts", None) or [])
    return interrupts


def invoke_with_login_interrupt(graph, initial: CollectorState, *, thread_id: str | None = None) -> dict:
    """Run the graph, pausing for manual LinkedIn login when collect_posts interrupts."""
    config = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}
    result = graph.invoke(initial, config)
    while _pending_interrupts(graph, config):
        payload = _pending_interrupts(graph, config)[0]
        value = getattr(payload, "value", payload)
        message = value.get("message") if isinstance(value, dict) else str(value)
        print()
        print(message)
        print()
        input("Press Enter after you have logged in to LinkedIn...")
        result = graph.invoke(Command(resume=True), config)
    return result


def open_repository(settings: Settings) -> Repository:
    settings.ensure_directories()
    engine = make_engine(settings.database_path)
    init_db(engine)
    return Repository(make_session_factory(engine)(), raw_dump_dir=settings.raw_dump_dir)


def run_workflow(
    settings: Settings,
    *,
    skip_scrape: bool = False,
    posts: list[CollectedPost] | None = None,
    extractor: QAExtractor | None = None,
    handle_login: bool = True,
    transcriber: SlideTranscriber | None = None,
) -> dict:
    """Orchestrate collect → classify → extract → validate → save → PDF."""
    repo = open_repository(settings)
    extractor = extractor or QAExtractor(
        api_key=settings.require_openai_key(),
        base_url=settings.openai_base_url,
        model=settings.openai_model,
    )
    if transcriber is None and isinstance(extractor, QAExtractor) and settings.openai_api_key:
        transcriber = SlideTranscriber(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.openai_model,
        )
    graph = build_graph(
        settings=settings, extractor=extractor, repo=repo, transcriber=transcriber
    )

    initial: CollectorState = {
        "posts": [post.model_dump() for post in posts or []],
        "interview_posts": [],
        "qa_pairs": [],
        "validation_results": [],
        "extract_attempts": 0,
        "skipped_posts": [],
        "pdf_path": "",
        "skip_scrape": skip_scrape,
        "logs": [],
    }
    if skip_scrape or not handle_login:
        result = graph.invoke(initial, {"configurable": {"thread_id": str(uuid.uuid4())}})
    else:
        result = invoke_with_login_interrupt(graph, initial)
    repo.session.close()
    return result
