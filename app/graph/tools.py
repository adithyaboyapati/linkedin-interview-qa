"""Tools used by graph nodes. These wrap existing modules; they are not agents."""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.tools import StructuredTool

from app.config import Settings
from app.linkedin.browser import open_linkedin_session
from app.linkedin.scraper import collect_posts
from app.models import CollectedPost


def collect_accessible_posts(settings: Settings) -> list[CollectedPost]:
    """Collect accessible posts from the configured LinkedIn profile.

    Opens a persistent Playwright session. If LinkedIn shows a login page, the
    user must sign in manually. Passwords and CAPTCHA are never automated.
    """
    profile_url = settings.require_profile_url()
    print(f"Collecting accessible posts from {profile_url}", flush=True)
    with open_linkedin_session(settings) as page:
        return collect_posts(
            page,
            profile_url,
            max_scrolls=settings.max_scrolls,
            scroll_pause_ms=settings.scroll_pause_ms,
            max_idle_scrolls=settings.max_idle_scrolls,
            max_posts=settings.max_posts,
            raw_dump_dir=settings.raw_dump_dir,
        )


def make_collect_posts_tool(
    settings: Settings,
    collect_fn: Callable[[], list[CollectedPost]] | None = None,
) -> StructuredTool:
    """Playwright collection exposed as a LangGraph/LangChain tool."""

    def _run() -> list[dict]:
        posts = collect_fn() if collect_fn is not None else collect_accessible_posts(settings)
        return [post.model_dump() for post in posts]

    return StructuredTool.from_function(
        func=_run,
        name="collect_linkedin_posts",
        description=(
            "Collect accessible LinkedIn posts using Playwright. "
            "Requires a manually authenticated persistent browser profile."
        ),
    )
