"""Playwright tests against a local HTML fixture (no LinkedIn network)."""

from __future__ import annotations

from pathlib import Path

import pytest
from app.linkedin.media import capture_slides_for_posts
from app.linkedin.scraper import expand_see_more, extract_visible_posts

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "feed.html"
CAROUSEL = Path(__file__).resolve().parent / "fixtures" / "carousel.html"


def test_extract_and_expand_local_feed() -> None:
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(FIXTURE.as_uri())
            before = extract_visible_posts(page)
            assert len(before) == 2
            assert before[0].author == "Ada Lovelace"
            assert "What is the GIL?" in before[0].raw_text
            assert before[0].post_url.endswith("urn:li:activity:111")

            clicked = expand_see_more(page)
            assert clicked >= 1
            page.wait_for_timeout(200)
            after = extract_visible_posts(page)
            assert any("Extra details after expanding" in post.raw_text for post in after)
            browser.close()
    except Exception as exc:
        message = str(exc).lower()
        if "executable doesn't exist" in message or "playwright install" in message:
            pytest.skip("Chromium is not installed for Playwright")
        raise


def test_capture_carousel_slides(tmp_path: Path) -> None:
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(CAROUSEL.as_uri())
            posts = extract_visible_posts(page)
            urns = {post.post_urn for post in posts}
            assert "urn:li:activity:carousel" in urns
            assert "urn:li:activity:image-only" in urns
            image_only = next(post for post in posts if post.post_urn == "urn:li:activity:image-only")
            assert image_only.raw_text == "[image post]"
            capture_slides_for_posts(page, posts, tmp_path)
            carousel = next(post for post in posts if post.post_urn == "urn:li:activity:carousel")
            slides = sorted((tmp_path / carousel.content_hash / "slides").glob("slide_*.png"))
            assert len(slides) >= 2
            assert slides[0].read_bytes() != slides[1].read_bytes()
            browser.close()
    except Exception as exc:
        message = str(exc).lower()
        if "executable doesn't exist" in message or "playwright install" in message:
            pytest.skip("Chromium is not installed for Playwright")
        raise
