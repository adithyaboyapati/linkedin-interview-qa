"""Capture LinkedIn post images and carousel slides while they are on screen."""

from __future__ import annotations

import hashlib
from pathlib import Path

from playwright.sync_api import Locator, Page
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from app.models import CollectedPost

CONTENT_IMAGE_SELECTOR = (
    ".update-components-image img, "
    ".update-components-carousel img, "
    ".feed-shared-image img, "
    "img.update-components-image__image"
)
NEXT_SELECTORS = (
    'button[aria-label*="Next image" i]',
    'button[aria-label*="Next photo" i]',
    'button[aria-label*="Next slide" i]',
    'button[aria-label="Next"]',
    "button.artdeco-carousel__button--next",
    "button.artdeco-carousel__button.artdeco-button--next",
    "button.carousel-slide-container__next-btn",
)
MAX_SLIDES = 16


def slide_dir(raw_dump_dir: Path, content_hash: str) -> Path:
    return raw_dump_dir / content_hash / "slides"


def capture_slides_for_posts(page: Page, posts: list[CollectedPost], raw_dump_dir: Path) -> None:
    """Screenshot each visible post image, clicking through carousels when present."""
    if not posts:
        return
    raw_dump_dir.mkdir(parents=True, exist_ok=True)
    for post in posts:
        if not post.post_urn:
            continue
        card = page.locator(f'div.feed-shared-update-v2[data-urn="{post.post_urn}"]').first
        try:
            if card.count() == 0:
                continue
        except Exception:
            continue
        dest = slide_dir(raw_dump_dir, post.content_hash)
        saved = _capture_card_slides(page, card, dest)
        if saved:
            print(f"  saved {saved} image slide(s) for {post.post_urn}", flush=True)


def _capture_card_slides(page: Page, card: Locator, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    saved = 0
    for index in range(MAX_SLIDES):
        image = card.locator(CONTENT_IMAGE_SELECTOR).first
        try:
            if image.count() == 0:
                break
            png = image.screenshot(type="png")
        except Exception:
            break
        digest = hashlib.sha256(png).hexdigest()
        if digest in seen:
            break
        seen.add(digest)
        (dest / f"slide_{index:02d}.png").write_bytes(png)
        saved += 1
        try:
            image.hover(timeout=500)
        except Exception:
            pass
        if not _click_next_slide(card, page):
            break
    return saved


def _click_next_slide(card: Locator, page: Page) -> bool:
    for selector in NEXT_SELECTORS:
        button = card.locator(selector).first
        try:
            if button.count() == 0 or not button.is_enabled():
                continue
            button.click(timeout=800, force=True)
            page.wait_for_timeout(350)
            return True
        except (PlaywrightTimeoutError, Exception):
            continue
    return False
