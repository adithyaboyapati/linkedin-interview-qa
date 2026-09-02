"""Collect accessible LinkedIn posts by scrolling the profile activity feed."""

from __future__ import annotations

from urllib.parse import urlparse, urlunparse

from playwright.sync_api import Page
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from app.extraction.normalizer import content_hash, normalize_text
from app.linkedin.media import capture_slides_for_posts
from app.models import CollectedPost

IMAGE_POST_PLACEHOLDER = "[image post]"

EXTRACT_POSTS_JS = """
() => {
  const selectors = [
    'div.feed-shared-update-v2[data-urn]',
    'div[data-urn^="urn:li:activity"]',
    'div[data-urn^="urn:li:ugcPost"]',
    'div[data-urn^="urn:li:share"]',
  ];
  const cards = [];
  const seenNodes = new Set();
  for (const selector of selectors) {
    for (const node of document.querySelectorAll(selector)) {
      if (seenNodes.has(node)) continue;
      seenNodes.add(node);
      cards.push(node);
    }
  }

  const results = [];
  const seenUrns = new Set();
  for (const card of cards) {
    const urn = (card.getAttribute('data-urn') || '').trim();
    if (urn && seenUrns.has(urn)) continue;
    if (urn) seenUrns.add(urn);

    const textNode =
      card.querySelector('.update-components-text') ||
      card.querySelector('.feed-shared-update-v2__description') ||
      card.querySelector('.update-components-update-v2__commentary') ||
      card.querySelector('.feed-shared-inline-show-more-text') ||
      card.querySelector('.break-words');
    const images = card.querySelectorAll(
      '.update-components-image img, .update-components-carousel img, img.update-components-image__image'
    );
    let rawText = (textNode ? textNode.innerText : '').trim();
    if (!rawText && images.length === 0) continue;
    if (!rawText) rawText = '[image post]';

    const authorNode =
      card.querySelector('.update-components-actor__title') ||
      card.querySelector('.update-components-actor__name') ||
      card.querySelector('span.update-components-actor__title span[aria-hidden="true"]');
    let author = authorNode ? authorNode.innerText.trim() : null;
    if (author) {
      author = author.split('\\n')[0].replace(/\\s+•.*$/, '').trim();
    }

    const timeNode =
      card.querySelector('a.update-components-actor__sub-description-link') ||
      card.querySelector('span.update-components-actor__sub-description') ||
      card.querySelector('time');
    let posted = timeNode ? timeNode.innerText.trim() : null;
    if (posted) {
      posted = posted.split('\\n')[0].replace(/\\s+•.*$/, '').trim();
    }

    const link =
      card.querySelector('a[href*="/feed/update/"]') ||
      card.querySelector('a[href*="/posts/"]') ||
      card.querySelector('a.app-aware-link[href*="activity"]');
    let url = link ? link.href : null;
    if (!url && urn) {
      url = 'https://www.linkedin.com/feed/update/' + urn;
    }

    results.push({
      post_urn: urn || null,
      post_url: url,
      author: author,
      posted_at_text: posted,
      raw_text: rawText,
    });
  }
  return results;
}
"""

SEE_MORE_SELECTORS = (
    "button.feed-shared-inline-show-more-text__see-more-less-toggle",
    'button[aria-label*="see more" i]',
    'button:has-text("see more")',
    'button:has-text("See more")',
    'button:has-text("…more")',
)


def profile_activity_urls(profile_url: str) -> list[str]:
    parsed = urlparse(profile_url.strip())
    path = parsed.path.rstrip("/")
    if not path.endswith("/recent-activity/shares") and "/recent-activity/" not in path:
        shares = urlunparse(parsed._replace(path=f"{path}/recent-activity/shares/", query="", fragment=""))
        all_activity = urlunparse(
            parsed._replace(path=f"{path}/recent-activity/all/", query="", fragment="")
        )
        return [shares, all_activity]
    return [urlunparse(parsed._replace(query="", fragment=""))]


def payload_to_post(payload: dict) -> CollectedPost | None:
    text = normalize_text(str(payload.get("raw_text") or ""))
    if not text:
        return None
    urn = (payload.get("post_urn") or None) or None
    url = (payload.get("post_url") or None) or None
    author = (payload.get("author") or None) or None
    posted = (payload.get("posted_at_text") or None) or None
    if author:
        author = normalize_text(author)
    if posted:
        posted = normalize_text(posted)
    if url:
        url = url.split("?")[0]
    hash_source = f"{urn}\n{text}" if text == IMAGE_POST_PLACEHOLDER and urn else text
    return CollectedPost(
        post_urn=urn if urn else None,
        post_url=url,
        author=author,
        posted_at_text=posted,
        raw_text=text,
        content_hash=content_hash(hash_source),
    )


def expand_see_more(page: Page, limit: int = 40) -> int:
    clicked = 0
    for selector in SEE_MORE_SELECTORS:
        buttons = page.locator(selector)
        try:
            count = min(buttons.count(), limit)
        except Exception:
            continue
        for index in range(count):
            try:
                button = buttons.nth(index)
                if not button.is_visible():
                    continue
                button.click(timeout=800, force=True)
                clicked += 1
            except Exception:
                continue
        if clicked:
            break
    return clicked


def extract_visible_posts(page: Page) -> list[CollectedPost]:
    payloads = page.evaluate(EXTRACT_POSTS_JS)
    posts: list[CollectedPost] = []
    for payload in payloads or []:
        post = payload_to_post(payload)
        if post is not None:
            posts.append(post)
    return posts


def _blocked_by_linkedin(page: Page) -> str | None:
    url = page.url.lower()
    if "authwall" in url or "/signup" in url:
        return "LinkedIn is showing an authentication wall for this page."
    body = ""
    try:
        body = (page.inner_text("body") or "").lower()
    except Exception:
        return None
    if "security check" in body and "captcha" in body:
        return "LinkedIn is showing a security check. Complete it in the browser if you want, then retry."
    return None


def open_activity_feed(page: Page, profile_url: str) -> str:
    last_error: str | None = None
    for url in profile_activity_urls(profile_url):
        try:
            page.goto(url, wait_until="domcontentloaded")
            page.wait_for_timeout(1500)
        except PlaywrightTimeoutError:
            last_error = f"Timed out opening {url}"
            continue
        blocked = _blocked_by_linkedin(page)
        if blocked:
            last_error = blocked
            continue
        _open_posts_tab(page)
        return page.url
    raise RuntimeError(last_error or "Could not open the profile activity page.")


def _open_posts_tab(page: Page) -> None:
    """Prefer the Posts filter so comments/reactions do not bury original posts."""
    selectors = (
        'button:has-text("Posts")',
        'a:has-text("Posts")',
        '[role="tab"]:has-text("Posts")',
        'button.artdeco-pill:has-text("Posts")',
    )
    for selector in selectors:
        loc = page.locator(selector)
        try:
            if loc.count() == 0:
                continue
            loc.first.click(timeout=2000)
            page.wait_for_timeout(1200)
            print("Switched to Posts filter.", flush=True)
            return
        except Exception:
            continue


def collect_posts(
    page: Page,
    profile_url: str,
    *,
    max_scrolls: int,
    scroll_pause_ms: int,
    max_idle_scrolls: int,
    max_posts: int = 100,
    on_batch=None,
    raw_dump_dir=None,
) -> list[CollectedPost]:
    """Scroll the activity feed and capture posts as they appear.

    LinkedIn virtualizes the feed, so posts are extracted during scrolling
    rather than only at the end.
    """
    print(f"Opening activity feed for {profile_url}", flush=True)
    open_activity_feed(page, profile_url)
    print(f"Activity page: {page.url}", flush=True)

    collected: dict[str, CollectedPost] = {}
    idle = 0
    limit = max(1, max_posts)

    for scroll in range(max(1, max_scrolls)):
        print(f"Scrolling ({scroll + 1}/{max_scrolls})… {len(collected)}/{limit} post(s) so far", flush=True)
        expand_see_more(page)
        page.wait_for_timeout(400)
        batch = extract_visible_posts(page)
        new_posts: list[CollectedPost] = []
        for post in batch:
            if len(collected) >= limit:
                break
            key = post.post_urn or post.content_hash
            if key in collected:
                continue
            collected[key] = post
            new_posts.append(post)

        if new_posts and raw_dump_dir is not None:
            capture_slides_for_posts(page, new_posts, raw_dump_dir)

        if on_batch and new_posts:
            on_batch(new_posts)

        if len(collected) >= limit:
            print(f"Reached {limit} posts, stopping.", flush=True)
            break

        if new_posts:
            idle = 0
        else:
            idle += 1
            if idle >= max_idle_scrolls:
                break

        page.evaluate("window.scrollBy(0, Math.max(window.innerHeight, 800))")
        page.wait_for_timeout(scroll_pause_ms)

    return list(collected.values())
