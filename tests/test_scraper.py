"""Scraper helper tests that do not require LinkedIn."""

from __future__ import annotations

from app.linkedin.browser import looks_authenticated, looks_like_login
from app.linkedin.scraper import payload_to_post, profile_activity_urls


class _FakeLocator:
    def __init__(self, count: int) -> None:
        self._count = count

    def count(self) -> int:
        return self._count


class _FakePage:
    def __init__(self, url: str, selectors: dict[str, int]) -> None:
        self.url = url
        self._selectors = selectors

    def locator(self, selector: str) -> _FakeLocator:
        return _FakeLocator(self._selectors.get(selector, 0))


def test_profile_activity_urls_appends_shares_and_all() -> None:
    urls = profile_activity_urls("https://www.linkedin.com/in/example-profile")
    assert urls[0].endswith("/recent-activity/shares/")
    assert urls[1].endswith("/recent-activity/all/")


def test_payload_to_post_normalizes_and_hashes() -> None:
    post = payload_to_post(
        {
            "post_urn": "urn:li:activity:123",
            "post_url": "https://www.linkedin.com/feed/update/urn:li:activity:123?trk=x",
            "author": "Ada Lovelace\nVerified",
            "posted_at_text": "2d • Edited",
            "raw_text": "What is the GIL? See more",
        }
    )
    assert post is not None
    assert post.post_url == "https://www.linkedin.com/feed/update/urn:li:activity:123"
    assert "See more" not in post.raw_text
    assert post.content_hash


def test_payload_to_post_rejects_empty_text() -> None:
    assert payload_to_post({"raw_text": "   "}) is None


def test_payload_to_post_keeps_image_placeholder_and_unique_hashes() -> None:
    first = payload_to_post({"raw_text": "[image post]", "post_urn": "urn:li:activity:1"})
    second = payload_to_post({"raw_text": "[image post]", "post_urn": "urn:li:activity:2"})
    assert first is not None and second is not None
    assert first.raw_text == "[image post]"
    assert first.content_hash != second.content_hash


def test_login_detection_from_url_and_form() -> None:
    login_page = _FakePage("https://www.linkedin.com/login", {})
    assert looks_like_login(login_page)
    assert not looks_authenticated(login_page)

    home = _FakePage("https://www.linkedin.com/feed/", {})
    assert looks_authenticated(home)
    assert not looks_like_login(home)
