"""Persistent Playwright context with manual LinkedIn login only."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from playwright.sync_api import BrowserContext, Page, Playwright, sync_playwright

from app.config import Settings, configure_runtime_env

LINKEDIN_HOME = "https://www.linkedin.com/"
LOGIN_URL_HINTS = ("/login", "/checkpoint", "/authwall", "/uas/login")
LOGIN_SELECTORS = (
    'input[name="session_key"]',
    "input#username",
    "form.login__form",
)
AUTHENTICATED_SELECTORS = (
    "#global-nav",
    "nav.global-nav",
    ".global-nav__me",
    "img.global-nav__me-photo",
)


def looks_like_login(page: Page) -> bool:
    url = page.url.lower()
    if any(hint in url for hint in LOGIN_URL_HINTS):
        return True
    for selector in LOGIN_SELECTORS:
        try:
            if page.locator(selector).count() > 0:
                return True
        except Exception:
            continue
    return False


def looks_authenticated(page: Page) -> bool:
    url = page.url.lower()
    if any(hint in url for hint in LOGIN_URL_HINTS):
        return False
    path = url.split("linkedin.com", 1)[-1]
    if any(part in path for part in ("/feed", "/in/", "/recent-activity", "/mynetwork", "/jobs")):
        return True
    for selector in AUTHENTICATED_SELECTORS:
        try:
            if page.locator(selector).count() > 0:
                return True
        except Exception:
            continue
    return False


def wait_for_manual_login(page: Page, timeout_seconds: int) -> None:
    """Block until the user finishes login in the visible browser. Never fills credentials."""
    if looks_authenticated(page):
        return

    print(flush=True)
    print("LinkedIn login required.", flush=True)
    print("Log in manually in the opened browser window.", flush=True)
    print("This tool will not type your password, solve CAPTCHA, or bypass restrictions.", flush=True)
    print(f"Waiting up to {timeout_seconds} seconds...", flush=True)
    print(flush=True)

    deadline = time.time() + timeout_seconds
    last_url = ""
    while time.time() < deadline:
        if looks_authenticated(page):
            print("Login detected. Continuing.", flush=True)
            return
        if page.url != last_url:
            last_url = page.url
            print(f"Waiting for session… current page: {last_url}", flush=True)
        page.wait_for_timeout(1500)

    raise TimeoutError(
        "Timed out waiting for a LinkedIn session. "
        "Run collect again after logging in; the browser profile is reused."
    )


def launch_persistent_context(
    playwright: Playwright,
    user_data_dir: Path,
    *,
    headless: bool,
) -> BrowserContext:
    user_data_dir.mkdir(parents=True, exist_ok=True)
    configure_runtime_env()
    context = playwright.chromium.launch_persistent_context(
        user_data_dir=str(user_data_dir),
        headless=headless,
        viewport={"width": 1400, "height": 900},
        locale="en-US",
        accept_downloads=False,
        args=["--disable-blink-features=AutomationControlled"],
    )
    context.set_default_timeout(30_000)
    return context


@contextmanager
def open_linkedin_session(settings: Settings) -> Iterator[Page]:
    """Open a persistent Chromium profile and yield an authenticated page when possible."""
    with sync_playwright() as playwright:
        print("Launching Chromium...", flush=True)
        context = launch_persistent_context(
            playwright,
            settings.browser_user_data_dir,
            headless=settings.headless,
        )
        try:
            page = context.pages[0] if context.pages else context.new_page()
            print("Chromium is open. Navigating to LinkedIn...", flush=True)
            page.goto(LINKEDIN_HOME, wait_until="domcontentloaded")
            print(f"Loaded {page.url}", flush=True)
            wait_for_manual_login(page, settings.login_timeout_seconds)
            yield page
        finally:
            context.close()
