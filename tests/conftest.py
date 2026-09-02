"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings, clear_settings_cache, configure_runtime_env

configure_runtime_env()


@pytest.fixture
def tmp_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    clear_settings_cache()
    monkeypatch.setenv("LINKEDIN_PROFILE_URL", "https://www.linkedin.com/in/example-profile")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("BROWSER_USER_DATA_DIR", str(tmp_path / "browser"))
    monkeypatch.setenv("RAW_DUMP_DIR", str(tmp_path / "raw"))
    monkeypatch.setenv("PDF_OUTPUT_PATH", str(tmp_path / "output" / "linkedin_interview_qa.pdf"))
    monkeypatch.setenv("MAX_POSTS", "100")
    settings = Settings()
    settings.ensure_directories()
    yield settings
    clear_settings_cache()
