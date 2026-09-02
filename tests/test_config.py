"""Configuration tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from app.config import PROJECT_ROOT, Settings


def test_settings_load_from_env(tmp_settings: Settings) -> None:
    assert tmp_settings.linkedin_profile_url.endswith("example-profile")
    assert tmp_settings.openai_api_key == "test-key"
    assert tmp_settings.database_path.name == "test.db"
    assert tmp_settings.database_url.startswith("sqlite:///")
    assert tmp_settings.max_posts == 100


def test_relative_paths_resolve_under_project_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_PATH", "data/custom.db")
    settings = Settings()
    assert settings.database_path == PROJECT_ROOT / "data" / "custom.db"


def test_profile_url_is_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LINKEDIN_PROFILE_URL", "https://www.linkedin.com/in/foo/")
    settings = Settings()
    assert settings.linkedin_profile_url == "https://www.linkedin.com/in/foo"


def test_require_profile_url_rejects_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LINKEDIN_PROFILE_URL", "")
    settings = Settings()
    with pytest.raises(ValueError, match="LINKEDIN_PROFILE_URL"):
        settings.require_profile_url()


def test_require_openai_key_rejects_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    settings = Settings()
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        settings.require_openai_key()


def test_ensure_directories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "db" / "x.db"))
    monkeypatch.setenv("BROWSER_USER_DATA_DIR", str(tmp_path / "browser"))
    settings = Settings()
    settings.ensure_directories()
    assert settings.database_path.parent.is_dir()
    assert settings.browser_user_data_dir.is_dir()
