"""Application configuration loaded from environment / .env."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLAYWRIGHT_BROWSERS_DIR = PROJECT_ROOT / "data" / "ms-playwright"


def configure_runtime_env() -> None:
    """Point Playwright at the project-local Chromium install when present."""
    if PLAYWRIGHT_BROWSERS_DIR.exists():
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(PLAYWRIGHT_BROWSERS_DIR)
    else:
        os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(PLAYWRIGHT_BROWSERS_DIR))


class Settings(BaseSettings):
    """Runtime settings. Values come from environment variables or `.env`."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    linkedin_profile_url: str = Field(
        default="",
        description="LinkedIn profile URL whose accessible posts should be collected.",
    )
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    database_path: Path = Path("data/linkedin_qa.db")
    browser_user_data_dir: Path = Path("data/browser_profile")
    max_posts: int = 100
    max_scrolls: int = 80
    scroll_pause_ms: int = 2000
    max_idle_scrolls: int = 3
    headless: bool = False
    login_timeout_seconds: int = 300
    raw_dump_dir: Path = Path("data/raw")
    pdf_output_path: Path = Path("data/output/linkedin_interview_qa.pdf")

    @field_validator("linkedin_profile_url")
    @classmethod
    def _strip_url(cls, value: str) -> str:
        return (value or "").strip().rstrip("/")

    @field_validator(
        "database_path",
        "browser_user_data_dir",
        "raw_dump_dir",
        "pdf_output_path",
        mode="before",
    )
    @classmethod
    def _resolve_path(cls, value: str | Path) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    @property
    def database_url(self) -> str:
        return f"sqlite:///{self.database_path}"

    def ensure_directories(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.browser_user_data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dump_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_output_path.parent.mkdir(parents=True, exist_ok=True)

    def require_profile_url(self) -> str:
        if not self.linkedin_profile_url:
            raise ValueError(
                "LINKEDIN_PROFILE_URL is not set. Copy .env.example to .env and add the profile URL."
            )
        if "linkedin.com" not in self.linkedin_profile_url:
            raise ValueError("LINKEDIN_PROFILE_URL must be a LinkedIn profile URL.")
        return self.linkedin_profile_url

    def require_openai_key(self) -> str:
        if not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add an OpenAI-compatible key."
            )
        return self.openai_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    configure_runtime_env()
    settings = Settings()
    settings.ensure_directories()
    return settings


def clear_settings_cache() -> None:
    get_settings.cache_clear()
