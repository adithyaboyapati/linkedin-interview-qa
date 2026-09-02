# LinkedIn Interview Q&A Collector

Collect accessible posts from a LinkedIn profile, extract interview questions and answers with an OpenAI-compatible LLM, and generate a structured PDF.

This tool does **not** automate passwords, CAPTCHA, or LinkedIn restriction bypasses. You log in once in a persistent Playwright browser window. Later runs reuse that session.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
cp .env.example .env
```

Chromium is installed into `data/ms-playwright` when you use:

```bash
PLAYWRIGHT_BROWSERS_PATH=data/ms-playwright playwright install chromium
```

Edit `.env`:

```env
LINKEDIN_PROFILE_URL=https://www.linkedin.com/in/the-profile/
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

`OPENAI_BASE_URL` can point at any OpenAI-compatible API.

WeasyPrint is used for PDF output when its system libraries are available. If WeasyPrint cannot import, the generator falls back to fpdf2 automatically.

On macOS, WeasyPrint typically needs:

```bash
brew install pango gdk-pixbuf libffi
```

## Commands

```bash
python -m app.main check
python -m app.main demo
python -m app.main collect
python -m app.main extract
python -m app.main generate-pdf
python -m app.main stats
python -m app.main run
python -m app.main run --from-db
```

Start with `check` then `demo`. `demo` generates a sample PDF from fixture posts and does not need LinkedIn or an API key.

Useful flags:

```bash
python -m app.main collect --max-posts 100
python -m app.main extract --limit 10
python -m app.main extract --force
```

The PDF is written to:

```text
data/output/linkedin_interview_qa.pdf
```

## Workflow

1. `collect` opens Chromium with a persistent profile under `data/browser_profile`. Log in manually if needed, then the scraper opens the profile's recent activity and scrolls until it has 100 accessible posts (configurable via `MAX_POSTS`).
2. Raw posts are stored in SQLite (`data/linkedin_qa.db`) and mirrored as JSON under `data/raw/`. Extraction can be rerun without scraping again.
3. `extract` sends posts to the LLM with structured JSON output. Non-interview posts are skipped. Image slides with no interview Q&A are dropped. Answers that are not grounded in the remaining post text (caption plus transcribed interview slides) are dropped so the model cannot silently invent them.
4. Posts and Q&A pairs are deduplicated. Later runs only insert new content.
5. `generate-pdf` groups answered Q&A by category and writes the interview-prep PDF.

Carousel and image posts: many interview questions live in LinkedIn image slides, not in the caption. `collect` screenshots every visible post image (including carousel slides) under `data/raw/<content_hash>/slides/`. `extract` then uses a vision model to transcribe interview Q&A from those images and merges it with the caption. Title cards, logos, and other slides with no questions or answers are skipped. Re-run `collect` so slides are on disk, then `extract --force` to pick up posts that were previously stored as caption-only.

## LangGraph workflow

`python -m app.main run` orchestrates the same scraper, SQLite store, LLM extractor, and PDF generator through a single LangGraph:

```text
START → collect_posts → classify_posts → extract_qa → validate_qa
          ↳ invalid? retry extract_qa once, then skip the post
       → deduplicate → save_to_db → generate_pdf → END
```

The graph is the orchestrator. It does not reimplement scraping, storage, or PDF output.

| Concept | Where it lives |
| --- | --- |
| State | `app/graph/state.py` (`CollectorState`) |
| Nodes / edges | `app/graph/nodes.py`, `app/graph/workflow.py` |
| Conditional edge | after `validate_qa`: valid → `deduplicate`, invalid → retry `extract_qa` once |
| Tool | Playwright collector in `app/graph/tools.py` |
| Structured LLM output | `PostClassification` and `ExtractionResult` |
| Human-in-the-loop | `interrupt()` in `collect_posts` for manual LinkedIn login |
| Checkpointing | in-memory LangGraph checkpointer (required for interrupts) |

`--from-db` skips scraping and runs classify/extract/PDF on posts already stored by `collect`.

## Tests

```bash
pytest
```

Tests cover configuration, models, SQLite persistence, deduplication, scraper helpers, LLM grounding, PDF generation, and CLI wiring. They do not log in to LinkedIn or call a live LLM.
