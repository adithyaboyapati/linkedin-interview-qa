"""Read interview text from post images using a vision-capable LLM."""

from __future__ import annotations

import base64
import re
from collections.abc import Callable
from pathlib import Path

from openai import OpenAI

from app.extraction.normalizer import normalize_text

TRANSCRIBE_PROMPT = (
    "This is a LinkedIn post image, often one slide in a carousel. "
    "If the image contains interview questions and/or answers, transcribe that text verbatim. "
    "Preserve questions, answers, bullet points, and scenario prompts even if they lack a '?'. "
    "If the image is only a title card, logo, CTA, or otherwise has no interview Q&A, "
    "return an empty string. Do not summarize, translate, or invent missing text."
)

_NUMBERED_ITEM_RE = re.compile(r"(?m)^\s*(?:\d+[.)]|[-*•])\s+\S")
_QUESTION_CUES = (
    "?",
    "q:",
    "a:",
    "how would you",
    "how do you",
    "how can you",
    "what is",
    "what are",
    "what would",
    "explain",
    "difference between",
    "design a",
    "implement",
    "they asked",
    "asked me",
    "interview question",
)


def looks_like_interview_qa(text: str) -> bool:
    """True when transcribed image text looks like interview questions or answers."""
    lowered = normalize_text(text).lower()
    if len(lowered) < 20:
        return False
    if any(cue in lowered for cue in _QUESTION_CUES):
        return True
    return bool(_NUMBERED_ITEM_RE.search(text))


def filter_interview_image_text(image_text: str) -> str:
    """Keep only slide chunks that contain interview Q&A; drop title cards and CTAs."""
    text = (image_text or "").strip()
    if not text:
        return ""
    if "[Slide " not in text:
        return text if looks_like_interview_qa(text) else ""
    chunks: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.startswith("[Slide ") and current:
            block = "\n".join(current).strip()
            if looks_like_interview_qa(block):
                chunks.append(block)
            current = [line]
        else:
            current.append(line)
    if current:
        block = "\n".join(current).strip()
        if looks_like_interview_qa(block):
            chunks.append(block)
    return "\n\n".join(chunks)


VisionFn = Callable[[Path], str]


def slide_paths(raw_dump_dir: Path | None, content_hash: str) -> list[Path]:
    if raw_dump_dir is None:
        return []
    folder = raw_dump_dir / content_hash / "slides"
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})


def combined_source_text(caption: str, image_text: str = "") -> str:
    parts = [normalize_text(caption)]
    extra = normalize_text(filter_interview_image_text(image_text))
    if extra:
        parts.append("--- TEXT FROM POST IMAGES ---")
        parts.append(extra)
    return "\n\n".join(part for part in parts if part)


def source_text_for_post(
    *,
    caption: str,
    image_text: str = "",
    content_hash: str = "",
    dump_dir: Path | None = None,
    transcriber: SlideTranscriber | None = None,
    force: bool = False,
) -> tuple[str, str]:
    """Merge caption + interview Q&A from slides. Decorative images are dropped."""
    text = filter_interview_image_text(image_text or "")
    paths = slide_paths(dump_dir, content_hash)
    if transcriber is not None and paths and (force or not text.strip()):
        text = transcriber.transcribe_directory(paths)
    text = filter_interview_image_text(text)
    return combined_source_text(caption, text), text


class SlideTranscriber:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        transcribe_fn: VisionFn | None = None,
    ) -> None:
        self.model = model
        self._transcribe_fn = transcribe_fn
        self._client = None if transcribe_fn else OpenAI(api_key=api_key, base_url=base_url)

    def transcribe_directory(self, paths: list[Path]) -> str:
        chunks: list[str] = []
        skipped = 0
        for path in paths:
            text = self.transcribe_image(path).strip()
            if text and looks_like_interview_qa(text):
                chunks.append(f"[Slide {path.name}]\n{text}")
            elif text:
                skipped += 1
        if skipped:
            print(f"  skipped {skipped} image(s) with no interview Q&A", flush=True)
        return "\n\n".join(chunks)

    def transcribe_image(self, path: Path) -> str:
        if self._transcribe_fn is not None:
            return self._transcribe_fn(path)
        assert self._client is not None
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
        mime = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=4000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": TRANSCRIBE_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{payload}"},
                        },
                    ],
                }
            ],
        )
        return (response.choices[0].message.content or "").strip()
