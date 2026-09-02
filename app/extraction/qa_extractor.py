"""Extract interview Q&A from a post using an OpenAI-compatible LLM."""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from openai import OpenAI
from pydantic import ValidationError

from app.extraction.normalizer import normalize_for_hash, normalize_text
from app.models import ExtractionResult, PostClassification, QAPairDraft

CLASSIFY_PROMPT = """Decide whether a LinkedIn post is interview-related.

Return JSON only:
{
  "is_interview_related": boolean,
  "reason": string
}

A post is interview-related if it shares interview questions, answers, hiring rounds,
or interview experiences that include technical or behavioral questions.
Job-change announcements, congratulations, and unrelated commentary are not interview-related.
"""

SYSTEM_PROMPT = """You extract interview questions and answers from a LinkedIn post.

Return JSON only, matching this schema:
{
  "is_interview_related": boolean,
  "reason": string,
  "qa_pairs": [
    {
      "question": string,
      "answer": string or null,
      "answered": boolean,
      "category": string
    }
  ]
}

Rules:
1. A post is interview-related if it shares interview questions, answers, hiring rounds, or interview experiences with technical/behavioral questions.
2. If the post is not interview-related, set is_interview_related to false and qa_pairs to [].
3. Extract only questions that actually appear in the post, including Q/A, "asked me", numbered lists, or similar formats. If the post includes a section titled TEXT FROM POST IMAGES, treat that transcribed slide text as part of the post.
4. Copy answers from the post with only trivial whitespace cleanup. Do not paraphrase, expand, correct, or invent.
5. Never use outside knowledge. If the post does not contain an answer, set answered=false and answer=null.
6. Do not include unanswered questions unless they are clearly asked in the post; they will be dropped from the final PDF anyway.
7. category must be one of: Python, SQL, DSA, System Design, ML, GenAI, RAG, AWS, Java, JavaScript, DevOps, Behavioral, Other.
8. One post may contain multiple Q&A pairs.
9. Extract EVERY question that has an answer in the post. Do not stop after the first pair.
10. Keep the original question wording from the post whenever it is present.
11. If the post only lists questions and does not contain answers, return an empty qa_pairs list.
12. For "decoded" or themed posts, treat each heading plus the explanation under it as one Q&A pair and copy the explanation as the answer.
"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_WORD_RE = re.compile(r"[a-z0-9]{3,}")

INTERVIEW_HINTS = (
    "interview",
    "interviewer",
    "asked me",
    "they asked",
    "question",
    "q:",
    "a:",
    "round 1",
    "round 2",
    "technical round",
    "coding round",
    "system design",
    "hiring",
    "recruiter",
    "what is",
    "difference between",
    "explain",
    "how would you",
    "scenario based",
    "swipe through",
)


class JsonCompleter(Protocol):
    def __call__(self, messages: list[dict[str, str]]) -> str: ...


def looks_interview_related(text: str) -> bool:
    lowered = normalize_text(text).lower()
    return any(hint in lowered for hint in INTERVIEW_HINTS)


def answer_is_grounded(answer: str, post_text: str) -> bool:
    """Require the answer to come from the post. Drop likely hallucinations."""
    answer_norm = normalize_for_hash(answer)
    post_norm = normalize_for_hash(post_text)
    if not answer_norm:
        return False
    if answer_norm in post_norm:
        return True
    words = _WORD_RE.findall(answer_norm)
    if not words:
        return False
    present = sum(1 for word in words if word in post_norm)
    return (present / len(words)) >= 0.85


def _parse_json_content(content: str) -> dict[str, Any]:
    text = (content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_RE.search(text)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("LLM output is not a JSON object.")
    return data


def ground_result(result: ExtractionResult, post_text: str) -> ExtractionResult:
    grounded: list[QAPairDraft] = []
    for pair in result.qa_pairs:
        answer = pair.answer
        if not pair.answered or not answer:
            continue
        if not answer_is_grounded(answer, post_text):
            continue
        grounded.append(
            pair.model_copy(
                update={
                    "answered": True,
                    "answer": answer.strip(),
                    "question": pair.question.strip(),
                }
            )
        )
    is_related = result.is_interview_related or bool(grounded)
    return result.model_copy(update={"qa_pairs": grounded, "is_interview_related": is_related})


class QAExtractor:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        completer: JsonCompleter | None = None,
    ) -> None:
        self.model = model
        self._completer = completer
        self._client = None if completer else OpenAI(api_key=api_key, base_url=base_url)

    def classify(self, post_text: str) -> PostClassification:
        messages = [
            {"role": "system", "content": CLASSIFY_PROMPT},
            {"role": "user", "content": f"Classify this LinkedIn post:\n\n{post_text}"},
        ]
        raw = self._complete(messages)
        try:
            return PostClassification.model_validate(raw)
        except ValidationError as exc:
            raise ValueError(f"LLM classification failed validation: {exc}") from exc

    def extract(self, post_text: str, *, ground: bool = True) -> ExtractionResult:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Extract interview Q&A from this LinkedIn post. "
                    "Use only the post text.\n\n"
                    f"{post_text}"
                ),
            },
        ]
        raw = self._complete(messages)
        try:
            result = ExtractionResult.model_validate(raw)
        except ValidationError as exc:
            raise ValueError(f"LLM output failed validation: {exc}") from exc
        if ground:
            return ground_result(result, post_text)
        return result

    def _complete(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        if self._completer is not None:
            content = self._completer(messages)
            return _parse_json_content(content)

        assert self._client is not None
        last_error: Exception | None = None
        for _ in range(2):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0,
                    max_tokens=8000,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content or ""
                return _parse_json_content(content)
            except Exception as exc:  # noqa: BLE001 - retry once on parse/API errors
                last_error = exc
        raise ValueError(f"LLM extraction failed: {last_error}") from last_error
