"""Minimal structured-output helper: validate + retry.

The caller is responsible for telling the LLM the desired JSON schema in the
system/user prompt.  This module only does three things:

1. Call the LLM.
2. Try to parse the reply with a Pydantic model.
3. If parsing fails, append the ValidationError to the context and retry.
"""

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Type, TypeVar
from openai import APIConnectionError, APIStatusError, AsyncOpenAI
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


@dataclass
class GenerateAttempt:
    """Per-LLM-call record exposed via ``StructuredGenerator.last_attempts``.

    ``messages_sent`` is the full messages list (system + user + any
    retry-correction turns) that was sent for this attempt; ``response_text``
    is the raw assistant content; ``duration`` is wall-clock of the API call.
    Callers (e.g. NodeMerger) sum these across all attempts to attribute
    the *real* token / time cost of one ``generate()`` invocation, including
    retries that the caller never sees.
    """

    messages_sent: List[dict] = field(default_factory=list)
    response_text: str = ""
    duration: float = 0.0


class StructuredGenerator:
    def __init__(
        self,
        client: AsyncOpenAI | Any,
        model: str,
        max_retries: int = 3,
        temperature: float = 0.1,
        semaphore: asyncio.Semaphore | None = None,
    ):
        self._client_or_pool = client
        self.model = model
        self.max_retries = max_retries
        self.temperature = temperature
        self._semaphore = semaphore
        # Populated on every ``generate()`` call. Order = chronological:
        # last_attempts[0] is the first try, last_attempts[-1] is the
        # successful one (or the final failure if generate() raised).
        self.last_attempts: List[GenerateAttempt] = []

    async def _call(self, messages: List[dict]) -> Any:
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }

        async def _do_call() -> Any:
            if hasattr(self._client_or_pool, "acquire"):
                async with self._client_or_pool.acquire() as client:
                    return await client.chat.completions.create(**kwargs)
            return await self._client_or_pool.chat.completions.create(**kwargs)

        if self._semaphore is not None:
            async with self._semaphore:
                return await _do_call()
        return await _do_call()

    async def generate(
        self,
        messages: List[dict],
        schema: Type[T],
        extra_validate: Callable[[T], str | None] | None = None,
    ) -> T:
        """Call the LLM and guarantee a valid *schema* instance or raise.

        On ValidationError the error text is fed back to the LLM as a new user
        message and the request is retried up to *max_retries* times.
        Every API call (initial + retries) is recorded on ``self.last_attempts``
        so callers can attribute the full cost.

        ``extra_validate`` runs *after* schema validation succeeds. Return
        ``None`` to accept the parsed instance, or a non-empty string to reject
        it — the string is appended to the conversation as a user-correction
        turn and the request retries on the same retry budget as
        ValidationError. Use this for semantic checks the JSON schema can't
        express (e.g. "this id must be one of the candidates I gave you").
        """
        msgs = list(messages)
        last_raw = "{}"
        self.last_attempts = []

        for attempt in range(self.max_retries + 1):
            t0 = time.monotonic()
            try:
                response = await self._call(msgs)
            except (APIStatusError, APIConnectionError) as exc:
                # Failed API calls still consumed time and (usually) tokens.
                # For APIStatusError the server *did* return a body — capture
                # it so the caller's token accounting reflects the real output
                # the provider billed for, not zero. APIConnectionError never
                # reached a response, so we leave response_text empty but
                # still record the attempt to attribute the prompt cost.
                err_body = ""
                if isinstance(exc, APIStatusError):
                    body = getattr(exc, "body", None)
                    if body is not None:
                        err_body = body if isinstance(body, str) else str(body)
                    else:
                        err_body = getattr(exc, "message", "") or str(exc)
                self.last_attempts.append(GenerateAttempt(
                    messages_sent=list(msgs),
                    response_text=err_body,
                    duration=time.monotonic() - t0,
                ))
                status = getattr(exc, "status_code", None)
                if status in (429, 502, 503, 524) and attempt < self.max_retries:
                    delay = (2 ** attempt) + random.random()
                    await asyncio.sleep(delay)
                    continue
                raise

            duration = time.monotonic() - t0

            if response is None:
                self.last_attempts.append(GenerateAttempt(
                    messages_sent=list(msgs),
                    response_text="",
                    duration=duration,
                ))
                if attempt >= self.max_retries:
                    raise RuntimeError("LLM API returned None after all retries")
                msgs.append({
                    "role": "user",
                    "content": "The API returned an empty response. Please try again with raw JSON only.",
                })
                continue
            last_raw = response.choices[0].message.content or "{}"
            last_raw = _strip_fences(last_raw)
            self.last_attempts.append(GenerateAttempt(
                messages_sent=list(msgs),
                response_text=last_raw,
                duration=duration,
            ))

            try:
                parsed = schema.model_validate_json(last_raw)
            except ValidationError as exc:
                if attempt >= self.max_retries:
                    raise
                msgs.append({
                    "role": "user",
                    "content": (
                        f"JSON validation failed:\n{exc}\n\n"
                        "Please fix the output and respond with raw JSON only."
                    ),
                })
                continue

            if extra_validate is not None:
                err = extra_validate(parsed)
                if err:
                    if attempt >= self.max_retries:
                        raise ValueError(f"extra_validate rejected output after retries: {err}")
                    msgs.append({
                        "role": "user",
                        "content": (
                            f"Your previous output was rejected: {err}\n\n"
                            "Please fix the output and respond with raw JSON only."
                        ),
                    })
                    continue
            return parsed

        return schema.model_validate_json(last_raw)
