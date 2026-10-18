from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from .config import Settings


SYSTEM_PROMPT = (
    "You are a careful assistant answering questions over a document corpus. "
    "Use ONLY the context chunks provided. For every fact you state, cite the "
    "chunk id in square brackets, e.g. [07e1ec5b63567268]. "
    "Do not invent chunk ids. Do not use outside knowledge. "
    "Be concise and summarize your answer. "
    "If the context does not contain the answer, respond with exactly: "
    "I don't know based on the available documents."
)

ABSTAIN_TOKEN = "I don't know based on the available documents"


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class GrokClient:
    def __init__(self, settings: Settings) -> None:
        if not settings.grok_api_key:
            raise RuntimeError(
                "RAG_GROK_API_KEY is not set. "
                "Get a key at https://console.x.ai/ and add it to .env"
            )
        self._client = OpenAI(
            api_key=settings.grok_api_key,
            base_url=settings.grok_base_url,
        )
        self.model = settings.grok_model
        self.max_tokens = settings.llm_max_tokens
        self.temperature = settings.llm_temperature

    def complete(self, system: str, user: str) -> GenerationResult:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        text = (resp.choices[0].message.content or "").strip()
        usage = resp.usage
        return GenerationResult(
            text=text,
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "completion_tokens", 0),
            total_tokens=getattr(usage, "total_tokens", 0),
        )