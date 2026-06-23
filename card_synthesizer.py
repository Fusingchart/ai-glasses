"""Synthesize a display card JSON from RAG + web context."""
import json
from dataclasses import dataclass

from openai import OpenAI

SYSTEM_PROMPT = """\
You are an AR card synthesizer for smart glasses.
Given a user's spoken topic and context from their personal docs and the web, \
output a single display card as JSON:
{
  "title": "short title (max 6 words)",
  "body": "1-2 sentence insight (max 30 words)",
  "source": "doc filename or URL",
  "type": "doc | web | both"
}
Be concise. The card appears in the user's field of view. No markdown, no extra fields."""


@dataclass
class Card:
    title: str
    body: str
    source: str
    card_type: str

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "body": self.body,
            "source": self.source,
            "type": self.card_type,
        }


class CardSynthesizer:
    def __init__(self, client: OpenAI, model: str = "gpt-4o-mini", max_tokens: int = 200):
        self._client = client
        self._model = model
        self._max_tokens = max_tokens

    def synthesize(
        self,
        topics: list[str],
        doc_context: str,
        web_context: str,
    ) -> Card | None:
        if not topics:
            return None

        user_content = (
            f"Topics: {', '.join(topics)}\n\n"
            f"Personal docs:\n{doc_context or 'No relevant docs found.'}\n\n"
            f"Web:\n{web_context or 'No web results.'}"
        )

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=self._max_tokens,
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
        return Card(
            title=data.get("title", ""),
            body=data.get("body", ""),
            source=data.get("source", ""),
            card_type=data.get("type", "both"),
        )
