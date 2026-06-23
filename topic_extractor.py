"""Extract topics/entities from transcript text via GPT-4o-mini."""
import json
from dataclasses import dataclass, field

from openai import OpenAI

SYSTEM_PROMPT = """\
You are a topic extractor for an AR knowledge assistant.
Given a snippet of speech, extract 1-3 specific topics, concepts, or named entities \
the speaker is engaging with. Return JSON only:
{"topics": ["topic1", "topic2"]}
If there's nothing specific enough to look up, return {"topics": []}.
Keep topics concise (2-5 words each). No explanations."""


@dataclass
class Topics:
    topics: list[str] = field(default_factory=list)

    @property
    def has_topics(self) -> bool:
        return len(self.topics) > 0


class TopicExtractor:
    def __init__(self, client: OpenAI, model: str = "gpt-4o-mini"):
        self._client = client
        self._model = model
        self._prompt_history: list[dict] = []

    def extract(self, text: str) -> Topics:
        if not text.strip():
            return Topics()

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            max_tokens=100,
            temperature=0,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
        topics = data.get("topics", [])
        self._prompt_history.append({"text": text, "topics": topics})
        return Topics(topics=topics)

    @property
    def history(self) -> list[dict]:
        return list(self._prompt_history)
