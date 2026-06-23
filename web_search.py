"""Web search using OpenAI's built-in web search tool."""
from dataclasses import dataclass, field

from openai import OpenAI


@dataclass
class SearchResult:
    snippets: list[str] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)

    @property
    def context(self) -> str:
        return "\n\n".join(self.snippets)


class WebSearcher:
    def __init__(self, client: OpenAI, max_results: int = 3):
        self._client = client
        self._max_results = max_results

    def search(self, query: str) -> SearchResult:
        if not query.strip():
            return SearchResult()

        response = self._client.responses.create(
            model="gpt-4o-mini",
            tools=[{"type": "web_search_preview"}],
            input=f"Search for: {query}. Return the top {self._max_results} relevant snippets.",
        )

        snippets: list[str] = []
        urls: list[str] = []
        for item in response.output:
            if hasattr(item, "content"):
                for block in item.content:
                    if hasattr(block, "text"):
                        snippets.append(block.text)
            if hasattr(item, "url"):
                urls.append(item.url)

        return SearchResult(snippets=snippets[: self._max_results], urls=urls[: self._max_results])
