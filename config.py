import os
from dataclasses import dataclass, field


@dataclass
class Config:
    openai_api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    qdrant_url: str = field(default_factory=lambda: os.environ.get("QDRANT_URL", "http://localhost:6333"))
    qdrant_collection: str = "ai_glasses_docs"
    whisper_model: str = "base.en"
    whisper_device: str = "cpu"
    vad_threshold: float = 0.5
    silence_duration_ms: int = 700
    topic_model: str = "gpt-4o-mini"
    synthesis_model: str = "gpt-4o-mini"
    latency_target_s: float = 4.0
    card_max_tokens: int = 200
    rag_top_k: int = 3
    web_search_results: int = 3
