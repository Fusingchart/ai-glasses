"""Main AI pipeline: VAD → Whisper → Topics → RAG + Web (parallel) → Card."""
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from card_synthesizer import Card, CardSynthesizer
from config import Config
from rag import RAGRetriever
from topic_extractor import TopicExtractor
from transcriber import Transcriber
from vad import VADDetector, VADResult
from web_search import WebSearcher


class Pipeline:
    def __init__(self, cfg: Config):
        self._cfg = cfg
        self._client = OpenAI(api_key=cfg.openai_api_key)
        self._vad = VADDetector(
            threshold=cfg.vad_threshold,
            silence_duration_ms=cfg.silence_duration_ms,
        )
        self._transcriber = Transcriber(
            model_size=cfg.whisper_model,
            device=cfg.whisper_device,
        )
        self._topics = TopicExtractor(client=self._client, model=cfg.topic_model)
        self._rag = RAGRetriever(
            qdrant_url=cfg.qdrant_url,
            collection=cfg.qdrant_collection,
            top_k=cfg.rag_top_k,
        )
        self._searcher = WebSearcher(client=self._client, max_results=cfg.web_search_results)
        self._synthesizer = CardSynthesizer(
            client=self._client,
            model=cfg.synthesis_model,
            max_tokens=cfg.card_max_tokens,
        )
        self._executor = ThreadPoolExecutor(max_workers=4)
        self.on_card: list = []

    def _process_audio(self, result: VADResult) -> Card | None:
        t0 = time.monotonic()

        transcript = self._transcriber.transcribe(result.audio_chunk)
        if not transcript.text:
            return None

        topics = self._topics.extract(transcript.text)
        if not topics.has_topics:
            return None

        query = " ".join(topics.topics)

        # RAG and web search in parallel
        with ThreadPoolExecutor(max_workers=2) as pool:
            rag_future = pool.submit(self._rag.retrieve, query)
            web_future = pool.submit(self._searcher.search, query)
            rag_result = rag_future.result()
            web_result = web_future.result()

        card = self._synthesizer.synthesize(
            topics=topics.topics,
            doc_context=rag_result.context,
            web_context=web_result.context,
        )

        elapsed = time.monotonic() - t0
        print(f"[pipeline] latency={elapsed:.2f}s topics={topics.topics}")
        if elapsed > self._cfg.latency_target_s:
            print(f"[pipeline] WARNING: exceeded {self._cfg.latency_target_s}s target")

        return card

    def feed(self, audio_chunk) -> None:
        vad_result = self._vad.process_chunk(audio_chunk)
        if vad_result is None:
            return
        card = self._process_audio(vad_result)
        if card:
            for cb in self.on_card:
                cb(card)

    def register_card_callback(self, fn) -> None:
        self.on_card.append(fn)
