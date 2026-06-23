"""Qdrant-backed RAG retriever using llama-index."""
from dataclasses import dataclass, field

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


@dataclass
class RAGResult:
    chunks: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

    @property
    def context(self) -> str:
        return "\n\n".join(self.chunks)


class RAGRetriever:
    def __init__(self, qdrant_url: str, collection: str, top_k: int = 3):
        self._client = QdrantClient(url=qdrant_url)
        self._collection = collection
        self._top_k = top_k
        vector_store = QdrantVectorStore(client=self._client, collection_name=collection)
        storage_ctx = StorageContext.from_defaults(vector_store=vector_store)
        self._index = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_ctx
        )
        self._retriever = self._index.as_retriever(similarity_top_k=top_k)

    def retrieve(self, query: str) -> RAGResult:
        if not query.strip():
            return RAGResult()
        nodes = self._retriever.retrieve(query)
        chunks = [n.get_content() for n in nodes]
        sources = [n.metadata.get("file_name", "unknown") for n in nodes]
        return RAGResult(chunks=chunks, sources=sources)
