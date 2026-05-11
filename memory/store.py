"""Long-term semantic memory backed by the same ChromaDB vector store."""

from __future__ import annotations

import hashlib
import time
from typing import Optional

from rag.vector_store import VectorStore


class MemoryStore:
    """Store and retrieve durable conversation memories."""

    def __init__(
        self,
        embedder,
        persist_dir: str,
        collection_name: str = "long_term_memories",
        top_k: int = 5,
        score_threshold: float = 0.55,
        max_chars_per_turn: int = 1200,
        trace_logger=None,
    ):
        self.embedder = embedder
        self.store = VectorStore(persist_dir=persist_dir, collection_name=collection_name)
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.max_chars_per_turn = max_chars_per_turn
        self.trace_logger = trace_logger

    def remember(self, user_text: str, assistant_text: str, session_id: str = "default") -> Optional[str]:
        """Persist one interaction as a semantic memory."""
        user_text = (user_text or "").strip()
        assistant_text = (assistant_text or "").strip()
        if not user_text or not assistant_text:
            return None

        memory_text = (
            f"用户说：{user_text[:self.max_chars_per_turn]}\n"
            f"助手回应：{assistant_text[:self.max_chars_per_turn]}"
        )
        digest = hashlib.sha1(f"{session_id}:{time.time()}:{memory_text[:120]}".encode("utf-8")).hexdigest()[:16]
        memory_id = f"mem_{digest}"
        embedding = self.embedder.embed_documents([memory_text])
        self.store.add_documents(
            embeddings=embedding,
            documents=[memory_text],
            metadatas=[{"session_id": session_id, "created_at": time.time()}],
            ids=[memory_id],
        )

        if self.trace_logger:
            self.trace_logger.log("memory.write", {"memory_id": memory_id, "session_id": session_id})
        return memory_id

    def search(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        """Retrieve memories related to the current user query."""
        query = (query or "").strip()
        if not query or self.store.count() == 0:
            return []

        k = top_k or self.top_k
        query_embedding = self.embedder.embed_query(query)
        results = self.store.similarity_search(query_embedding, k=k)
        distance_threshold = 1.0 - self.score_threshold
        filtered = [item for item in results if item["score"] <= distance_threshold]

        if self.trace_logger:
            self.trace_logger.log(
                "memory.search",
                {"query": query, "hits": len(filtered), "candidate_count": len(results)},
            )
        return filtered

    def format_for_prompt(self, query: str) -> str:
        """Format retrieved memories as prompt context."""
        memories = self.search(query)
        if not memories:
            return ""

        parts = []
        for i, item in enumerate(memories, 1):
            parts.append(f"[记忆 {i}] {item['document']}")
        return "\n\n".join(parts)

    def clear(self) -> None:
        self.store.delete_collection()
        if self.trace_logger:
            self.trace_logger.log("memory.clear", {})
