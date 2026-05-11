"""检索管线 — 编排加载、分块、嵌入、存储和检索全流程。"""

import hashlib
import re
import time
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document as LCDocument

from rag.loader import load_directory, load_file
from rag.chunker import TextChunker
from rag.vector_store import VectorStore


class Retriever:
    """端到端检索管线。

    提供文档导入 (ingest) 和知识检索 (retrieve) 两大核心接口。
    """

    def __init__(
        self,
        embedder,
        vector_store: VectorStore,
        top_k: int = 5,
        score_threshold: float = 0.6,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        agentic_enabled: bool = True,
        multi_query_count: int = 3,
        candidate_multiplier: int = 3,
        trace_logger=None,
    ):
        self.embedder = embedder
        self.store = vector_store
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.agentic_enabled = agentic_enabled
        self.multi_query_count = multi_query_count
        self.candidate_multiplier = candidate_multiplier
        self.trace_logger = trace_logger
        self.last_trace: dict = {}

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[dict]:
        """完整检索流程：嵌入查询 → 向量搜索 → 阈值过滤。

        Args:
            query: 用户查询文本。
            top_k: 返回结果数，不传则使用默认值。

        Returns:
            过滤后的检索结果列表。
        """
        k = top_k or self.top_k
        query_emb = self.embedder.embed_query(query)
        results = self.store.similarity_search(query_emb, k=k)

        # ChromaDB 使用余弦距离 (cosine distance)，越接近 0 越相似
        # 阈值转距离：score_threshold=0.6 → distance_threshold=0.4
        distance_threshold = 1.0 - self.score_threshold
        filtered = [r for r in results if r["score"] <= distance_threshold]

        return filtered

    def agentic_retrieve(self, query: str, top_k: Optional[int] = None) -> List[dict]:
        """Agentic RAG: plan, rewrite, multi-retrieve, rerank, and self-check."""
        started = time.perf_counter()
        k = top_k or self.top_k
        rewritten_queries = self._rewrite_queries(query)
        candidate_k = max(k, k * self.candidate_multiplier)

        candidates_by_id: dict[str, dict] = {}
        for rewritten in rewritten_queries:
            query_emb = self.embedder.embed_query(rewritten)
            results = self.store.similarity_search(query_emb, k=candidate_k)
            for result in results:
                existing = candidates_by_id.get(result["id"])
                if not existing or result["score"] < existing["score"]:
                    item = dict(result)
                    item["matched_query"] = rewritten
                    candidates_by_id[result["id"]] = item

        distance_threshold = 1.0 - self.score_threshold
        filtered = [item for item in candidates_by_id.values() if item["score"] <= distance_threshold]
        reranked = sorted(filtered, key=lambda item: self._rerank_score(query, item))[:k]

        confidence = self._estimate_confidence(query, reranked)
        self.last_trace = {
            "mode": "agentic_rag",
            "plan": [
                "analyze_query",
                "rewrite_query",
                "multi_retrieve",
                "deduplicate",
                "rerank",
                "self_check",
            ],
            "query": query,
            "rewritten_queries": rewritten_queries,
            "candidate_count": len(candidates_by_id),
            "filtered_count": len(filtered),
            "selected_count": len(reranked),
            "confidence": confidence,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

        if self.trace_logger:
            self.trace_logger.log("rag.retrieve", self.last_trace)
        return reranked

    def retrieve_and_format(self, query: str, top_k: Optional[int] = None) -> str:
        """检索并将结果格式化为 LLM 可读的上下文文本。

        Args:
            query: 用户查询文本。

        Returns:
            格式化后的上下文字符串，无结果时返回空字符串。
        """
        if self.agentic_enabled:
            results = self.agentic_retrieve(query, top_k=top_k)
        else:
            results = self.retrieve(query, top_k=top_k)
            self.last_trace = {
                "mode": "simple_rag",
                "query": query,
                "selected_count": len(results),
            }
            if self.trace_logger:
                self.trace_logger.log("rag.retrieve", self.last_trace)
        if not results:
            return ""

        context_parts = []
        for i, r in enumerate(results, 1):
            source = r["metadata"].get("source", "未知来源")
            page = r["metadata"].get("page", "")
            source_str = f"{source}" + (f" 第{page}页" if page != "" else "")
            context_parts.append(f"[{i}] (来源: {source_str})\n{r['document']}\n")

        return "\n---\n".join(context_parts)

    def _rewrite_queries(self, query: str) -> List[str]:
        """Create deterministic query variants for multi-query retrieval."""
        normalized = re.sub(r"\s+", " ", query).strip()
        cleanup = normalized
        for token in ("根据知识库", "基于知识库", "请问", "请", "一下", "帮我", "你能不能", "能不能"):
            cleanup = cleanup.replace(token, "")
        cleanup = cleanup.strip(" ，。？！;；")

        variants = [normalized]
        if cleanup and cleanup != normalized:
            variants.append(cleanup)

        clauses = [
            part.strip(" ，。？！;；")
            for part in re.split(r"[，。？！;；\n]", normalized)
            if len(part.strip()) >= 4
        ]
        for clause in clauses:
            if clause not in variants:
                variants.append(clause)

        deduped = []
        for item in variants:
            if item and item not in deduped:
                deduped.append(item)
        return deduped[: max(1, self.multi_query_count)]

    def _rerank_score(self, query: str, item: dict) -> float:
        """Lower score is better: combine vector distance and lexical overlap."""
        distance = float(item.get("score", 1.0))
        doc = item.get("document", "")
        overlap = self._char_overlap(query, doc)
        return distance - (0.08 * overlap)

    def _char_overlap(self, query: str, doc: str) -> float:
        query_chars = {ch for ch in query.lower() if "\u4e00" <= ch <= "\u9fff" or ch.isalnum()}
        doc_chars = {ch for ch in doc.lower() if "\u4e00" <= ch <= "\u9fff" or ch.isalnum()}
        if not query_chars or not doc_chars:
            return 0.0
        return len(query_chars & doc_chars) / len(query_chars)

    def _estimate_confidence(self, query: str, results: List[dict]) -> str:
        if not results:
            return "none"
        best_distance = min(float(item.get("score", 1.0)) for item in results)
        best_overlap = max(self._char_overlap(query, item.get("document", "")) for item in results)
        if best_distance <= 0.22 or best_overlap >= 0.65:
            return "high"
        if best_distance <= 0.4 or best_overlap >= 0.4:
            return "medium"
        return "low"

    def ingest_document(self, file_path: str) -> int:
        """完整导入流程：加载 → 分块 → 嵌入 → 存储。

        Args:
            file_path: 文档路径。

        Returns:
            导入的文本块数量。
        """
        documents = load_file(file_path)
        return self.ingest_documents(documents, default_source=str(Path(file_path).resolve()))

    def ingest_directory(self, dir_path: str) -> int:
        """批量导入目录中的 PDF/TXT/MD 文件，支持递归子目录。"""
        documents = load_directory(dir_path)
        if not documents:
            return 0
        return self.ingest_documents(documents, default_source=str(Path(dir_path).resolve()))

    def ingest_documents(self, documents: List[LCDocument], default_source: str = "") -> int:
        """导入已加载的 LangChain 文档列表。"""
        chunks = self.chunker.chunk_documents(documents)
        if not chunks:
            return 0

        texts = [chunk.page_content for chunk in chunks]
        metadatas = []
        for chunk in chunks:
            meta = dict(chunk.metadata)
            source = meta.get("source") or default_source or "unknown"
            meta["source"] = str(Path(source).resolve()) if source != "unknown" else source
            metadatas.append(meta)

        ids = []
        for i, (text, meta) in enumerate(zip(texts, metadatas)):
            key = f"{meta.get('source', default_source)}::{meta.get('page', '')}::{i}::{text[:80]}"
            digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
            ids.append(f"doc_{digest}")

        embeddings = self.embedder.embed_documents(texts)

        self.store.add_documents(embeddings, texts, metadatas, ids)

        return len(texts)
