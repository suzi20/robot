"""Embedding 封装 — 支持本地 (fastembed) 和 DashScope 两种后端。"""

import os
import time
from typing import List, Optional


class LocalEmbedder:
    """基于 fastembed 的本地 embedding（无需 API，支持中英文）。"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def embed_query(self, text: str) -> List[float]:
        return list(self.model.embed(text))[0].tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [emb.tolist() for emb in self.model.embed(texts)]


class DashScopeEmbedder:
    """DashScope text-embedding-v3（备选方案）。"""

    def __init__(self, api_key: str = None, dimension: int = 1024):
        import dashscope
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.dimension = dimension
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY 未设置")
        dashscope.api_key = self.api_key

    def embed_query(self, text: str) -> List[float]:
        from dashscope import TextEmbedding
        from http import HTTPStatus
        resp = TextEmbedding.call(
            model=TextEmbedding.Models.text_embedding_v3,
            input=text,
            dimension=self.dimension,
        )
        if resp.status_code == HTTPStatus.OK:
            return resp.output["embeddings"][0]["embedding"]
        raise RuntimeError(f"Embedding API 失败: {resp}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        from dashscope import TextEmbedding
        from http import HTTPStatus
        results = []
        batch_size = 25
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            results.extend(self._embed_batch(batch))
        return results

    def _embed_batch(self, texts: List[str], max_retries: int = 3) -> List[List[float]]:
        from dashscope import TextEmbedding
        from http import HTTPStatus
        for attempt in range(max_retries):
            resp = TextEmbedding.call(
                model=TextEmbedding.Models.text_embedding_v3,
                input=texts,
                dimension=self.dimension,
            )
            if resp.status_code == HTTPStatus.OK:
                return [item["embedding"] for item in resp.output["embeddings"]]
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Embedding API 失败 (重试{max_retries}次): {resp}")


def create_embedder(
    provider: str = "local",
    model_name: str = "BAAI/bge-small-zh-v1.5",
    api_key: Optional[str] = None,
    dimension: int = 384,
):
    """工厂方法：根据 provider 创建对应的 Embedder 实例。"""
    if provider == "local":
        return LocalEmbedder(model_name=model_name)
    elif provider in ("dashscope", "qwen"):
        return DashScopeEmbedder(api_key=api_key, dimension=dimension)
    else:
        raise ValueError(f"不支持的 embedding provider: {provider}，可选: local, dashscope")
