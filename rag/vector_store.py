"""向量存储 — 基于 ChromaDB 的持久化向量数据库操作。"""

from typing import List

import chromadb


class VectorStore:
    """ChromaDB 向量存储封装，提供增、删、查操作。"""

    def __init__(
        self,
        persist_dir: str = "./knowledge_base",
        collection_name: str = "documents",
    ):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self,
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[dict],
        ids: List[str],
    ):
        """将嵌入后的文档添加到向量库。

        Args:
            embeddings: 向量列表。
            documents: 原始文本列表。
            metadatas: 元数据列表。
            ids: 唯一标识列表。
        """
        self.collection.upsert(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )

    def similarity_search(
        self, query_embedding: List[float], k: int = 5
    ) -> List[dict]:
        """按向量相似度搜索。

        Args:
            query_embedding: 查询向量。
            k: 返回结果数。

        Returns:
            格式化的搜索结果列表。
        """
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        return self._format_results(results)

    def count(self) -> int:
        """返回当前集合中的文档总数。"""
        return self.collection.count()

    def delete_collection(self):
        """删除并重建集合（清空所有数据）。"""
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def _format_results(self, raw_results) -> List[dict]:
        """将 ChromaDB 原始结果转换为统一格式。"""
        formatted = []
        ids_list = raw_results["ids"][0]
        docs_list = raw_results["documents"][0]
        metas_list = raw_results["metadatas"][0]
        dists_list = raw_results["distances"][0]

        for i in range(len(ids_list)):
            formatted.append({
                "id": ids_list[i],
                "document": docs_list[i],
                "metadata": metas_list[i] or {},
                "score": dists_list[i],
            })
        return formatted
