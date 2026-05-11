from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Document:
    """加载后的原始文档。"""
    page_content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """切分后的文本块。"""
    content: str
    metadata: dict = field(default_factory=dict)
    chunk_id: Optional[str] = None


@dataclass
class QueryResult:
    """检索结果项。"""
    id: str
    document: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0
