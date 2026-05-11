"""文本分块器 — 将长文档拆分为可检索的块。"""

from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LCDocument


class TextChunker:
    """递归字符文本分块器，支持中英文混合文本。"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", ".", "！", "？", "；", " ", ""],
            length_function=len,
        )

    def chunk_documents(self, documents: List[LCDocument]) -> List[LCDocument]:
        """将文档列表切分成更小的块。

        Args:
            documents: 原始文档列表。

        Returns:
            切分后的文档块列表。
        """
        return self.splitter.split_documents(documents)
