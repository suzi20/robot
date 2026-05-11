"""文档加载器 — 支持 PDF / TXT / MD 格式。"""

from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document as LCDocument


def load_file(file_path: str) -> List[LCDocument]:
    """根据文件扩展名自动选择合适的加载器。

    Args:
        file_path: 文件绝对或相对路径。

    Returns:
        LangChain Document 列表。

    Raises:
        ValueError: 不支持的文件类型。
    """
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in (".txt", ".md"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"不支持的文件类型: {ext}，仅支持 PDF/TXT/MD")

    return loader.load()


def load_directory(dir_path: str) -> List[LCDocument]:
    """批量加载目录中所有受支持的文件。

    Args:
        dir_path: 目标目录路径。

    Returns:
        所有文档的列表。
    """
    docs: List[LCDocument] = []
    dir_path_obj = Path(dir_path)

    if not dir_path_obj.is_dir():
        raise NotADirectoryError(f"目录不存在: {dir_path}")

    supported_suffixes = {".pdf", ".txt", ".md"}
    for file_path in sorted(dir_path_obj.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in supported_suffixes:
            continue
        try:
            docs.extend(load_file(str(file_path)))
        except Exception as exc:
            raise RuntimeError(f"加载文件失败: {file_path}") from exc

    return docs
