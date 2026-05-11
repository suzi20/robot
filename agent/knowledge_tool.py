"""知识检索工具 — 作为 Qwen-Agent 的注册工具，将 RAG 检索能力桥接到 Agent 的函数调用系统。"""

from typing import Optional

from qwen_agent.tools.base import BaseTool, register_tool

from rag.retriever import Retriever


@register_tool("knowledge_retrieval")
class KnowledgeRetrieval(BaseTool):
    """从本地知识库中检索与问题相关的文档内容。"""

    description = (
        "从本地知识库中检索与用户问题相关的文档信息。"
        "当你需要回答关于已上传文档（如 PDF、TXT、MD 文件）中的内容时，请调用此工具。"
        "如果知识库中没有相关信息，工具会返回空结果。"
    )
    parameters = [
        {
            "name": "query",
            "type": "string",
            "description": "用于检索相关信息的搜索查询，应包含关键概念和术语",
            "required": True,
        },
        {
            "name": "top_k",
            "type": "integer",
            "description": "返回的相关文档片段数量 (默认: 5)",
            "required": False,
        },
    ]

    def __init__(self, retriever: Optional[Retriever] = None):
        super().__init__()
        self._retriever = retriever

    def call(self, params: str, **kwargs) -> str:
        """执行知识检索。

        Args:
            params: JSON 字符串，包含 query 和可选的 top_k。

        Returns:
            格式化后的检索结果文本。
        """
        if isinstance(params, dict):
            params_dict = params
        else:
            import json5
            try:
                params_dict = json5.loads(params)
            except Exception:
                return "参数解析失败，请提供有效的 JSON 格式参数。"

        query = params_dict.get("query", "")
        if not query:
            return "检索失败：query 参数不能为空。"

        top_k = params_dict.get("top_k", None)

        if not self._retriever:
            return "错误：知识库未初始化，请检查配置。"

        try:
            result = self._retriever.retrieve_and_format(query, top_k=top_k)
            if not result:
                return "知识库中未找到与查询相关的信息。"
            trace = getattr(self._retriever, "last_trace", {})
            if trace:
                result += (
                    "\n\n[检索诊断]\n"
                    f"- 模式: {trace.get('mode', 'unknown')}\n"
                    f"- 改写查询: {', '.join(trace.get('rewritten_queries', [query]))}\n"
                    f"- 候选片段: {trace.get('candidate_count', trace.get('selected_count', 0))}\n"
                    f"- 选中片段: {trace.get('selected_count', 0)}\n"
                    f"- 置信度: {trace.get('confidence', 'unknown')}"
                )
            return result
        except Exception as e:
            return f"知识检索失败: {str(e)}"
