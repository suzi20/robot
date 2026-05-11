"""核心 Agent 组装 — 使用 DeepSeek LLM + MCP 工具 + RAG 检索。"""

import time
from typing import Dict, Iterator, List

from qwen_agent.agents import Assistant as QwenAssistant
from qwen_agent.tools import TOOL_REGISTRY

from config.settings import Settings
from rag.embedder import create_embedder
from rag.retriever import Retriever
from rag.vector_store import VectorStore
from memory import MemoryStore
from mcp_config.manager import MCPConfigManager
from observability import TraceLogger
from agent.knowledge_tool import KnowledgeRetrieval
from agent.system_prompt import get_system_prompt


class RAGMCPAssistant:
    """整合 RAG 检索 + MCP 工具的智能问答助手 (DeepSeek + 本地 Embedding)。"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.mcp_servers: List[str] = []
        self.trace_logger = TraceLogger(
            settings.observability_log_path,
            enabled=settings.observability_enabled,
        )

        # 1. 初始化 Embedder（本地 fastembed 不需要 API Key，可选 DashScope）
        embedding_api_key = (
            settings.effective_embedding_api_key
            if settings.embedding_provider.lower() in ("dashscope", "qwen")
            else None
        )
        self.embedder = create_embedder(
            provider=settings.embedding_provider,
            model_name=settings.embedding_model,
            api_key=embedding_api_key,
            dimension=settings.embedding_dimension,
        )

        # 2. 初始化向量存储与检索管线
        self.vector_store = VectorStore(
            persist_dir=settings.chroma_persist_dir,
            collection_name=settings.chroma_collection_name,
        )
        self.retriever = Retriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            top_k=settings.retrieval_top_k,
            score_threshold=settings.retrieval_score_threshold,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            agentic_enabled=settings.retrieval_agentic_enabled,
            multi_query_count=settings.retrieval_multi_query_count,
            candidate_multiplier=settings.retrieval_candidate_multiplier,
            trace_logger=self.trace_logger,
        )
        self.memory_store = (
            MemoryStore(
                embedder=self.embedder,
                persist_dir=settings.chroma_persist_dir,
                collection_name=settings.memory_collection_name,
                top_k=settings.memory_top_k,
                score_threshold=settings.memory_score_threshold,
                max_chars_per_turn=settings.memory_max_chars_per_turn,
                trace_logger=self.trace_logger,
            )
            if settings.long_term_memory_enabled
            else None
        )

        # 3. 注册 knowledge_retrieval 工具
        self._register_knowledge_tool()

        # 4. 构建 function_list: MCP 工具 + 自定义工具
        function_list = self._build_function_list()

        # 5. 使用 qwen-agent 的 oai (OpenAI 兼容) model_type 对接 DeepSeek
        self.agent = QwenAssistant(
            llm={
                "model": settings.llm_model,
                "model_type": "oai",                      # ← 关键：使用 OpenAI 兼容模式
                "api_key": settings.effective_api_key,
                "base_url": settings.llm_base_url,         # https://api.deepseek.com
            },
            system_message=get_system_prompt(settings.system_prompt_template, settings=settings),
            function_list=function_list,
        )

    def _register_knowledge_tool(self):
        tool_instance = KnowledgeRetrieval(retriever=self.retriever)
        TOOL_REGISTRY["knowledge_retrieval"] = lambda _cfg=None: tool_instance

    def _build_function_list(self) -> list:
        function_list: list = []
        mcp_manager = MCPConfigManager(self.settings.mcp_config_path)
        mcp_configs = mcp_manager.to_qwen_format()
        self.mcp_servers = mcp_manager.list_servers()
        if mcp_configs:
            function_list.extend(mcp_configs)
        function_list.append("knowledge_retrieval")
        return function_list

    def run(self, messages: List[Dict], stream: bool = True) -> Iterator:
        return self._run_with_memory_and_tracing(messages=messages, stream=stream)

    def remember(self, user_text: str, assistant_text: str, session_id: str = "default") -> None:
        if self.memory_store:
            self.memory_store.remember(user_text, assistant_text, session_id=session_id)

    def clear_memory(self) -> None:
        if self.memory_store:
            self.memory_store.clear()

    def _run_with_memory_and_tracing(self, messages: List[Dict], stream: bool = True) -> Iterator:
        started = time.perf_counter()
        prepared_messages = list(messages)
        last_user_message = self._last_user_message(prepared_messages)
        memory_context = ""

        if self.memory_store and last_user_message:
            memory_context = self.memory_store.format_for_prompt(last_user_message)
            if memory_context:
                prepared_messages = [
                    {
                        "role": "system",
                        "content": (
                            "以下是与当前对话相关的长期记忆。它们可能有帮助，但如果和用户当前表达冲突，"
                            "以用户当前表达为准。\n\n"
                            f"{memory_context}"
                        ),
                    },
                    *prepared_messages,
                ]

        final_response = ""
        chunk_count = 0
        try:
            for chunk in self.agent.run(messages=prepared_messages, stream=stream):
                chunk_count += 1
                content = self._last_assistant_content(chunk)
                if content:
                    final_response = content
                yield chunk
        finally:
            self.trace_logger.log(
                "agent.run",
                {
                    "query": last_user_message,
                    "stream": stream,
                    "chunk_count": chunk_count,
                    "memory_context_used": bool(memory_context),
                    "response_chars": len(final_response),
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            )

    def _last_user_message(self, messages: List[Dict]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content", "")
                return content if isinstance(content, str) else str(content)
        return ""

    def _last_assistant_content(self, chunk) -> str:
        for message in reversed(chunk or []):
            role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
            content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
            if role == "assistant" and content:
                return content if isinstance(content, str) else str(content)
        return ""
