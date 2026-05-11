"""Application configuration.

Configuration is loaded in this order:
1. built-in defaults
2. values from ``config/settings.yaml``
3. values from ``.env`` and process environment variables

Environment variables use upper-case field names, for example
``LLM_MODEL`` or ``CHROMA_PERSIST_DIR``. ``DEEPSEEK_API_KEY`` is also
accepted as the LLM API key.
"""

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import dotenv_values
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings for the Qwen-Agent MCP + RAG assistant."""

    # LLM (DeepSeek / OpenAI-compatible)
    llm_model: str = "deepseek-chat"
    llm_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "llm_api_key",
            "LLM_API_KEY",
            "DEEPSEEK_API_KEY",
            "OPENAI_API_KEY",
        ),
    )
    llm_base_url: str = "https://api.deepseek.com"

    # Embedding
    embedding_provider: str = "local"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_dimension: int = 512
    embedding_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "embedding_api_key",
            "EMBEDDING_API_KEY",
            "DASHSCOPE_API_KEY",
        ),
    )

    # ChromaDB
    chroma_persist_dir: str = "./knowledge_base"
    chroma_collection_name: str = "documents"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Retrieval. Chroma returns cosine distance, so this is converted to a
    # distance threshold in the retriever.
    retrieval_top_k: int = 5
    retrieval_score_threshold: float = 0.6
    retrieval_agentic_enabled: bool = True
    retrieval_multi_query_count: int = 3
    retrieval_candidate_multiplier: int = 3

    # MCP
    mcp_config_path: str = "./mcp_config/servers.yaml"

    # Agent
    system_prompt_template: str = "zh"
    max_tool_rounds: int = 10
    enable_streaming: bool = True

    # Chatbot persona
    chatbot_mode: bool = True
    assistant_name: str = "小智"
    user_name: str = "你"
    relationship_style: str = "像一个可靠、松弛、有边界感的朋友"
    personality: str = "温暖、自然、好奇、会认真听人说话，也能给出清楚实用的建议"
    response_style: str = "口语化、简洁、有来有回；不要每次都写成正式报告，除非用户要求"
    proactive_follow_up: bool = True
    chat_memory_enabled: bool = True
    chat_history_path: str = "./data/chat_history.json"
    chat_history_max_messages: int = 30

    # Long-term semantic memory
    long_term_memory_enabled: bool = True
    memory_collection_name: str = "long_term_memories"
    memory_top_k: int = 5
    memory_score_threshold: float = 0.55
    memory_max_chars_per_turn: int = 1200

    # Observability and evaluation
    observability_enabled: bool = True
    observability_log_path: str = "./logs/agent_events.jsonl"
    eval_report_path: str = "./logs/eval_report.json"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    @classmethod
    def load(cls, config_path: str = "config/settings.yaml", env_file: str = ".env") -> "Settings":
        """Load settings from YAML, then override with .env/environment values."""
        data: dict[str, Any] = {}

        path = Path(config_path)
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                data.update(yaml.safe_load(f) or {})
        elif config_path:
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        env_data = cls._collect_env_overrides(env_file)
        data.update(env_data)
        return cls(**data)

    @classmethod
    def _collect_env_overrides(cls, env_file: str) -> dict[str, Any]:
        values: dict[str, str] = {}
        env_path = Path(env_file)
        if env_path.exists():
            values.update({k: v for k, v in dotenv_values(env_path).items() if v is not None})
        values.update(os.environ)

        mapping = {
            "llm_model": ("LLM_MODEL",),
            "llm_api_key": ("LLM_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY"),
            "llm_base_url": ("LLM_BASE_URL", "DEEPSEEK_BASE_URL", "OPENAI_BASE_URL"),
            "embedding_provider": ("EMBEDDING_PROVIDER",),
            "embedding_model": ("EMBEDDING_MODEL",),
            "embedding_dimension": ("EMBEDDING_DIMENSION",),
            "embedding_api_key": ("EMBEDDING_API_KEY", "DASHSCOPE_API_KEY"),
            "chroma_persist_dir": ("CHROMA_PERSIST_DIR",),
            "chroma_collection_name": ("CHROMA_COLLECTION_NAME",),
            "chunk_size": ("CHUNK_SIZE",),
            "chunk_overlap": ("CHUNK_OVERLAP",),
            "retrieval_top_k": ("RETRIEVAL_TOP_K",),
            "retrieval_score_threshold": ("RETRIEVAL_SCORE_THRESHOLD",),
            "retrieval_agentic_enabled": ("RETRIEVAL_AGENTIC_ENABLED",),
            "retrieval_multi_query_count": ("RETRIEVAL_MULTI_QUERY_COUNT",),
            "retrieval_candidate_multiplier": ("RETRIEVAL_CANDIDATE_MULTIPLIER",),
            "mcp_config_path": ("MCP_CONFIG_PATH",),
            "system_prompt_template": ("SYSTEM_PROMPT_TEMPLATE",),
            "max_tool_rounds": ("MAX_TOOL_ROUNDS",),
            "enable_streaming": ("ENABLE_STREAMING",),
            "chatbot_mode": ("CHATBOT_MODE",),
            "assistant_name": ("ASSISTANT_NAME",),
            "user_name": ("USER_NAME",),
            "relationship_style": ("RELATIONSHIP_STYLE",),
            "personality": ("PERSONALITY",),
            "response_style": ("RESPONSE_STYLE",),
            "proactive_follow_up": ("PROACTIVE_FOLLOW_UP",),
            "chat_memory_enabled": ("CHAT_MEMORY_ENABLED",),
            "chat_history_path": ("CHAT_HISTORY_PATH",),
            "chat_history_max_messages": ("CHAT_HISTORY_MAX_MESSAGES",),
            "long_term_memory_enabled": ("LONG_TERM_MEMORY_ENABLED",),
            "memory_collection_name": ("MEMORY_COLLECTION_NAME",),
            "memory_top_k": ("MEMORY_TOP_K",),
            "memory_score_threshold": ("MEMORY_SCORE_THRESHOLD",),
            "memory_max_chars_per_turn": ("MEMORY_MAX_CHARS_PER_TURN",),
            "observability_enabled": ("OBSERVABILITY_ENABLED",),
            "observability_log_path": ("OBSERVABILITY_LOG_PATH",),
            "eval_report_path": ("EVAL_REPORT_PATH",),
        }

        overrides: dict[str, Any] = {}
        for field_name, env_names in mapping.items():
            for env_name in env_names:
                if env_name in values and values[env_name] not in (None, ""):
                    overrides[field_name] = values[env_name]
                    break
        return overrides

    @property
    def effective_api_key(self) -> str:
        """Return the LLM API key, or raise a clear configuration error."""
        if self.llm_api_key:
            return self.llm_api_key
        raise ValueError("API Key 未设置。请在 .env 中设置 DEEPSEEK_API_KEY，或通过环境变量提供。")

    @property
    def effective_embedding_api_key(self) -> Optional[str]:
        """Return the embedding API key when a remote embedding provider needs one."""
        return self.embedding_api_key or self.llm_api_key
