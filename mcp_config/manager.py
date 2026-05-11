"""MCP 配置管理器 — 加载、转换、热重载 MCP 服务器定义。"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class MCPConfigManager:
    """MCP 服务配置管理。

    负责从 YAML 文件加载 MCP 服务器定义，并将其转换为
    qwen-agent 可接受的格式。实际的连接生命周期由
    qwen-agent 内部的 MCPManager 处理。
    """

    def __init__(self, config_path: str = "./mcp_config/servers.yaml"):
        self.config_path = config_path
        self.servers: Dict[str, Any] = {}
        self._load()

    def _load(self):
        """从 YAML 文件加载 MCP 服务器定义。"""
        path = Path(self.config_path)
        if not path.exists():
            self.servers = {}
            return

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        raw_servers = data.get("mcpServers", {}) if data else {}
        if raw_servers is None:
            raw_servers = {}

        # 环境变量替换 (例如 ${BRAVE_API_KEY})
        self.servers = {}
        for name, config in raw_servers.items():
            resolved = self._resolve_env_vars(config)
            self.servers[name] = resolved

    def _resolve_env_vars(self, config: dict) -> dict:
        """递归解析配置中的 ${ENV_VAR} 占位符。"""
        if not isinstance(config, dict):
            return config

        resolved = {}
        for key, value in config.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                env_value = os.getenv(env_var)
                if env_value is not None:
                    resolved[key] = env_value
                else:
                    resolved[key] = value  # 保留原样
            elif isinstance(value, dict):
                resolved[key] = self._resolve_env_vars(value)
            elif isinstance(value, list):
                resolved[key] = [
                    self._resolve_env_vars(item) if isinstance(item, dict)
                    else item
                    for item in value
                ]
            else:
                resolved[key] = value
        return resolved

    def to_qwen_format(self) -> List[Dict[str, Any]]:
        """转换为 qwen-agent 的 function_list 格式。"""
        if not self.servers:
            return []
        return [{"mcpServers": self.servers}]

    def list_servers(self) -> List[str]:
        """返回所有已配置的 MCP 服务器名称列表。"""
        return list(self.servers.keys())

    def reload(self):
        """从磁盘重新加载配置。"""
        self._load()

    def is_empty(self) -> bool:
        """是否没有任何 MCP 服务器配置。"""
        return len(self.servers) == 0
