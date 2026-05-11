"""Lightweight JSONL observability for agent runs, retrieval, and memory."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional


class TraceLogger:
    """Append-only JSONL event logger."""

    def __init__(self, log_path: str, enabled: bool = True):
        self.log_path = Path(log_path)
        self.enabled = enabled

    def log(self, event_type: str, payload: Optional[dict[str, Any]] = None) -> None:
        if not self.enabled:
            return

        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "event_type": event_type,
            "payload": payload or {},
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
