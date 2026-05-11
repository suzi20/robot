"""
Qwen-Agent MCP + RAG 智能问答助手
===================================
基于 DeepSeek LLM + qwen-agent 框架，整合 MCP 工具调用和 RAG 知识库检索。

用法:
  # 交互式问答
  python main.py

  # 单轮查询
  python main.py "请介绍一下公司的发展历程"

  # 导入文档到知识库
  python main.py --ingest data/report.pdf
  python main.py --ingest data/docs

  # 重建知识库索引
  python main.py --reindex
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from config.settings import Settings
from agent.assistant import RAGMCPAssistant

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.WARNING)
logging.getLogger().setLevel(logging.WARNING)
_qwen_logger = logging.getLogger("qwen_agent_logger")
_qwen_logger.setLevel(logging.WARNING)
_qwen_logger.propagate = False
for _handler in _qwen_logger.handlers:
    _handler.setLevel(logging.WARNING)


def _load_chat_history(settings: Settings) -> list[dict]:
    """Load recent chat history for interactive chatbot mode."""
    if not settings.chat_memory_enabled:
        return []

    path = Path(settings.chat_history_path)
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    if not isinstance(data, list):
        return []

    messages = [
        item for item in data
        if isinstance(item, dict)
        and item.get("role") in {"user", "assistant"}
        and isinstance(item.get("content"), str)
    ]
    return messages[-settings.chat_history_max_messages:]


def _save_chat_history(settings: Settings, history: list[dict]) -> None:
    """Persist recent chat history for the next interactive session."""
    if not settings.chat_memory_enabled:
        return

    path = Path(settings.chat_history_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    recent = history[-settings.chat_history_max_messages:]
    path.write_text(json.dumps(recent, ensure_ascii=False, indent=2), encoding="utf-8")


def _clear_chat_history(settings: Settings) -> None:
    """Clear persisted chat history."""
    path = Path(settings.chat_history_path)
    if path.exists():
        path.unlink()


def _load_eval_cases(eval_path: str) -> list[dict]:
    """Load eval cases from JSON array or JSONL."""
    path = Path(eval_path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        return data if isinstance(data, list) else []
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def run_eval(assistant: RAGMCPAssistant, settings: Settings, eval_path: str) -> None:
    """Run a lightweight local evaluation set."""
    cases = _load_eval_cases(eval_path)
    results = []

    for index, case in enumerate(cases, 1):
        question = case.get("question") or case.get("query") or ""
        expected_keywords = case.get("expected_keywords", [])
        messages = [{"role": "user", "content": question}]
        started = time.perf_counter()
        answer = _collect_response(assistant, messages, stream=False, emit=False)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)

        answer_for_match = answer.lower()
        matched = [
            keyword for keyword in expected_keywords
            if str(keyword).lower() in answer_for_match
        ]
        passed = len(matched) == len(expected_keywords)
        result = {
            "index": index,
            "question": question,
            "passed": passed,
            "matched_keywords": matched,
            "expected_keywords": expected_keywords,
            "answer_chars": len(answer),
            "answer_preview": answer[:500],
            "latency_ms": latency_ms,
        }
        results.append(result)
        print(f"[{index}/{len(cases)}] {'PASS' if passed else 'FAIL'} {question}")

    pass_count = sum(1 for item in results if item["passed"])
    report = {
        "eval_path": eval_path,
        "total": len(results),
        "passed": pass_count,
        "pass_rate": round(pass_count / len(results), 4) if results else 0.0,
        "results": results,
    }

    report_path = Path(settings.eval_report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    assistant.trace_logger.log("eval.run", {"eval_path": eval_path, "total": len(results), "passed": pass_count})
    print(f"[OK] 评测完成: {pass_count}/{len(results)} 通过，报告已写入 {report_path}")


def _message_value(message, key: str, default=None):
    """Read a field from qwen-agent Message objects or dict responses."""
    if isinstance(message, dict):
        return message.get(key, default)
    return getattr(message, key, default)


def _collect_response(
    assistant: RAGMCPAssistant,
    messages: list[dict],
    stream: bool,
    emit: bool = True,
) -> str:
    """Collect qwen-agent output once and optionally print deltas."""
    last_content = ""
    last_reasoning = ""

    for chunk in assistant.run(messages, stream=stream):
        assistant_messages = [
            msg for msg in chunk
            if _message_value(msg, "role") == "assistant" and _message_value(msg, "content")
        ]
        if not assistant_messages:
            continue

        msg = assistant_messages[-1]
        content = _message_value(msg, "content", "")
        if not isinstance(content, str):
            content = str(content)

        if content.startswith(last_content):
            delta = content[len(last_content):]
        else:
            delta = content

        if delta:
            if emit:
                print(delta, end="", flush=True)
            last_content = content

        reasoning = _message_value(msg, "reasoning_content", "")
        if reasoning and reasoning != last_reasoning:
            if emit:
                print(f"\n[思考] {reasoning}")
            last_reasoning = reasoning

    return last_content


def _stream_response(assistant: RAGMCPAssistant, messages: list[dict], stream: bool) -> str:
    """Print qwen-agent output once and return the final assistant text."""
    return _collect_response(assistant, messages, stream=stream, emit=True)


def print_banner(settings: Settings):
    """打印启动横幅。"""
    print("=" * 60)
    print(f"  {settings.assistant_name} - AI 聊天机器人")
    print("  输入 exit/quit 退出，输入 /reset 清空聊天记忆")
    print("=" * 60)


def run_interactive(assistant: RAGMCPAssistant, settings: Settings):
    """交互式问答模式。

    Args:
        assistant: 问答助手实例。
        settings: 配置对象。
    """
    print_banner(settings)

    # 显示已加载的配置信息
    kb_count = assistant.vector_store.count()
    mcp_servers = assistant.mcp_servers
    print(f"  模型: {settings.llm_model}")
    print(f"  知识库: {kb_count} 个文档块")
    print(f"  MCP 服务: {len(mcp_servers)} 个已注册" if mcp_servers else "  MCP 服务: 未配置")
    print(f"  聊天记忆: {'开启' if settings.chat_memory_enabled else '关闭'}")
    print()

    history = _load_chat_history(settings)
    if history:
        print(f"  已加载最近 {len(history)} 条聊天记录。")

    while True:
        try:
            query = input(f"\n{settings.user_name}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n再见。")
            break

        if query.lower() in ("exit", "quit", "退出"):
            print("再见。")
            break
        if query.lower() == "/reset":
            history = []
            _clear_chat_history(settings)
            assistant.clear_memory()
            print("聊天记忆已清空。")
            continue
        if not query:
            continue

        history.append({"role": "user", "content": query})

        print(f"\n{settings.assistant_name}: ", end="", flush=True)

        try:
            full_response = _stream_response(assistant, history, stream=settings.enable_streaming)
        except Exception as e:
            print(f"\n[错误] {e}")
            history.pop()
            continue

        history.append({"role": "assistant", "content": full_response})
        assistant.remember(query, full_response)
        history = history[-settings.chat_history_max_messages:]
        _save_chat_history(settings, history)
        print()


def run_single_query(assistant: RAGMCPAssistant, query: str, settings: Settings):
    """单轮问答模式。

    Args:
        assistant: 问答助手实例。
        query: 用户问题。
        settings: 配置对象。
    """
    messages = [{"role": "user", "content": query}]
    try:
        full_response = _stream_response(assistant, messages, stream=settings.enable_streaming)
        assistant.remember(query, full_response)
        print()
    except Exception as e:
        print(f"\n[错误] {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Qwen-Agent MCP + RAG 智能问答助手",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                        # 交互式问答
  python main.py "公司去年的营收是多少"  # 单轮查询
  python main.py -i data/report.pdf     # 导入文档
  python main.py -i data/docs           # 批量导入目录
  python main.py --eval data/eval.json  # 运行评测集
  python main.py --reindex              # 重建索引
        """,
    )
    parser.add_argument(
        "query", nargs="?",
        help="单轮查询问题（省略则进入交互模式）",
    )
    parser.add_argument(
        "--ingest", "-i",
        help="将文件或目录导入知识库（支持 PDF/TXT/MD）",
        metavar="PATH",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="清空并重建知识库索引",
    )
    parser.add_argument(
        "--config", "-c",
        default="config/settings.yaml",
        help="配置文件路径 (默认: config/settings.yaml)",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="禁用流式输出",
    )
    parser.add_argument(
        "--eval",
        metavar="FILE",
        help="运行 RAG/Agent 离线评测集（支持 JSON 数组或 JSONL）",
    )

    args = parser.parse_args()

    # 1. 加载配置（YAML + .env / 环境变量）
    try:
        settings = Settings.load(args.config, env_file=".env")
    except Exception as e:
        print(f"[配置错误] {e}", file=sys.stderr)
        sys.exit(1)

    # 命令行覆盖配置
    if args.no_stream:
        settings.enable_streaming = False

    # 2. 初始化助手
    try:
        assistant = RAGMCPAssistant(settings)
    except ValueError as e:
        print(f"[配置错误] {e}", file=sys.stderr)
        sys.exit(1)

    if args.eval:
        run_eval(assistant, settings, args.eval)
        return

    # 3. 处理 --ingest (文档导入)
    if args.ingest:
        ingest_path = Path(args.ingest)
        print(f"正在导入知识库: {args.ingest}")
        try:
            if ingest_path.is_dir():
                n_chunks = assistant.retriever.ingest_directory(args.ingest)
            else:
                n_chunks = assistant.retriever.ingest_document(args.ingest)
            total = assistant.vector_store.count()
            print(f"[OK] 导入完成: {n_chunks} 个文本块已添加 (知识库共 {total} 个块)")
        except Exception as e:
            print(f"[错误] 导入失败: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # 4. 处理 --reindex (重建索引)
    if args.reindex:
        confirm = input("确定要清空知识库吗? (y/N): ")
        if confirm.lower() == "y":
            assistant.vector_store.delete_collection()
            print("[OK] 知识库已清空，请重新导入文档。")
        else:
            print("已取消。")
        return

    # 5. 问答模式
    if args.query:
        run_single_query(assistant, args.query, settings)
    else:
        run_interactive(assistant, settings)


if __name__ == "__main__":
    main()
