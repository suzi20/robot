# MCP + Agentic RAG 智能聊天助手

这是一个基于 Qwen-Agent、DeepSeek、MCP、ChromaDB 和 FastEmbed 构建的智能体问答系统。项目支持本地文档知识库、Agentic RAG 检索增强、长期语义记忆、MCP 工具扩展、可观测日志和离线评测。

## 核心能力

- **本地知识库 RAG**：支持 PDF/TXT/MD 文档导入、递归目录批量导入、文本切块、embedding 生成和 ChromaDB 向量检索。
- **Agentic RAG**：在普通 RAG 基础上加入 Query 改写、多路召回、候选去重、轻量重排和置信度诊断。
- **MCP 工具扩展**：基于 qwen-agent 注册工具，可扩展接入 Web Search、数据库、文件系统等 MCP 服务。
- **长期语义记忆**：将历史对话写入独立向量集合，并在后续对话中按语义召回相关记忆。
- **可观测性与评测**：通过 JSONL 记录 Agent 调用、RAG 检索、Memory 读写和评测结果，支持离线评测报告生成。
- **可配置聊天人设**：支持配置助手名称、对话风格、短期聊天历史和流式输出。

## 项目结构

```text
.
├── agent/              # Agent 组装、系统提示词、知识库工具
├── config/             # 应用配置
├── data/               # 示例评测集
├── mcp_config/         # MCP 服务配置
├── memory/             # 长期语义记忆
├── observability/      # JSONL 可观测日志
├── rag/                # 文档加载、切块、Embedding、向量库、检索管线
├── main.py             # CLI 入口
└── requirements.txt
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

复制环境变量模板：

```bash
copy .env.example .env
```

在 `.env` 中填写自己的 DeepSeek API Key：

```text
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

> 注意：`.env` 已加入 `.gitignore`，不要把真实 Key 上传到公开仓库。

### 3. 导入知识库

导入单个文档：

```bash
python main.py --ingest data/report.pdf
```

导入目录中的 PDF/TXT/MD：

```bash
python main.py --ingest data/docs
```

向量库默认保存在 `knowledge_base/`，该目录不会提交到 Git。

### 4. 启动聊天

```bash
python main.py
```

单轮查询：

```bash
python main.py "这个项目支持哪些能力？"
```

清空知识库：

```bash
python main.py --reindex
```

### 5. 运行离线评测

```bash
python main.py --eval data/eval.sample.json
```

评测报告默认输出到：

```text
logs/eval_report.json
```

## 配置说明

主要配置在 `config/settings.yaml`：

```yaml
retrieval_agentic_enabled: true
retrieval_multi_query_count: 3
retrieval_candidate_multiplier: 3

long_term_memory_enabled: true
memory_collection_name: long_term_memories

observability_enabled: true
observability_log_path: ./logs/agent_events.jsonl
```

## 简历亮点表达

可以概括为：

> 基于 MCP + Agentic RAG + 长期记忆 + 可观测评测体系构建智能体问答系统，实现文档知识库、查询改写、多路召回、候选重排、语义记忆召回和离线评测闭环。

## 安全说明

以下文件和目录不会上传到 Git：

- `.env`
- `knowledge_base/`
- `logs/`
- `data/chat_history.json`
- `__pycache__/`

