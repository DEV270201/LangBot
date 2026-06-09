# LangBot

A conversational AI chatbot built with **LangGraph** and **LangChain**, powered by a local **Ollama** LLM. It features tool calling (calculator and web search), streaming responses, PostgreSQL-backed conversation persistence, and a custom short-term memory system that summarizes older messages to stay within a token budget.

## Features

- **LangGraph agent loop** — memory management → chat → tools → chat (repeat as needed)
- **Tool calling** — arithmetic via a custom calculator tool; current events via DuckDuckGo search
- **Streaming UI** — Streamlit chat interface with live token streaming and tool-use status indicators
- **Conversation threads** — each chat session has a unique thread ID; past conversations are persisted and reloadable from the sidebar
- **Short-term memory** — when history exceeds a token threshold, older message clusters are summarized and trimmed instead of dropped silently
- **PostgreSQL checkpointing** — full graph state (messages + summary) is saved across sessions

## Architecture
<img width="641" height="282" alt="image" src="https://github.com/user-attachments/assets/0257a471-198f-4ca5-87cd-b84ebd1b4dbe" />

### Memory Archtecture
<img width="527" height="438" alt="image" src="https://github.com/user-attachments/assets/4860d39b-0f6c-4414-9679-41f8ee1cd55f" />


### Graph flow

1. **`manage_memory`** — counts tokens in the message history; if over threshold, summarizes evicted messages and removes them via LangGraph's `RemoveMessage` API
2. **`chat_node`** — invokes the LLM with a system prompt (including the running summary) and bound tools
3. **`tools`** — executes tool calls (calculator or search) when the LLM requests them
4. **Conditional edge** — loops back to `chat_node` after tool execution until the LLM produces a final answer

## Concepts & Technologies

| Concept | How it's used |
|---------|---------------|
| **LangGraph `StateGraph`** | Defines the chatbot as a directed graph of nodes with shared `ChatState` |
| **`ChatState` (TypedDict)** | Holds `messages` (with `add_messages` reducer) and a running `summary` string |
| **`ToolNode` + `tools_condition`** | LangGraph prebuilt components for tool routing |
| **`PostgresSaver` checkpointer** | Persists graph state per `thread_id` so conversations survive restarts |
| **Token budgeting** | `WINDOW`, `BUDGET`, `TRIM_THRESHOLD`, and `KEEP_TARGET` in `config.py` control when and how much history to trim |
| **Incremental summarization** | Evicted messages are folded into the existing summary (never re-summarized from scratch) |
| **Cluster-safe trimming** | Trims only at `HumanMessage` boundaries so tool-call clusters stay intact |
| **`ChatOllama`** | LangChain integration with Ollama for local inference |
| **HuggingFace tokenizer** | Accurate token counting via `apply_chat_template` (with a character-based fallback) |
| **Streamlit `write_stream`** | Streams LLM tokens to the UI; filters to only show `chat_node` output (hides summarization) |

## Stack Used
1. Python
2. LangChain
3. LangGraph
4. Cursor
5. PostgreSQL in Docker
6. Streamlit

## Prerequisites

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** (recommended) or pip
- **[Ollama](https://ollama.com/)** running locally with your chosen model pulled
- **PostgreSQL** database

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/chatbot.git
cd chatbot
```

### 2. Install dependencies
```bash
uv sync
```

### 3. Configure environment variables

Create ```Server/.env``` with the following:
```bash
OLLAMA_URL=http://localhost:11434/api/chat
LLM_MODEL=llama3.2          # or any model you have pulled in Ollama
DATABASE_URI=postgresql://user:password@localhost:5432/chatbot
TOKENIZER_NAME=meta-llama/Llama-3.2-3B  # HuggingFace tokenizer matching your model
```
### 4. Start Ollama
```bash
ollama serve
ollama pull llama3.2   # or your chosen model
```

### 5. Ensure PostgreSQL is running

Create a database (e.g. chatbot) and make sure the connection URI in .env is valid. The checkpointer runs setup() automatically on first launch to create required tables.

#### Running the App
From the project root:

```bash
streamlit run Client/ui.py
```
Open the URL Streamlit prints (typically http://localhost:8501).

### Using the UI
1. Type messages in the chat input at the bottom<br>
2. Click New Chat in the sidebar to start a fresh conversation (new thread ID)<br>
3. Click any previous thread ID in the sidebar to reload that conversation from PostgreSQL<br>

### Memory Tuning
Adjust these constants in ```Server/config.py``` to match your model's context window and hardware:

| Constant | Default | Purpose |
|----------|---------|---------|
| `WINDOW` | 2048 | Model context window size |
| `BUDGET` | 75% of window | How much of the window is allocated to history |
| `TRIM_THRESHOLD` | 90% of budget | Token count that triggers summarization |
| `KEEP_TARGET` | 25% of budget | Recent verbatim messages to keep after trimming |

<h3 align="center"><b>Developed with :heart: by <a href="https://github.com/DEV270201">Devansh Shah</a></b></h3>
