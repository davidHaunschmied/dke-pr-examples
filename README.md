# dke-pr-examples

This repository contains three AI-based Python programs, ordered by complexity:
- **context_engineering/de_en_translator.py**: A German ↔ English translator that demonstrates how careful prompt design ("context engineering") steers LLM output — no tools, no external data.
- **rag/win_qa.py**: A Retrieval-Augmented Generation (RAG) system that answers questions about the JKU Linz "Wirtschaftsinformatik" Bachelor and Master programs using document-based retrieval and OpenRouter for embeddings and inference.
- **agent/todo_react_agent.py**: A TODO list assistant that manages tasks via natural language, using autonomous tool use (ReAct framework) and OpenRouter.

## Requirements

- Python 3.8 or newer
- API key for OpenRouter (environment variable `OPEN_ROUTER_API_KEY`) — required for all scripts. Get your free key at [OpenRouter](https://openrouter.ai/).
- Internet connection (for OpenRouter API)
- Recommended: Virtual environment (`python -m venv .venv`)

Install the required packages:
```cmd
pip install -r requirements.txt
```


## context_engineering/de_en_translator.py

A German ↔ English translator that demonstrates **context engineering** — the practice of carefully designing the system prompt to control LLM behaviour without any tools, retrieval, or external data.

### What is context engineering?

Context engineering is about crafting the *input context* (system prompt, user prompt, examples) so the model produces the output you want. A well-designed system prompt can enforce style, format, tone, and domain constraints — all without writing any extra code. This script shows how a single, carefully worded system prompt turns a general-purpose LLM into a specialised translator.

### How to start

```cmd
python context_engineering/de_en_translator.py
```

The script prints its system prompt at startup so you can see exactly which instructions shape the model's output.

### Example output

```
============================================================
  German ↔ English Translator  (Context Engineering Demo)
============================================================

--- System prompt (this is the context engineering) ---
You are an expert German ↔ English translator. Follow these rules strictly:
1. AUTO-DETECT the input language. ...
--- End of system prompt ---

Enter German text to get English, or English text to get German.
Type 'quit' or 'exit' to stop.

Text: Da steppt der Bär!
  → That's where the party is!

Text: I'm looking forward to the weekend.
  → Ich freue mich auf das Wochenende.
```

### Key prompt-engineering techniques used

| Technique | Example from the system prompt |
|---|---|
| Role assignment | "You are an expert German ↔ English translator" |
| Numbered rules | Explicit, ordered list of constraints |
| Tone preservation | "Formal input → formal output" |
| Idiom handling | "Find an equivalent idiom rather than translating literally" |
| Output formatting | "Output ONLY the translated text. No explanations, no notes." |

## agent/todo_react_agent.py

A TODO list assistant that lets you manage tasks using **natural language**. Under the hood it follows the ReAct framework (Thought → Action → Observation loop), autonomously choosing CRUD tools to fulfil each request.

### Available tools

| Tool | Description |
|---|---|
| `add_todo(text)` | Adds a new todo item |
| `list_todos()` | Lists all todos with IDs and status |
| `complete_todo(id)` | Marks a todo as done |
| `delete_todo(id)` | Permanently deletes a todo |

Todos are persisted in `agent/todos.json` so they survive restarts.

### How to start

```cmd
python agent/todo_react_agent.py
```

### Process

1. The system prompt and available tools are shown at startup.
2. You type natural-language requests (e.g. "Add buy groceries to my list").
3. The agent reasons step by step (Thought, Action, Observation) and calls the appropriate tools.
4. Once the request is fulfilled, the agent responds with a friendly summary.

## rag/win_qa.py

A document-based RAG (Retrieval-Augmented Generation) system that answers questions about the JKU Linz **Wirtschaftsinformatik** Bachelor and Master programs.

### How it works

1. **Program selection** – At startup you choose between the Bachelor and Master curriculum.
2. **PDF extraction** – The official curriculum PDF (stored in `db/`) is parsed with PyPDF2.
3. **Chunking** – The extracted text is split into overlapping 500-character chunks.
4. **Embedding** – Each chunk is embedded via the OpenRouter API (`nvidia/llama-nemotron-embed-vl-1b-v2:free`). Embeddings are cached in a JSON file inside `db/` so they only need to be computed once.
5. **Retrieval** – When you ask a question, your query is embedded and compared to all chunk embeddings via cosine similarity. The top 3 most relevant chunks (above a similarity threshold) are returned.
6. **Generation** – The retrieved chunks plus your question are sent to an OpenRouter LLM (`nvidia/nemotron-3-nano-30b-a3b:free`) which generates the answer.

### Project structure

```
rag/
  win_qa.py                          # Main RAG script
  db/
    win_bachelor.pdf                 # Bachelor curriculum PDF
    win_bachelor_embeddings.json     # Cached embeddings for Bachelor
    win_master.pdf                   # Master curriculum PDF
    win_master_embeddings.json       # Cached embeddings for Master
```

### How to start

First, set your OpenRouter API key (or add it to a `.env` file):
```cmd
$env:OPEN_ROUTER_API_KEY="your_api_key_here"
```

Then run the script:
```cmd
python rag/win_qa.py
```

On the first run for a given program, the PDF will be chunked and embedded via the OpenRouter API. Subsequent runs use the cached embeddings JSON.

### Example output

```
============================================================
  Wirtschaftsinformatik @ JKU - RAG Q&A System
============================================================

Please select the program type:
  master: Masterstudium Wirtschaftsinformatik
  bachelor: Bachelorstudium Wirtschaftsinformatik
Your choice (master/bachelor): master

Selected: Masterstudium Wirtschaftsinformatik

Loaded 246 cached chunks from db/win_master_embeddings.json

Ready! Ask questions about the Wirtschaftsinformatik program.
Type 'quit' or 'exit' to stop.

Your question: Welche Pflichtfächer gibt es?

Retrieving relevant passages ...
  Retrieved 3 relevant chunks (similarities: [0.229, 0.207, 0.201])
Generating answer ...

=== Answer ===
Die Pflichtfächer im Masterstudium Wirtschaftsinformatik umfassen ...
```

### Configuration

You can adjust these constants at the top of `win_qa.py`:

| Constant | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 500 | Characters per chunk |
| `CHUNK_OVERLAP` | 100 | Overlap between consecutive chunks |
| `TOP_K` | 3 | Number of similar chunks to retrieve |
| `SIMILARITY_THRESHOLD` | 0.10 | Minimum cosine similarity for retrieval |
| `INFERENCE_MODEL` | `nvidia/nemotron-3-nano-30b-a3b:free` | OpenRouter model for answer generation |
| `EMBEDDING_MODEL` | `nvidia/llama-nemotron-embed-vl-1b-v2:free` | OpenRouter model for embeddings |

## Notes

- The three scripts illustrate increasing levels of complexity: **context engineering** (prompt design only) → **RAG** (retrieval + generation) → **agent** (autonomous tool use).
- All scripts use OpenRouter with free models. Set `OPEN_ROUTER_API_KEY` in your `.env` file or as an environment variable.

---

Enjoy experimenting!
