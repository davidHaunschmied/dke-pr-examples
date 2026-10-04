# dke-pr-examples

This repository contains four AI-based Python programs. They walk along **one axis: who fills the model's context?**

| Level | Who fills the context? | Pattern | Program |
|---|---|---|---|
| 1 | You, fixed at design time | Prompt / context engineering | `context_engineering/de_en_translator.py` |
| 2 | A deterministic pipeline before the call | RAG ("search first, then generate") | `rag/win_qa.py` |
| 3 | The model itself, in a loop | Agent | `agent/todo_react_agent.py`, `agent/todo_langgraph_agent.py` |

Same model, same building blocks — the only difference is how much control you hand over. Capability and error rate rise together, so autonomy is a design decision, not a quality feature.

- **context_engineering/de_en_translator.py**: A German ↔ English translator that demonstrates how careful prompt design ("context engineering") steers LLM output — no tools, no external data.
- **rag/win_qa.py**: A Retrieval-Augmented Generation (RAG) system that answers questions about the JKU Linz "Wirtschaftsinformatik" Bachelor and Master programs using hybrid retrieval (dense + BM25), page citations and refusal when nothing relevant is found.
- **agent/todo_react_agent.py**: A TODO list assistant with a hand-written harness (native tool calling) — shows how an agent loop works under the hood.
- **agent/todo_langgraph_agent.py**: The same TODO assistant built with LangGraph — shows how a framework automates the agent loop.
- **eval/**: A golden dataset and an evaluation script (Recall@k, MRR, pass rate) for the RAG example.
- **tracing.py**: Shared helpers — the AI notice and JSONL traces (`traces/*.jsonl`, one line per run).

## Requirements

- Python 3.8 or newer
- API key for OpenRouter (environment variable `OPEN_ROUTER_API_KEY`) — required for all scripts. Get your free key at [OpenRouter](https://openrouter.ai/). (Free tiers change almost monthly; alternatives without credit card are Google AI Studio, Groq, Mistral and local Ollama — any OpenAI-compatible endpoint works by changing `base_url` and the model name. Plan a fallback on HTTP 429. Free tiers usually train on your inputs: send no personal data.)
- Internet connection (for OpenRouter API)
- Recommended: Virtual environment (`python -m venv .venv`)

Install the required packages:
```cmd
pip install -r requirements.txt
```


## context_engineering/de_en_translator.py

A German ↔ English translator that demonstrates **context engineering** — the practice of carefully designing the system prompt to control LLM behaviour without any tools, retrieval, or external data.

Context engineering means crafting the *input context* (system prompt, examples) so the model behaves as wanted — here one carefully worded system prompt turns a general LLM into a translator, with no tools or data.

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

A TODO list assistant for natural-language task management. The model picks tools; the surrounding code is the **harness**: model proposes a tool call → harness executes it → result goes back into the context → repeat. Native tool calling replaces the old "Thought:/Action:" text format.

### Available tools

| Tool | Description |
|---|---|
| `add_todo(description)` | Adds a new todo item |
| `list_todos()` | Lists all todos with IDs and status |
| `complete_todo(todo_id)` | Marks a todo as done |
| `delete_todo(todo_id)` | Permanently deletes a todo (**asks for confirmation first**) |

Todos are persisted in `agent/todos.json` so they survive restarts.

Safeguards in the harness: iteration limit (`MAX_ITERATIONS`), tool errors returned to the model, confirmation before `delete_todo`, one trace line per run.

### How to start

```cmd
python agent/todo_react_agent.py
```

## agent/todo_langgraph_agent.py

The **same TODO assistant**, but built with [LangGraph](https://langchain-ai.github.io/langgraph/) instead of a hand-written loop. Comparing the two files shows what a framework automates for you:

| Manual (`todo_react_agent.py`) | Framework (`todo_langgraph_agent.py`) |
|---|---|
| Tool schemas written as JSON | Tools declared with `@tool` decorator |
| Explicit `for` loop with `MAX_ITERATIONS` | `agent.invoke()` runs until done (`recursion_limit`) |
| Manual message list | Built-in message state |
| Own tool dispatch and error handling | Automatic tool execution |
| Own trace writing | Own trace writing (same helper) |

### How to start

```cmd
python agent/todo_langgraph_agent.py
```

Both agents share the same `agent/todos.json` file, so you can switch between them freely.

## rag/win_qa.py

A document-based RAG (Retrieval-Augmented Generation) system that answers questions about the JKU Linz **Wirtschaftsinformatik** Bachelor and Master programs.

### How it works

1. **Extract** the curriculum PDF page by page (PyPDF2), so chunks know their page.
2. **Chunk** at sentence boundaries (~600 characters, last sentence repeated).
3. **Embed** each chunk via OpenRouter (prefixed with program and page); cached in `db/`, rebuilt when PDF or chunking changes.
4. **Retrieve** hybrid: cosine similarity (meaning) + BM25 (exact terms), merged with Reciprocal Rank Fusion; top 3 chunks.
5. **Refuse** without an LLM call if no chunk reaches the similarity threshold.
6. **Generate** from the passages (passed as delimited data), citing pages like `[Seite 12]`.
7. **Trace** the run to `traces/rag.jsonl`.

A cross-encoder reranker is the natural next step if retrieval finds the right chunk but ranks it too low.

### Project structure

```
rag/
  win_qa.py                          # Main RAG script
  db/
    win_bachelor.pdf                 # Bachelor curriculum PDF
    win_bachelor_embeddings.json     # Cached embeddings for Bachelor
    win_master.pdf                   # Master curriculum PDF
    win_master_embeddings.json       # Cached embeddings for Master
eval/
  golden.jsonl                       # 15 test cases, 3 of them negative (20 %)
  eval_rag.py                        # Recall@k, MRR, pass rate over 3 runs
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

Loaded 297 cached chunks from db/win_master_embeddings.json

Ready! Ask questions about the Wirtschaftsinformatik program.
Type 'quit' or 'exit' to stop.

Your question: Welche Pflichtfächer gibt es?

Retrieving relevant passages ...
  Seite 12: dense=0.412 bm25=9.8 rrf=0.0328
  Seite 15: dense=0.371 bm25=4.1 rrf=0.0301
  Seite 11: dense=0.352 bm25=0.0 rrf=0.0164
Generating answer ...

=== Answer ===
Das Pflichtprogramm umfasst 36 ECTS ... [Seite 12]
```

### Configuration

You can adjust these constants at the top of `win_qa.py`:

| Constant | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 600 | Target characters per chunk (cut at sentence boundaries) |
| `TOP_K` | 3 | Number of chunks passed to the LLM |
| `CANDIDATES` | 20 | Candidates per search method before fusion |
| `RRF_K` | 60 | Reciprocal Rank Fusion constant |
| `SIMILARITY_THRESHOLD` | 0.10 | Minimum cosine similarity — below it the system refuses |
| `INFERENCE_MODEL` | `nvidia/nemotron-3-nano-30b-a3b:free` | OpenRouter model for answer generation |
| `EMBEDDING_MODEL` | `nvidia/llama-nemotron-embed-vl-1b-v2:free` | OpenRouter model for embeddings |

## Evaluation (eval/)

Components are measured separately, retrieval first. `eval/golden.jsonl` has 15 cases (`id`, `program`, `question`, `expected_pages`, `key_phrases`, `category`), 20 % of them negative (questions the curriculum cannot answer).

```cmd
python eval/eval_rag.py --retrieval   # Recall@k and MRR
python eval/eval_rag.py               # + pass rate over 3 runs per case
```

A single LLM run is a sample, not a result — hence 3 runs per case. With 15 cases, report tendencies, not decimals.

## Notes

- Levels of autonomy: **context engineering** → **RAG** → **agent**. Choose the lowest level that solves the problem.
- All scripts use OpenRouter with free models; set `OPEN_ROUTER_API_KEY` in `.env` or the environment.
- Every program shows an AI notice (EU AI Act, Art. 50). Retrieved text is passed as data, and deleting asks for confirmation.
- Runtime data (`traces/`, `agent/todos.json`) is git-ignored.

---

Enjoy experimenting!
