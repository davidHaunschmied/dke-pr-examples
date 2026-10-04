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

A TODO list assistant that lets you manage tasks using **natural language**. The model decides which tool to call; the surrounding code is the **harness**:

```
model proposes a tool call -> harness executes it -> result goes back into the context -> repeat
```

Modern models have reasoning built in and return *structured* tool calls (name + JSON arguments), so the older "Thought: / Action:" text format (ReAct) no longer has to be prompted and regex-parsed.

### Available tools

| Tool | Description |
|---|---|
| `add_todo(description)` | Adds a new todo item |
| `list_todos()` | Lists all todos with IDs and status |
| `complete_todo(todo_id)` | Marks a todo as done |
| `delete_todo(todo_id)` | Permanently deletes a todo (**asks for confirmation first**) |

Each tool has a name, a description and a parameter schema; the description is part of the prompt. Todos are persisted in `agent/todos.json` so they survive restarts.

### Harness safeguards

Errors multiply over steps (95 % per step → 36 % over 20 steps), so the loop is kept short and controlled:

- hard iteration limit (`MAX_ITERATIONS`)
- tool errors are returned to the model as readable messages
- destructive actions need a confirmation from the user
- every run is logged as one line in `traces/todo_react_agent.jsonl`

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

Both agents share the same `agent/todos.json` file, so you can switch between them freely. Neither needs a framework: LangGraph is optional and mainly pays off for state machines, human-in-the-loop and persistence.

## rag/win_qa.py

A document-based RAG (Retrieval-Augmented Generation) system that answers questions about the JKU Linz **Wirtschaftsinformatik** Bachelor and Master programs.

### How it works

1. **Program selection** – At startup you choose between the Bachelor and Master curriculum.
2. **PDF extraction** – The official curriculum PDF (stored in `db/`) is parsed page by page with PyPDF2, so every chunk knows its page.
3. **Chunking** – Text is cut at sentence boundaries (about 600 characters, the last sentence repeats in the next chunk) instead of at a fixed character count.
4. **Embedding** – Each chunk is embedded via the OpenRouter API (`nvidia/llama-nemotron-embed-vl-1b-v2:free`), prefixed with a short classification ("Masterstudium Wirtschaftsinformatik, Seite 12: …"). Embeddings are cached in a JSON file inside `db/` and rebuilt when the PDF or the chunking changes.
5. **Hybrid retrieval** – Dense search (cosine similarity) finds similar *meaning*, BM25 finds exact *terms* (identifiers, names, numbers). The two rankings are merged with **Reciprocal Rank Fusion**; the top 3 chunks go to the LLM.
6. **Refusal** – If no chunk reaches the similarity threshold, the program answers "keine Information" without calling the LLM — better no answer than an irrelevant one.
7. **Generation** – The passages are passed as delimited *data* (`<passage page="…">`); the LLM must cite pages (`[Seite 12]`) and may refuse if the passages do not contain the answer.
8. **Trace** – Question, hits with scores, final prompt, answer, tokens and latency are appended to `traces/rag.jsonl`.

Not included on purpose (KISS): a cross-encoder reranker (retrieve ~50 cheaply, re-sort to the top 5) — the natural next step if retrieval metrics show the right chunk found but ranked too low.

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

Without measurement every optimization is a guess — especially for non-deterministic systems. The evaluation measures **components separately**, retrieval first:

| Question | Measure | Where |
|---|---|---|
| Does retrieval find the right pages? | Recall@k, MRR (a hit is relevant if its page is in `expected_pages`) | `eval_rag.py --retrieval` |
| Does the whole system answer correctly? | Pass rate per case over 3 runs (answer contains all `key_phrases`) | `eval_rag.py` |
| Does it refuse when it should? | Negative cases must answer "keine Information" | `eval_rag.py` |

`eval/golden.jsonl` holds one test case per line (`id`, `program`, `question`, `expected_pages`, `key_phrases`, `category`) with fact questions, one multi-step question and 20 % negative cases (questions the curriculum cannot answer). Without negative cases you build a system that never learns to refuse.

```cmd
python eval/eval_rag.py --retrieval
python eval/eval_rag.py
```

Notes: a single run is a sample, not a result — `temperature=0` does not make LLM output reproducible, so each case runs three times and unstable cases (passed once, failed once) are the interesting ones. With 15 cases, report tendencies and error classes, not decimals. Check by hand whether the key-phrase matching is too strict before trusting a number. Traces of every run are in `traces/rag.jsonl`.

## Notes

- The scripts illustrate increasing levels of autonomy: **context engineering** (you fill the context) → **RAG** (a pipeline fills it) → **agent** (the model fills it in a loop). Choose the lowest level that solves the problem.
- All scripts use OpenRouter with free models. Set `OPEN_ROUTER_API_KEY` in your `.env` file or as an environment variable.
- **AI notice (EU AI Act, Art. 50):** every program tells the user at startup that they are interacting with an AI system. Any application with a chat interface needs this line.
- **Security:** models do not reliably separate instructions from data, so retrieved text is passed as delimited data. Tools that write or delete (`delete_todo`) ask for confirmation first.
- Runtime data (`traces/`, `agent/todos.json`) is git-ignored.

---

Enjoy experimenting!
