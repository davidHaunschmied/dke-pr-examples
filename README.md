# dke-pr-examples

This repository contains two AI-based Python programs:
- **llm-agent.py**: An autonomous agent that solves tasks using tools and Google Gemini.
- **rag.py**: A Retrieval-Augmented Generation (RAG) system that answers questions about the JKU Linz "Wirtschaftsinformatik" Bachelor and Master programs using document-based retrieval and OpenRouter for embeddings and inference.

## Requirements

- Python 3.8 or newer
- API key for Google Gemini (environment variable `GEMINI_API_KEY`) - required for llm-agent.py
- API key for OpenRouter (environment variable `OPEN_ROUTER_API_KEY`) - required for openrouter.py and rag.py
- Internet connection (for OpenRouter API and Gemini API)
- Recommended: Virtual environment (`python -m venv .venv`)

Install the required packages:
```cmd
pip install -r requirements.txt
```


## llm-agent.py

The agent solves tasks using tools such as a calculator or web search. It communicates in a Thought → Action → Observation loop, following the ReAct framework.

### How to start

```cmd
python llm-agent.py
```

### Process

1. After starting, you are prompted to enter a task (e.g., `What is 15 * 7?`).
2. The agent uses available tools to solve the task step by step.
3. The process follows the ReAct framework:
   - **Thought**: The agent explains its reasoning for the next step.
   - **Action**: The agent chooses a tool and provides input.
   - **Observation**: The agent receives the tool's output and continues reasoning.
4. The final answer is given with an explanation of the reasoning and tool results.

### Example output

```
Enter your task: What is 3 * 2?

--- Iteration 1 ---
Current memory: []

LLM Output:
Thought: I need to calculate 3 * 2 to answer the question. I will use the calculator tool for this.
Action: calculator(3 * 2)

Observation: 6

--- Iteration 2 ---
Current memory: [{'Thought': 'I need to calculate 3 * 2 to answer the question. I will use the calculator tool for this.', 'Action': 'calculator(3 * 2)', 'Observation': '6'}]

LLM Output:
Thought: The calculator returned 6, which means the answer to 3 * 2 is 6. I am explaining this result so the user understands how I arrived at the answer.
Action: final_answer(6)

=== Final Answer ===
6

Final Result: 6
```

**Explanation:**
The agent follows the ReAct framework, reasoning about each step, choosing actions, and observing results before providing the final answer with a short explanation.

## rag.py

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
python rag.py
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

You can adjust these constants at the top of `rag.py`:

| Constant | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 500 | Characters per chunk |
| `CHUNK_OVERLAP` | 100 | Overlap between consecutive chunks |
| `TOP_K` | 3 | Number of similar chunks to retrieve |
| `SIMILARITY_THRESHOLD` | 0.10 | Minimum cosine similarity for retrieval |
| `INFERENCE_MODEL` | `nvidia/nemotron-3-nano-30b-a3b:free` | OpenRouter model for answer generation |
| `EMBEDDING_MODEL` | `nvidia/llama-nemotron-embed-vl-1b-v2:free` | OpenRouter model for embeddings |

## Notes

- The Gemini API key must be set as the environment variable `GEMINI_API_KEY` for llm-agent.py.
- The OpenRouter API key must be set as the environment variable `OPEN_ROUTER_API_KEY` for rag.py. Get your free API key at [OpenRouter](https://openrouter.ai/).
- The web search tool in the agent is a mock and does not provide real search results.

---

Enjoy experimenting!
