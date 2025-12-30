# dke-pr-examples

This repository contains three AI-based Python programs:
- **llm-agent.py**: An autonomous agent that solves tasks using tools and Google Gemini.
- **rag.py**: A Retrieval-Augmented Generation (RAG) system that fetches weather data and answers questions using Gemini.
- **openrouter.py**: A demonstration script for using OpenRouter API and testing rate limits of free models.

## Requirements

- Python 3.8 or newer
- API key for Google Gemini (environment variable `GEMINI_API_KEY`) - required for llm-agent.py and rag.py
- API key for OpenRouter (environment variable `OPEN_ROUTER_API_KEY`) - required for openrouter.py
- Internet connection (for weather data and Gemini API)
- Recommended: Virtual environment (`python -m venv .venv`)

Install the required packages:
```cmd
pip install -r requirements.txt
```

## openrouter.py

This script demonstrates how to use the OpenRouter API, which provides access to various LLM models through a unified interface. The primary focus is to showcase API usage and test the rate limits of free models.

### Purpose

- **OpenRouter Integration**: Shows how to connect to OpenRouter using the OpenAI-compatible client
- **Rate Limit Testing**: Runs 100 iterations to test throughput and token limits of free models
- **Model Evaluation**: Currently configured with `nvidia/nemotron-3-nano-30b-a3b:free`, which appears to be the best available free model at the moment

### How to start

First, set your OpenRouter API key:
```cmd
$env:OPEN_ROUTER_API_KEY="your_api_key_here"
```

Then run the script:
```cmd
python openrouter.py
```

### What it does

1. Sends a meeting notes summarization task to the selected model 100 times
2. Tracks input and output token usage for each iteration
3. Calculates total token consumption across all iterations
4. Tests the model's consistency and rate limit handling

### Example output

```
Iteration 1: Input tokens: 808, Output tokens: 997
Total input tokens so far: 808, Total output tokens so far: 997

Iteration 2: Input tokens: 808, Output tokens: 945
Total input tokens so far: 1616, Total output tokens so far: 1942

...

Iteration 100: Input tokens: 808, Output tokens: 997
Total input tokens so far: 80800, Total output tokens so far: 92944
```

### Current Model

The script uses `nvidia/nemotron-3-nano-30b-a3b:free`, which:
- Is completely free to use through OpenRouter
- Handles ~80K input tokens and ~93K output tokens in 100 iterations without running into limits
- Provides consistent performance for summarization tasks
- Appears to be the best free model currently available on OpenRouter

You can change the model by modifying the `model` parameter in the `client.chat.completions.create()` call.

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

The RAG system answers weather-related questions for various cities.

### How to start

```cmd
python rag.py
```

### Process

1. Select a city from the list (e.g., `tokyo`).
2. Enter a weather-related question (e.g., `How warm is it?`).
3. The program fetches current weather data and generates an answer using Gemini.

### Example output

```
Available cities: london, new york, tokyo, sydney, paris, berlin, rome, moscow

Enter city name: tokyo

Ask a question about the weather: How warm is it?

Retrieving weather data...
Weather in tokyo:
- Temperature: 22.8°C
- Humidity: 61%
- Wind speed: 5.5 km/h
- Next hours temperatures: [20.2, 20.8, 21.4, 22.6, 23.4, 23.5]
- Precipitation probability: [0, 0, 0, 0, 0, 3]%

Generating answer...
=== Answer ===
It is 22.8°C.
```

## Notes

- The Gemini API key must be set as the environment variable `GEMINI_API_KEY` for llm-agent.py and rag.py.
- The OpenRouter API key must be set as the environment variable `OPEN_ROUTER_API_KEY` for openrouter.py. Get your free API key at [OpenRouter](https://openrouter.ai/).
- Weather data is retrieved from [Open-Meteo](https://open-meteo.com/).
- The web search tool in the agent is a mock and does not provide real search results.

---

Enjoy experimenting!
