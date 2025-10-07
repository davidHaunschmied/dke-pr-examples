# dke-pr-examples

This repository contains two AI-based Python programs:
- **llm-agent.py**: An autonomous agent that solves tasks using tools and Google Gemini.
- **rag.py**: A Retrieval-Augmented Generation (RAG) system that fetches weather data and answers questions using Gemini.

## Requirements

- Python 3.8 or newer
- API key for Google Gemini (environment variable `GEMINI_API_KEY`)
- Internet connection (for weather data and Gemini API)
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

- The Gemini API key must be set as the environment variable `GEMINI_API_KEY`.
- Weather data is retrieved from [Open-Meteo](https://open-meteo.com/).
- The web search tool in the agent is a mock and does not provide real search results.

---

Enjoy experimenting!
