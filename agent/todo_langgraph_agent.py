"""
LLM Agent: TODO List Assistant using LangGraph's prebuilt ReAct agent.

Functionally identical to todo_react_agent.py, but uses the LangGraph framework
instead of a hand-rolled Thought → Action → Observation loop.  Comparing both
files shows what a framework like LangGraph automates for you:

  todo_react_agent.py (manual)          todo_langgraph_agent.py (framework)
  ──────────────────────────────        ──────────────────────────────────────
  - hand-parsed Thought/Action          - LangGraph handles the ReAct loop
  - manual memory list                  - built-in message state
  - custom system prompt with           - tools declared as plain functions
    format examples                       with @tool decorator
  - regex-based output parsing          - automatic tool calling via LLM
  - explicit iteration loop             - graph.invoke() runs until done

Uses:
- LangGraph + LangChain for the agent framework
- OpenRouter (via ChatOpenAI) for LLM inference
- A simple JSON file for persistent TODO storage (shared with manual agent)
"""

import os
import json

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OPEN_ROUTER_API_KEY = os.environ.get("OPEN_ROUTER_API_KEY")

llm = ChatOpenAI(
    model="nvidia/nemotron-3-nano-30b-a3b:free",
    api_key=OPEN_ROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

SYSTEM_PROMPT = (
    "You are a helpful TODO list assistant. "
    "Use the provided tools to manage the user's tasks. "
    "Always respond in a friendly, concise way. "
    "When you list todos, format them nicely for the user."
)

# ---------------------------------------------------------------------------
# TODO storage (simple JSON file — shared with the manual agent)
# ---------------------------------------------------------------------------

TODO_FILE = os.path.join(os.path.dirname(__file__), "todos.json")


def _load_todos() -> list[dict]:
    """Load todos from the JSON file, or return an empty list."""
    if os.path.exists(TODO_FILE):
        with open(TODO_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_todos(todos: list[dict]):
    """Persist todos to the JSON file."""
    with open(TODO_FILE, "w", encoding="utf-8") as f:
        json.dump(todos, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Tools (declared with @tool so LangGraph can use them automatically)
# ---------------------------------------------------------------------------

@tool
def add_todo(description: str) -> str:
    """Add a new todo item. Input: the description text."""
    todos = _load_todos()
    new_id = max((t["id"] for t in todos), default=0) + 1
    todo = {"id": new_id, "description": description.strip(), "done": False}
    todos.append(todo)
    _save_todos(todos)
    return f"Added todo #{new_id}: '{todo['description']}'"


@tool
def list_todos() -> str:
    """List all current todos with their IDs and done/open status."""
    todos = _load_todos()
    if not todos:
        return "The TODO list is empty."
    lines = []
    for t in todos:
        status = "✓" if t["done"] else "○"
        lines.append(f"  [{status}] #{t['id']}: {t['description']}")
    return "Current TODOs:\n" + "\n".join(lines)


@tool
def complete_todo(todo_id: int) -> str:
    """Mark a todo as done. Input: the todo ID number."""
    todos = _load_todos()
    for t in todos:
        if t["id"] == todo_id:
            t["done"] = True
            _save_todos(todos)
            return f"Marked todo #{todo_id} ('{t['description']}') as done."
    return f"Error: No todo with ID #{todo_id} found."


@tool
def delete_todo(todo_id: int) -> str:
    """Delete a todo permanently. Input: the todo ID number."""
    todos = _load_todos()
    for i, t in enumerate(todos):
        if t["id"] == todo_id:
            removed = todos.pop(i)
            _save_todos(todos)
            return f"Deleted todo #{todo_id}: '{removed['description']}'."
    return f"Error: No todo with ID #{todo_id} found."


# ---------------------------------------------------------------------------
# Build the LangGraph ReAct agent
# ---------------------------------------------------------------------------

tools = [add_todo, list_todos, complete_todo, delete_todo]

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=SYSTEM_PROMPT,
)


# ---------------------------------------------------------------------------
# Main interactive loop
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  TODO List Assistant  (LangGraph Agent Demo)")
    print("=" * 60)

    print(f"\n--- System prompt ---")
    print(SYSTEM_PROMPT)
    print("--- End of system prompt ---")
    print(f"\nTools: {', '.join(t.name for t in tools)}\n")

    print("Manage your TODO list using natural language.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # Invoke the LangGraph agent
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]}
        )

        # The last message in the result is the agent's final response
        final_message = result["messages"][-1]
        print(f"\nAssistant: {final_message.content}\n")

