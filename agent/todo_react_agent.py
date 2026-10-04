"""
LLM Agent: TODO List Assistant with a hand-written harness (native tool calling).

Level 3 of the context axis: the model itself decides in a loop which tool to
call. The code around the model is the "harness" - it builds the context, calls
the model, executes tools, feeds results back and decides when to stop:

    model proposes tool call -> harness executes -> result back into context -> repeat

Modern models have reasoning built in and return structured tool calls, so no
"Thought:/Action:" text format has to be prompted and parsed any more.

Harness safeguards shown here:
- hard iteration limit (errors multiply over steps: keep loops short)
- tool errors are returned to the model as readable messages
- destructive tool (delete) needs confirmation from the user
- one JSONL trace per run in traces/todo_react_agent.jsonl

Uses:
- OpenRouter (via openai client) for LLM inference
- A simple JSON file for persistent TODO storage
"""

import os
import sys
import json

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tracing import AI_NOTICE, Timer, usage_dict, write_trace  # noqa: E402

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OPEN_ROUTER_API_KEY = os.environ.get("OPEN_ROUTER_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPEN_ROUTER_API_KEY,
)

INFERENCE_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"
MAX_ITERATIONS = 10

SYSTEM_PROMPT = (
    "You are a helpful TODO list assistant. "
    "Use the provided tools to manage the user's tasks; look up IDs with list_todos before changing a todo. "
    "Answer in a friendly, concise way."
)

# ---------------------------------------------------------------------------
# TODO storage (simple JSON file)
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
# Built-in tools (TODO CRUD)
# ---------------------------------------------------------------------------

def add_todo(description: str) -> str:
    """Add a new todo item and return confirmation."""
    todos = _load_todos()
    new_id = max((t["id"] for t in todos), default=0) + 1
    todo = {"id": new_id, "description": description.strip(), "done": False}
    todos.append(todo)
    _save_todos(todos)
    return f"Added todo #{new_id}: '{todo['description']}'"


def list_todos() -> str:
    """Return all todos as a formatted string."""
    todos = _load_todos()
    if not todos:
        return "The TODO list is empty."
    lines = []
    for t in todos:
        status = "✓" if t["done"] else "○"
        lines.append(f"  [{status}] #{t['id']}: {t['description']}")
    return "Current TODOs:\n" + "\n".join(lines)


def complete_todo(todo_id: int) -> str:
    """Mark a todo as done by its ID."""
    todos = _load_todos()
    for t in todos:
        if t["id"] == todo_id:
            t["done"] = True
            _save_todos(todos)
            return f"Marked todo #{todo_id} ('{t['description']}') as done."
    return f"Error: No todo with ID #{todo_id} found."


def delete_todo(todo_id: int) -> str:
    """Delete a todo by its ID."""
    todos = _load_todos()
    for i, t in enumerate(todos):
        if t["id"] == todo_id:
            removed = todos.pop(i)
            _save_todos(todos)
            return f"Deleted todo #{todo_id}: '{removed['description']}'."
    return f"Error: No todo with ID #{todo_id} found."


# ---------------------------------------------------------------------------
# Tool schemas (name + description + parameter schema; the description is part of the prompt)
# ---------------------------------------------------------------------------

def _tool(name, description, params=None, required=()):
    return {"type": "function", "function": {
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": params or {}, "required": list(required)},
    }}


TOOLS = [
    _tool("add_todo", "Add a new todo item.", {"description": {"type": "string"}}, ["description"]),
    _tool("list_todos", "List all todos with their IDs and done/open status."),
    _tool("complete_todo", "Mark a todo as done.", {"todo_id": {"type": "integer"}}, ["todo_id"]),
    _tool("delete_todo", "Delete a todo permanently (the user must confirm).", {"todo_id": {"type": "integer"}}, ["todo_id"]),
]

FUNCTIONS = {
    "add_todo": add_todo,
    "list_todos": list_todos,
    "complete_todo": complete_todo,
    "delete_todo": delete_todo,
}


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def execute_tool(name: str, args: dict) -> str:
    """Run one tool call. Errors become observations the model can react to."""
    if name not in FUNCTIONS:
        return f"Error: unknown tool '{name}'. Available tools: {list(FUNCTIONS)}"
    if name == "delete_todo":
        answer = input(f"  Confirm: delete todo #{args.get('todo_id')}? [y/N] ").strip().lower()
        if answer != "y":
            return "The user declined the deletion."
    try:
        return FUNCTIONS[name](**args)
    except Exception as e:
        return f"Error executing tool: {e}"


def run(task: str) -> str:
    """Agent loop: call the model until it answers without tool calls or the limit is hit."""
    timer = Timer()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    tool_calls_log, tokens = [], {"prompt_tokens": 0, "completion_tokens": 0}
    answer = "Stopped: maximum number of iterations reached without a final answer."

    for iteration in range(1, MAX_ITERATIONS + 1):
        response = client.chat.completions.create(
            model=INFERENCE_MODEL, messages=messages, tools=TOOLS,
        )
        for key, value in usage_dict(response).items():
            tokens[key] += value
        message = response.choices[0].message
        messages.append(message)

        if not message.tool_calls:
            answer = message.content or ""
            break

        for call in message.tool_calls:
            try:
                args = json.loads(call.function.arguments or "{}")
                result = execute_tool(call.function.name, args)
            except json.JSONDecodeError:
                args, result = call.function.arguments, "Error: arguments are not valid JSON."
            print(f"  [{iteration}] {call.function.name}({args}) -> {result}")
            tool_calls_log.append({"tool": call.function.name, "args": args, "result": result})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    write_trace("todo_react_agent", {
        "input": task, "model": INFERENCE_MODEL, "tool_calls": tool_calls_log,
        "output": answer, "usage": tokens, "latency_ms": timer.ms(),
    })
    return answer


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  TODO List Assistant  (LLM Agent Demo)")
    print("=" * 60)
    print(AI_NOTICE)

    print("\n--- System prompt ---")
    print(SYSTEM_PROMPT)
    print("--- End of system prompt ---")
    print(f"\nTools: {', '.join(t['function']['name'] for t in TOOLS)}\n")

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

        print(f"\nAssistant: {run(user_input)}\n")
