"""
LLM Agent: TODO List Assistant using a Thought → Action → Observation loop (ReAct).

Demonstrates autonomous tool use: the agent reasons about the user's natural-
language request, picks the right CRUD tool, observes the result, and responds.
This is the most complex of the three levels:  context_engineering → rag → agent.

Uses:
- OpenRouter (via openai client) for LLM inference
- A simple JSON file for persistent TODO storage
"""

import os
import re
import json

from dotenv import load_dotenv
from openai import OpenAI

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


# ---------------------------------------------------------------------------
# Tool abstraction
# ---------------------------------------------------------------------------

class SimpleTool:
    def __init__(self, name, func, description):
        self.name = name
        self.func = func
        self.description = description

    def execute(self, args):
        return self.func(args)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class SimpleAgent:
    def __init__(self):
        self.tools = {}
        self.memory = []
        self.max_iterations = 10
        self.system_prompt = None


    def add_tool(self, tool):
        self.tools[tool.name] = tool

    def get_system_prompt(self):
        tool_descriptions = ""
        for name, tool in self.tools.items():
            tool_descriptions += f"- {name}: {tool.description}\n"

        prompt = f"""You are a helpful TODO list assistant. You manage the user's tasks using a Thought → Action → Observation loop.

You have access to these tools:
{tool_descriptions}

For each step, you MUST strictly follow this format:
Thought: [your reasoning for this step]
Action: [tool_name(arguments)]

Do NOT include the Observation in your response; I will provide it after you take an Action.

MANDATORY:
- Every response MUST begin with a Thought: line, followed by an Action: line.
- Only output ONE Thought and ONE Action per step.
- After each Action, WAIT for the Observation before continuing.
- When you have completed the user's request and want to respond, use:
  Thought: [summarize what you did in a friendly way]
  Action: final_answer(your response to the user)

EXAMPLES:

User says: "Add buy groceries to my list"
Thought: The user wants to add a new todo item called "buy groceries".
Action: add_todo(buy groceries)

User says: "What's on my list?"
Thought: The user wants to see all their todos. I should list them.
Action: list_todos()

User says: "I finished the first task"
Thought: The user completed a task. I should first list the todos to find which one is first, then mark it done.
Action: list_todos()

After seeing the list:
Thought: The first todo is "buy groceries" with ID 1. I'll mark it as complete.
Action: complete_todo(1)

User says: "Remove the groceries task"
Thought: I need to find and delete the todo about groceries. Let me list them first.
Action: list_todos()

Never skip steps. Always use this exact format."""
        return prompt

    def parse_action(self, text):
        # Look for the FIRST Action: tool_name(args) that doesn't have an Observation yet
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('Action:'):
                match = re.search(r'Action:\s*([a-zA-Z_]+)\((.*?)\)', line)
                if match:
                    tool_name = match.group(1)
                    args_str = match.group(2)
                    return tool_name, args_str
        return None, None

    def parse_thought(self, text):
        # Look for the FIRST Thought:
        lines = text.split('\n')
        for line in lines:
            if line.strip().startswith('Thought:'):
                # Extract everything after 'Thought:'
                return line.strip()[len('Thought:'):].strip()
        return "None"

    def run(self, task):
        self.system_prompt = self.get_system_prompt()
        self.memory = []

        for iteration in range(self.max_iterations):
            print(f"\n--- Iteration {iteration + 1} ---")
            print(f"Current memory: {self.memory}\n")

            user_prompt = f"Task: {task}\n\nMemory:{self.memory}"

            # Get LLM response
            response = client.chat.completions.create(
                model=INFERENCE_MODEL,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            response_text = response.choices[0].message.content
            print(f"LLM Output:\n{response_text}")

            # Parse the thought
            thought = self.parse_thought(response_text)

            # Parse the action
            tool_name, args_str = self.parse_action(response_text)

            # Check if this is the final answer
            if tool_name == "final_answer":
                print(f"\n=== Final Answer ===")
                print(args_str)
                return args_str

            # Execute the tool
            if tool_name in self.tools:
                tool = self.tools[tool_name]
                try:
                    result = tool.execute(args_str)
                    observation = f"{result}"

                except Exception as e:
                    observation = f"Error executing tool: {str(e)}"
            else:
                observation = f"Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}"

            print(f"\nObservation: {observation}")

            # Add observation to memory
            self.memory.append({
                "Thought": thought,
                "Action": f"{tool_name}({args_str})",
                "Observation": observation
            })

        return "Max iterations reached without finding answer"


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


def list_todos(_args: str = "") -> str:
    """Return all todos as a formatted string."""
    todos = _load_todos()
    if not todos:
        return "The TODO list is empty."
    lines = []
    for t in todos:
        status = "✓" if t["done"] else "○"
        lines.append(f"  [{status}] #{t['id']}: {t['description']}")
    return "Current TODOs:\n" + "\n".join(lines)


def complete_todo(args: str) -> str:
    """Mark a todo as done by its ID."""
    try:
        todo_id = int(args.strip())
    except ValueError:
        return f"Error: '{args}' is not a valid ID. Please provide a number."
    todos = _load_todos()
    for t in todos:
        if t["id"] == todo_id:
            t["done"] = True
            _save_todos(todos)
            return f"Marked todo #{todo_id} ('{t['description']}') as done."
    return f"Error: No todo with ID #{todo_id} found."


def delete_todo(args: str) -> str:
    """Delete a todo by its ID."""
    try:
        todo_id = int(args.strip())
    except ValueError:
        return f"Error: '{args}' is not a valid ID. Please provide a number."
    todos = _load_todos()
    for i, t in enumerate(todos):
        if t["id"] == todo_id:
            removed = todos.pop(i)
            _save_todos(todos)
            return f"Deleted todo #{todo_id}: '{removed['description']}'."
    return f"Error: No todo with ID #{todo_id} found."


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Create agent with TODO tools
    agent = SimpleAgent()

    agent.add_tool(SimpleTool(
        name="add_todo",
        func=add_todo,
        description="Adds a new todo item. Input: the description text, e.g. add_todo(Buy groceries)"
    ))
    agent.add_tool(SimpleTool(
        name="list_todos",
        func=list_todos,
        description="Lists all current todos with their IDs and status. No input needed: list_todos()"
    ))
    agent.add_tool(SimpleTool(
        name="complete_todo",
        func=complete_todo,
        description="Marks a todo as done. Input: the todo ID number, e.g. complete_todo(1)"
    ))
    agent.add_tool(SimpleTool(
        name="delete_todo",
        func=delete_todo,
        description="Deletes a todo permanently. Input: the todo ID number, e.g. delete_todo(1)"
    ))

    print("=" * 60)
    print("  TODO List Assistant  (LLM Agent Demo)")
    print("=" * 60)

    print("\n--- System prompt ---")
    print(agent.get_system_prompt())
    print("--- End of system prompt ---\n")

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

        result = agent.run(user_input)
        print(f"\nAssistant: {result}\n")
