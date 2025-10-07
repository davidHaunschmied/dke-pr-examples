import os
import json
import re
import logging
os.environ["GRPC_VERBOSITY"] = "ERROR"

import google.generativeai as genai

# Configure the API
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))


class SimpleTool:
    def __init__(self, name, func, description):
        self.name = name
        self.func = func
        self.description = description

    def execute(self, args):
        return self.func(args)


class SimpleAgent:
    def __init__(self, model_name="gemini-2.5-flash-lite"):
        self.model = None
        self.model_name = model_name
        self.tools = {}
        self.memory = []
        self.max_iterations = 10

    def initializeModel(self):
        self.model = genai.GenerativeModel(self.model_name, system_instruction=self.get_system_prompt())

    def add_tool(self, tool):
        self.tools[tool.name] = tool

    def get_system_prompt(self):
        tool_descriptions = ""
        for name, tool in self.tools.items():
            tool_descriptions += f"- {name}: {tool.description}\n"

        prompt = f"""You are an autonomous agent that solves tasks using a Thought → Action → Observation loop, step by step.

You have access to these tools:
{tool_descriptions}

For each step, you MUST strictly follow this format:
Thought: [your reasoning for this step]
Action: [tool_name(arguments)]

MANDATORY:
- Every response MUST begin with a Thought: line, followed by an Action: line. Never output an Action without a preceding Thought.
- If you ever omit the Thought line, you will be penalized.
- Do NOT output the final answer until you have received all necessary Observations and completed all required steps.
- After each Action, WAIT for the Observation before continuing.
- When you are ready to give the final answer, use:
Thought: [your reasoning for the final answer, and explain any tool results so the user understands them. The user cannot see the raw tool results.]
Action: final_answer(result)

EXAMPLES:
Step:
Thought: I need to calculate 2 + 2 to answer the question.
Action: calculator(2 + 2)

Final step:
Thought: The calculator returned 4, which means the answer to 2 + 2 is 4. I am explaining this result so the user understands how I arrived at the answer.
Action: final_answer(4)

Never skip steps. Never omit the Thought: line. Always use this exact format and process, step by step."""
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
        self.initializeModel()
        self.memory = []

        for iteration in range(self.max_iterations):
            print(f"\n--- Iteration {iteration + 1} ---")
            print(f"Current memory: {self.memory}\n")

            full_prompt = f"Task: {task}\n\nMemory:{self.memory}"


            # Get LLM response
            response = self.model.generate_content(full_prompt)
            response_text = response.text
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


# Example tools
def calculator(args):
    # Simple eval-based calculator (use with caution in production)
    try:
        result = eval(args)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


def web_search(args):
    return f"Mock search results for: {args}"


# Main execution
if __name__ == "__main__":
    # Create agent
    agent = SimpleAgent()

    # Add tools
    calc_tool = SimpleTool(
        name="calculator",
        func=calculator,
        description="Performs mathematical calculations. Input should be a valid Python expression like '2+2' or '15*7'"
    )
    agent.add_tool(calc_tool)

    search_tool = SimpleTool(
        name="web_search",
        func=web_search,
        description="Searches the web for information. Input should be a search query string"
    )
    agent.add_tool(search_tool)

    print(f"\n-------------------------- System prompt -----------------------\n"
          f"{agent.get_system_prompt()}"
          f"\n----------------------------------------------------------------\n")

    # Prompt for task
    task = input("Enter your task: ")
    print(f"\nTask: {task}\n")
    result = agent.run(task)
    print(f"\n\nFinal Result: {result}")