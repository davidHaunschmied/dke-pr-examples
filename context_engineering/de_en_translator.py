"""
Context Engineering example: German <-> English Translator.

Demonstrates how careful system-prompt design ("context engineering") can steer
an LLM to produce high-quality, natural translations — without any tools, RAG,
or external data.  This is the simplest of the three complexity levels in this
repository:  context_engineering  →  rag  →  agent.

Uses:
- OpenRouter (via openai client) for LLM inference
"""

import os

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
# System prompt — this is the core of context engineering
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert German ↔ English translator. Follow these rules strictly:

1. AUTO-DETECT the input language.
   - If the input is German, translate it into English.
   - If the input is English, translate it into German.

2. Translate NATURALLY and IDIOMATICALLY — never word-for-word.
   Produce text that reads as if a native speaker of the target language wrote it.

3. PRESERVE the speaker's tone and register.
   - Formal input (e.g. "Sie") → formal output.
   - Informal input (e.g. "du") → informal output.
   - Technical, poetic, humorous, or colloquial style must carry over.

4. Handle IDIOMS by finding an equivalent idiom in the target language
   rather than translating literally.
   Example: "Da steppt der Bär" → "That's where the party is"
            (not "There the bear dances")

5. When the input is AMBIGUOUS, choose the most common, natural translation.
   Do not list alternatives or ask for clarification.

6. Output ONLY the translated text. No explanations, no notes, no quotation
   marks, no "Translation:" prefix — just the translation itself.\
"""

# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

def translate(text: str) -> str:
    """Send the user text to the LLM with the translation system prompt."""
    response = client.chat.completions.create(
        model=INFERENCE_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Main interactive loop
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  German ↔ English Translator  (Context Engineering Demo)")
    print("=" * 60)

    print("\n--- System prompt (this is the context engineering) ---")
    print(SYSTEM_PROMPT)
    print("--- End of system prompt ---\n")

    print("Enter German text to get English, or English text to get German.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("Text: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        translation = translate(user_input)
        print(f"  → {translation}\n")

