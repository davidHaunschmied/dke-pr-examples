"""
RAG (Retrieval-Augmented Generation) system for answering questions about the
JKU Linz "Wirtschaftsinformatik" master's program.

Uses:
- PyPDF2 for PDF text extraction
- sentence-transformers (all-MiniLM-L6-v2) for local embeddings
- OpenRouter (via openai client) for LLM inference
- numpy for cosine similarity
- A simple JSON file as the vector store (no external DB needed)
"""

import os
import json
import hashlib

import numpy as np
import PyPDF2
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OPEN_ROUTER_API_KEY = os.environ.get("OPEN_ROUTER_API_KEY", "YOUR_API_KEY_HERE")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPEN_ROUTER_API_KEY,
)

INFERENCE_MODEL = "google/gemini-2.0-flash-exp:free"

PDF_PATH = os.path.join(os.path.dirname(__file__), "wirtschaftsinformatik_master.pdf")
EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), "embeddings.json")

CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 100    # overlap between consecutive chunks
TOP_K = 3              # number of similar chunks to retrieve
SIMILARITY_THRESHOLD = 0.25  # minimum cosine similarity for retrieval

# ---------------------------------------------------------------------------
# Embedding model (local, no API key needed)
# ---------------------------------------------------------------------------

print("Loading embedding model (first run downloads ~80 MB) ...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model ready.\n")


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file using PyPDF2."""
    reader = PyPDF2.PdfReader(pdf_path)
    pages_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)
    return "\n".join(pages_text)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks of roughly `chunk_size` characters."""
    # Clean up excessive whitespace
    text = " ".join(text.split())

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def get_embedding(text: str) -> list[float]:
    """Return the embedding vector for a single piece of text."""
    vec = embedding_model.encode(text)
    return vec.tolist()


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Return embedding vectors for a batch of texts (much faster)."""
    vecs = embedding_model.encode(texts, show_progress_bar=True)
    return vecs.tolist()


def file_hash(path: str) -> str:
    """Return the SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Build / load embeddings store
# ---------------------------------------------------------------------------

def build_or_load_embeddings(pdf_path: str) -> list[dict]:
    """
    If embeddings.json already exists and was built from the same PDF,
    load and return it.  Otherwise extract -> chunk -> embed -> save.
    """
    pdf_hash = file_hash(pdf_path)

    if os.path.exists(EMBEDDINGS_PATH):
        with open(EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("pdf_hash") == pdf_hash:
            print(f"Loaded {len(data['chunks'])} cached chunks from {EMBEDDINGS_PATH}")
            return data["chunks"]
        else:
            print("PDF changed - rebuilding embeddings ...")

    # Extract
    print(f"Extracting text from {pdf_path} ...")
    text = extract_text_from_pdf(pdf_path)
    print(f"  Extracted {len(text)} characters from PDF.")

    # Chunk
    chunks = chunk_text(text)
    print(f"  Created {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")

    # Embed (batch for speed)
    print("  Generating embeddings (this may take a moment) ...")
    vectors = get_embeddings_batch([c for c in chunks])

    chunk_records = [
        {"text": chunks[i], "embedding": vectors[i]}
        for i in range(len(chunks))
    ]

    # Save
    store = {"pdf_hash": pdf_hash, "chunks": chunk_records}
    with open(EMBEDDINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False)
    print(f"  Saved embeddings to {EMBEDDINGS_PATH}\n")

    return chunk_records


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def cosine_similarity(a, b) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def retrieve_similar_texts(
    query: str,
    chunks: list[dict],
    top_k: int = TOP_K,
    threshold: float = SIMILARITY_THRESHOLD,
) -> list[str]:
    """
    Embed the user query, compute cosine similarity against all stored
    chunk embeddings, and return the top-k chunk texts above the threshold.
    """
    query_vec = get_embedding(query)

    scored = []
    for chunk in chunks:
        sim = cosine_similarity(query_vec, chunk["embedding"])
        if sim >= threshold:
            scored.append((sim, chunk["text"]))

    # Sort by similarity descending
    scored.sort(key=lambda x: x[0], reverse=True)

    top = scored[:top_k]
    if top:
        print(f"  Retrieved {len(top)} relevant chunks (similarities: {[round(s, 3) for s, _ in top]})")
    else:
        print("  No chunks above the similarity threshold found.")

    return [text for _, text in top]


# ---------------------------------------------------------------------------
# RAG query
# ---------------------------------------------------------------------------

def rag_query(question: str, chunks: list[dict]) -> str:
    """Retrieve relevant context and generate an answer via OpenRouter."""

    # Step 1: Retrieve
    print("\nRetrieving relevant passages ...")
    contexts = retrieve_similar_texts(question, chunks)

    if not contexts:
        context_block = "(No relevant passages found in the document.)"
    else:
        context_block = "\n\n---\n\n".join(contexts)

    # Step 2: Generate
    print("Generating answer ...\n")
    system_prompt = (
        "You are an expert assistant for the JKU Linz Wirtschaftsinformatik master's program. "
        "Answer the user's question based ONLY on the provided context passages from the official curriculum document. "
        "If the context does not contain enough information, say so. "
        "Always answer in the same language the user uses."
    )

    user_prompt = f"""Context from the curriculum document:

{context_block}

Question: {question}"""

    response = client.chat.completions.create(
        model=INFERENCE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Main interactive loop
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Wirtschaftsinformatik Master @ JKU - RAG Q&A System")
    print("=" * 60)
    print()

    # Build or load the embedding index
    chunks_db = build_or_load_embeddings(PDF_PATH)

    print("\nReady! Ask questions about the Wirtschaftsinformatik master's program.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            question = input("Your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        answer = rag_query(question, chunks_db)
        print(f"\n=== Answer ===\n{answer}\n")
