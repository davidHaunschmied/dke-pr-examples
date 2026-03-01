"""
RAG (Retrieval-Augmented Generation) system for answering questions about the
JKU Linz "Wirtschaftsinformatik" programs (Bachelor and Master).

Uses:
- PyPDF2 for PDF text extraction
- OpenRouter for both embeddings and LLM inference (via openai client)
- numpy for cosine similarity
- A simple JSON file as the vector store (no external DB needed)
"""

import os
import json
import hashlib
import urllib.request

import numpy as np
import PyPDF2
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
EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"

DB_DIR = os.path.join(os.path.dirname(__file__), "db")

DOCUMENTS = {
    "master": {
        "label": "Masterstudium Wirtschaftsinformatik",
        "pdf": os.path.join(DB_DIR, "win_master.pdf"),
        "embeddings": os.path.join(DB_DIR, "win_master_embeddings.json"),
    },
    "bachelor": {
        "label": "Bachelorstudium Wirtschaftsinformatik",
        "pdf": os.path.join(DB_DIR, "win_bachelor.pdf"),
        "embeddings": os.path.join(DB_DIR, "win_bachelor_embeddings.json"),
    },
}

CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 100    # overlap between consecutive chunks
TOP_K = 3              # number of similar chunks to retrieve
SIMILARITY_THRESHOLD = 0.10  # minimum cosine similarity for retrieval


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
    """Return the embedding vector for a single piece of text via OpenRouter."""
    # Use raw HTTP because the openai client can't parse OpenRouter's embedding response
    data = json.dumps({"model": EMBEDDING_MODEL, "input": text}).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/embeddings",
        data=data,
        headers={
            "Authorization": f"Bearer {OPEN_ROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read().decode())
    return body["data"][0]["embedding"]



def file_hash(path: str) -> str:
    """Return the SHA-256 hash of a file."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ---------------------------------------------------------------------------
# Build / load embeddings store
# ---------------------------------------------------------------------------

def build_or_load_embeddings(pdf_path: str, embeddings_path: str) -> list[dict]:
    """
    If the embeddings JSON already exists and was built from the same PDF,
    load and return it.  Otherwise extract -> chunk -> embed -> save.
    """
    pdf_hash = file_hash(pdf_path)

    if os.path.exists(embeddings_path):
        with open(embeddings_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("pdf_hash") == pdf_hash:
            print(f"Loaded {len(data['chunks'])} cached chunks from {embeddings_path}")
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

    # Embed
    print("  Generating embeddings (this may take a moment) ...")
    vectors = []
    for i, chunk in enumerate(chunks):
        print(f"    Embedding chunk {i + 1}/{len(chunks)} ...")
        vectors.append(get_embedding(chunk))

    chunk_records = [
        {"text": chunks[i], "embedding": vectors[i]}
        for i in range(len(chunks))
    ]

    # Save
    store = {"pdf_hash": pdf_hash, "chunks": chunk_records}
    with open(embeddings_path, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False)
    print(f"  Saved embeddings to {embeddings_path}\n")

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
        "You are an expert assistant for JKU Linz Wirtschaftsinformatik programs. "
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
    print("  Wirtschaftsinformatik @ JKU - RAG Q&A System")
    print("=" * 60)
    print()

    # Select program type
    program_type = ""
    while program_type not in DOCUMENTS:
        print("Please select the program type:")
        for key, doc in DOCUMENTS.items():
            print(f"  {key}: {doc['label']}")
        program_type = input("Your choice (master/bachelor): ").strip().lower()

    # Build or load the embedding index for the selected program
    doc = DOCUMENTS[program_type]
    print(f"\nSelected: {doc['label']}\n")
    chunks_db = build_or_load_embeddings(doc["pdf"], doc["embeddings"])

    print("\nReady! Ask questions about the Wirtschaftsinformatik program.")
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
