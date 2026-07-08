"""
Document QA Pipeline for Open-WebUI
====================================
Connects Open-WebUI → ChromaDB (RAG) → Ollama (gemma2:2b)

Context window strategy for gemma2:2b (8192 tokens):
  - Chunk size  : 512 tokens  (~1900 chars)
  - Chunk overlap: 64 tokens  (~240 chars)
  - Top-k chunks : 4
  - Total context used ≈ 3,500 tokens → leaves ~4,700 tokens for the answer

How to install this pipeline in Open-WebUI:
  1. Start Open-WebUI
  2. Go to Admin Panel → Settings → Pipelines
  3. Set pipeline directory to the folder containing this file
  4. Enable this pipeline for your workspace
"""

import hashlib
import io
import os
import re
from typing import Any, Generator, Iterator, List, Optional, Union

# ── Open-WebUI pipeline contract ────────────────────────────────────────────
from pydantic import BaseModel, Field

# ── PDF parsing ──────────────────────────────────────────────────────────────
try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader

# ── Text splitting ───────────────────────────────────────────────────────────
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Vector store ─────────────────────────────────────────────────────────────
import chromadb
from chromadb.config import Settings

# ── Embeddings via Ollama ────────────────────────────────────────────────────
import requests
import json

# ── HTTP client for Ollama chat ───────────────────────────────────────────────
import httpx

# ---------------------------------------------------------------------------
# Constants — tuned for gemma2:2b 8k context window
# ---------------------------------------------------------------------------
CHUNK_SIZE_CHARS = 1900          # ≈ 512 tokens at ~3.7 chars/token
CHUNK_OVERLAP_CHARS = 240        # ≈ 64 tokens overlap to preserve context
TOP_K = 4                        # Number of chunks to retrieve per query
EMBED_MODEL = "nomic-embed-text" # Ollama embedding model
CHAT_MODEL = "gemma2:2b"         # Ollama chat model
OLLAMA_BASE_URL = "http://localhost:11434"
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

# ---------------------------------------------------------------------------
# Embedding helper — calls Ollama's /api/embeddings endpoint
# ---------------------------------------------------------------------------

def get_embedding(text: str) -> List[float]:
    """Get a single embedding from Ollama nomic-embed-text."""
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts sequentially (Ollama doesn't support batch natively)."""
    return [get_embedding(t) for t in texts]


# ---------------------------------------------------------------------------
# ChromaDB helpers
# ---------------------------------------------------------------------------

def get_chroma_client() -> chromadb.PersistentClient:
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def collection_name_from_hash(file_hash: str) -> str:
    """ChromaDB collection names must be 3-63 chars, alphanumeric + hyphens."""
    return f"doc-{file_hash[:24]}"


def doc_already_indexed(client: chromadb.PersistentClient, collection_name: str) -> bool:
    try:
        col = client.get_collection(collection_name)
        return col.count() > 0
    except Exception:
        return False


def index_document(
    client: chromadb.PersistentClient,
    collection_name: str,
    chunks: List[str],
) -> chromadb.Collection:
    """Embed and store chunks in ChromaDB. Returns the collection."""
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    if collection.count() > 0:
        return collection  # already indexed

    print(f"[DocQA] Embedding {len(chunks)} chunks into '{collection_name}'…")
    embeddings = get_embeddings_batch(chunks)
    ids = [f"chunk-{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)
    print(f"[DocQA] Indexed {len(chunks)} chunks.")
    return collection


def retrieve_chunks(
    collection: chromadb.Collection,
    query: str,
    top_k: int = TOP_K,
) -> List[str]:
    """Retrieve the top-k most relevant chunks for a query."""
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
    )
    return results["documents"][0] if results["documents"] else []


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract all text from a PDF given its raw bytes."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        # Label pages so the LLM can cite them
        pages.append(f"[Page {i + 1}]\n{text.strip()}")
    return "\n\n".join(pages)


def chunk_text(text: str) -> List[str]:
    """Split document text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_CHARS,
        chunk_overlap=CHUNK_OVERLAP_CHARS,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_text(text)
    # Filter out very short/empty chunks
    return [c.strip() for c in chunks if len(c.strip()) > 50]


def md5_of_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a helpful document analysis assistant.
Answer the user's question using ONLY the provided document excerpts.
If the answer is not found in the excerpts, say "I could not find that information in the document."
Always be concise and accurate. Cite page numbers when available."""


def build_rag_prompt(chunks: List[str], question: str) -> str:
    context = "\n\n---\n\n".join(chunks)
    return f"""Document Excerpts:
{context}

---

Question: {question}

Answer:"""


# ---------------------------------------------------------------------------
# Open-WebUI Pipeline class
# ---------------------------------------------------------------------------

class Pipeline:
    """
    Open-WebUI compatible RAG pipeline.

    Valves (configurable via Admin UI):
      - ollama_base_url : Ollama server URL
      - embed_model     : Embedding model name in Ollama
      - chat_model      : Chat model name in Ollama
      - top_k           : Number of chunks to retrieve
      - chunk_size      : Chunk size in characters
      - chunk_overlap   : Overlap between chunks in characters
    """

    class Valves(BaseModel):
        ollama_base_url: str = Field(
            default=OLLAMA_BASE_URL,
            description="Ollama server base URL",
        )
        embed_model: str = Field(
            default=EMBED_MODEL,
            description="Ollama model used for embeddings",
        )
        chat_model: str = Field(
            default=CHAT_MODEL,
            description="Ollama model used for chat/QA",
        )
        top_k: int = Field(
            default=TOP_K,
            description="Number of document chunks to retrieve per query",
        )
        chunk_size: int = Field(
            default=CHUNK_SIZE_CHARS,
            description="Chunk size in characters (≈512 tokens for gemma2:2b)",
        )
        chunk_overlap: int = Field(
            default=CHUNK_OVERLAP_CHARS,
            description="Overlap between chunks in characters",
        )

    def __init__(self):
        self.name = "📄 Document QA (RAG)"
        self.valves = self.Valves()
        self._chroma_client: Optional[chromadb.PersistentClient] = None

    # -- Lifecycle -----------------------------------------------------------

    async def on_startup(self):
        print(f"[DocQA] Pipeline starting up. ChromaDB at: {CHROMA_PERSIST_DIR}")
        self._chroma_client = get_chroma_client()
        print("[DocQA] ChromaDB client ready.")

    async def on_shutdown(self):
        print("[DocQA] Pipeline shutting down.")

    # -- Main pipe method ----------------------------------------------------

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict,
    ) -> Union[str, Generator, Iterator]:
        """
        Called by Open-WebUI for every user message.

        Open-WebUI passes uploaded files in body["files"] as a list of dicts:
          { "name": "...", "type": "application/pdf", "data": "<base64>" }
        """
        import base64

        valves = self.valves

        # Ensure ChromaDB is initialised (in case on_startup wasn't awaited)
        if self._chroma_client is None:
            self._chroma_client = get_chroma_client()

        client = self._chroma_client

        # ── Step 1: Process any uploaded PDF files ───────────────────────────
        uploaded_files = body.get("files", [])
        active_collections: List[chromadb.Collection] = []

        for f in uploaded_files:
            mime = f.get("type", "")
            if "pdf" not in mime.lower():
                continue  # skip non-PDFs

            # Decode the file data
            raw_data = f.get("data", "")
            # Strip data-URI prefix if present: "data:application/pdf;base64,..."
            if "," in raw_data:
                raw_data = raw_data.split(",", 1)[1]
            pdf_bytes = base64.b64decode(raw_data)

            file_hash = md5_of_bytes(pdf_bytes)
            col_name = collection_name_from_hash(file_hash)

            if doc_already_indexed(client, col_name):
                print(f"[DocQA] '{f.get('name')}' already indexed as '{col_name}'.")
                col = client.get_collection(col_name)
            else:
                print(f"[DocQA] Processing PDF: {f.get('name')} ({len(pdf_bytes)//1024} KB)")
                text = extract_text_from_pdf(pdf_bytes)
                chunks = chunk_text(text)
                print(f"[DocQA] Split into {len(chunks)} chunks.")
                col = index_document(client, col_name, chunks)

            active_collections.append(col)

        # ── Step 2: Retrieve relevant chunks ─────────────────────────────────
        all_chunks: List[str] = []
        if active_collections:
            for col in active_collections:
                retrieved = retrieve_chunks(col, user_message, top_k=valves.top_k)
                all_chunks.extend(retrieved)

            # De-duplicate while preserving order
            seen = set()
            deduped = []
            for c in all_chunks:
                key = c[:80]
                if key not in seen:
                    seen.add(key)
                    deduped.append(c)
            all_chunks = deduped

        # ── Step 3: Build the prompt ─────────────────────────────────────────
        if all_chunks:
            rag_user_content = build_rag_prompt(all_chunks, user_message)
            # Replace the last user message with the RAG-enriched version
            ollama_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            # Include prior conversation (skip the last user turn — we'll replace it)
            for msg in messages[:-1]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Flatten content blocks (Open-WebUI sometimes sends lists)
                    content = " ".join(
                        part.get("text", "") for part in content if isinstance(part, dict)
                    )
                ollama_messages.append({"role": role, "content": content})
            ollama_messages.append({"role": "user", "content": rag_user_content})
        else:
            # No PDF uploaded or no matching chunks — pass through normally
            ollama_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        part.get("text", "") for part in content if isinstance(part, dict)
                    )
                ollama_messages.append({"role": role, "content": content})

        # ── Step 4: Stream response from Ollama ──────────────────────────────
        return self._stream_ollama(valves.ollama_base_url, valves.chat_model, ollama_messages)

    def _stream_ollama(
        self,
        base_url: str,
        model: str,
        messages: List[dict],
    ) -> Generator[str, None, None]:
        """Stream tokens from Ollama /api/chat endpoint."""
        url = f"{base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "num_ctx": 8192,    # Explicitly set context window
                "temperature": 0.3, # Lower temp for factual QA
                "top_p": 0.9,
            },
        }

        with httpx.Client(timeout=300) as http_client:
            with http_client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        token = data.get("message", {}).get("content", "")
                        if token:
                            yield token
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
