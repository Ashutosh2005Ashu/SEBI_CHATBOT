"""
Document QA — Open-WebUI Pipe Function  (v2)
=============================================
Upload this file via Admin Panel -> Functions in Open-WebUI.

How it works (NO tool-calling required):
  1. User uploads a PDF in the chat
  2. Open-WebUI stores it and creates a ChromaDB collection automatically
  3. This function reads that collection name from body["files"]
  4. Calls Open-WebUI's own /api/v1/retrieval/query/collection endpoint
     to retrieve top-k relevant chunks for the user's question
  5. Injects the chunks directly into the Ollama prompt — no tool calls needed
  6. Streams the response from Ollama back to the user

Context window budget for gemma2:2b (8192 tokens):
  Retrieved chunks : top-k=4  -> ~2048 tokens
  System prompt    :           ->  ~200 tokens
  Conversation hist:           -> ~1000 tokens
  User question    :           ->  ~200 tokens
  Answer budget    :           -> ~4750 tokens remaining
"""

import json
import os
import re
import time
from datetime import datetime
from typing import Generator, Iterator, List, Optional, Tuple, Union

import httpx
import requests
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_OLLAMA_URL   = "http://localhost:11434"
_WEBUI_URL    = "http://localhost:8080"
_CHAT_MODEL   = "gemma2:2b"
_TOP_K        = 4
_ENV_FILE     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".webui_admin.env")

_SYSTEM = (
    "You are an expert SEBI compliance assistant. "
    "Provide a detailed, comprehensive explanation to the user's question using ONLY the provided document excerpts. "
    "If the information is not found in the excerpts, clearly state: "
    "'I could not find that information in the uploaded documents.' "
    "Always cite the circular number, clause number, and page numbers when available in the text. "
    "IMPORTANT: When multiple excerpts discuss the same topic with different or conflicting rules, "
    "you MUST follow and cite the MOST RECENTLY DATED circular. "
    "Each excerpt is prefixed with its detected date in [Source date: ...] format — use this to determine recency. "
    "Do not stop in the middle of a thought; provide complete, well-rounded answers."
)


# ---------------------------------------------------------------------------
# Auth helper — reads saved admin creds to call Open-WebUI's internal API
# ---------------------------------------------------------------------------

def _load_admin_token() -> Optional[str]:
    """Read credentials from .webui_admin.env and get a fresh token."""
    if not os.path.exists(_ENV_FILE):
        return None
    creds = {}
    with open(_ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                creds[k.strip()] = v.strip()
    email    = creds.get("EMAIL")
    password = creds.get("PASSWORD")
    if not (email and password):
        return None
    try:
        r = requests.post(
            f"{_WEBUI_URL}/api/v1/auths/signin",
            json={"email": email, "password": password},
            timeout=10,
        )
        if r.status_code == 200:
            return r.json().get("token")
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Retrieval via Open-WebUI's internal API
# ---------------------------------------------------------------------------

def _query_webui_collections(
    collection_names: List[str],
    query: str,
    top_k: int,
    token: str,
    webui_url: str,
) -> List[str]:
    """
    Query Open-WebUI's own vector store for relevant chunks.
    This uses the same collection that Open-WebUI created when the user uploaded the file.
    """
    if not collection_names or not token:
        return []

    payload = {
        "collection_names": collection_names,
        "query": query,
        "k": top_k,
    }
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    # Try the retrieval endpoint
    endpoints = [
        f"{webui_url}/api/v1/retrieval/query/collection",
        f"{webui_url}/api/v1/rag/query/collection",
        f"{webui_url}/rag/api/v1/query/collection",
    ]
    for url in endpoints:
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=30)
            if r.status_code == 200:
                data = r.json()
                # Response shape: {"results": [{"documents": [[...]], ...}]}
                # OR: {"documents": [[...]], ...}
                chunks = []
                results = data.get("results", [data])
                for result in results:
                    docs = result.get("documents", [])
                    for doc_list in docs:
                        if isinstance(doc_list, list):
                            chunks.extend(d for d in doc_list if d)
                        elif doc_list:
                            chunks.append(doc_list)
                if chunks:
                    print(f"[DocQA] Retrieved {len(chunks)} chunks from {url}")
                    return chunks
        except Exception as e:
            print(f"[DocQA] Endpoint {url} failed: {e}")
            continue

    return []


def _extract_collection_names(body: dict) -> List[str]:
    """
    Extract ChromaDB collection names from Open-WebUI's body structure.
    This recursively searches the body for any 'collection_name' or 'collection_names' keys.
    """
    collections = []

    def _search(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "collection_name" and isinstance(v, str):
                    collections.append(v)
                elif k == "collection_names" and isinstance(v, list):
                    collections.extend([x for x in v if isinstance(x, str)])
                else:
                    _search(v)
        elif isinstance(obj, list):
            for item in obj:
                _search(item)

    _search(body)

    # De-duplicate while preserving order
    seen = set()
    unique = []
    for c in collections:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    if unique:
        print(f"[DocQA] Found collections: {unique}")
    else:
        # Debug: dump body structure keys so we can diagnose missing files
        print(f"[DocQA] WARNING: No collections found. body keys: {list(body.keys())}")
        files = body.get("files", [])
        print(f"[DocQA] metadata: {json.dumps(body.get('metadata', {}), default=str)}")

    return unique


def _get_user_message(body: dict) -> str:
    """Extract the latest user message text from the messages list."""
    for msg in reversed(body.get("messages", [])):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                return " ".join(
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            return str(content)
    return ""


# ---------------------------------------------------------------------------
# Date extraction & re-ranking helpers
# ---------------------------------------------------------------------------

# Matches formats found in SEBI circular headers, e.g.:
#   "March 22, 2021"  |  "22nd March, 2021"  |  "22/03/2021"  |  "2021-03-22"
_DATE_PATTERNS: List[Tuple[str, str]] = [
    # "March 22, 2021" or "March 22 2021"
    (r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})[,\s]\s*(\d{4})\b",
     "%B %d %Y"),
    # "22nd March, 2021" or "22 March 2021"
    (r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(January|February|March|April|May|June|July|August|September|October|November|December)[,\s]\s*(\d{4})\b",
     "%d %B %Y"),
    # "22/03/2021" or "22-03-2021"
    (r"\b(\d{2})[/\-](\d{2})[/\-](\d{4})\b",
     "%d/%m/%Y"),
    # ISO: "2021-03-22"
    (r"\b(\d{4})-(\d{2})-(\d{2})\b",
     "ISO"),
]


def _extract_circular_date(text: str) -> Tuple[datetime, str]:
    """
    Scan chunk text for a SEBI circular date.
    Returns (datetime_object, display_string).
    Falls back to (datetime.min, "unknown") if nothing is found.
    """
    for pattern, fmt in _DATE_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if not m:
            continue
        try:
            if fmt == "%B %d %Y":
                # groups: month_name, day, year
                date_str = f"{m.group(1)} {m.group(2)} {m.group(3)}"
                return datetime.strptime(date_str, "%B %d %Y"), date_str
            elif fmt == "%d %B %Y":
                # groups: day, month_name, year
                date_str = f"{m.group(1)} {m.group(2)} {m.group(3)}"
                return datetime.strptime(date_str, "%d %B %Y"), date_str
            elif fmt == "%d/%m/%Y":
                date_str = f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
                return datetime.strptime(date_str, "%d/%m/%Y"), date_str
            elif fmt == "ISO":
                date_str = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
                return datetime.strptime(date_str, "%Y-%m-%d"), date_str
        except ValueError:
            continue
    return datetime.min, "unknown"


def _rerank_by_date(chunks: List[str]) -> List[str]:
    """
    Sort retrieved chunks newest-first and prepend a [Source date: ...] label
    so the LLM can reason about recency.
    """
    if not chunks:
        return chunks

    dated: List[Tuple[datetime, str, str]] = []
    for chunk in chunks:
        dt, display = _extract_circular_date(chunk)
        dated.append((dt, display, chunk))

    # Sort descending: newest date first
    dated.sort(key=lambda x: x[0], reverse=True)

    reranked = []
    for dt, display, chunk in dated:
        label = f"[Source date: {display}]\n"
        reranked.append(label + chunk)

    print(
        f"[DocQA] Re-ranked {len(reranked)} chunks by date. "
        f"Newest: {dated[0][1]}, Oldest: {dated[-1][1]}"
    )
    return reranked


def _build_prompt(chunks: List[str], question: str) -> str:
    context = "\n\n---\n\n".join(chunks)
    return (
        f"Document Excerpts (sorted newest circular first):\n"
        f"{context}\n\n"
        f"---\n\n"
        f"REMINDER: If excerpts conflict, follow the MOST RECENT circular (highest [Source date]).\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


# ---------------------------------------------------------------------------
# Pipe Function class
# ---------------------------------------------------------------------------

class Pipe:
    """
    Document QA RAG Pipe (v2)
    - Queries Open-WebUI's built-in vector store for uploaded file chunks
    - No tool-calling, no separate ChromaDB — works with any Ollama model
    """

    class Valves(BaseModel):
        ollama_base_url: str = Field(default=_OLLAMA_URL, description="Ollama server URL")
        webui_base_url:  str = Field(default=_WEBUI_URL,  description="Open-WebUI base URL (for internal retrieval)")
        chat_model:      str = Field(default=_CHAT_MODEL,  description="Ollama chat model name")
        top_k:           int = Field(default=_TOP_K,       description="Number of chunks to retrieve")

    def __init__(self):
        self.type    = "pipe"
        self.id      = "document_qa_rag"
        self.name    = "Document QA (RAG)"
        self.valves  = self.Valves()
        self._token: Optional[str]  = None
        self._token_ts: float       = 0.0

    def _get_token(self) -> Optional[str]:
        """Get (and cache for 55 min) an admin auth token."""
        now = time.time()
        if self._token and (now - self._token_ts) < 3300:  # 55 min
            return self._token
        tok = _load_admin_token()
        if tok:
            self._token    = tok
            self._token_ts = now
        return self._token

    # ------------------------------------------------------------------

    def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
    ) -> Union[str, Generator, Iterator]:
        v   = self.valves
        tok = self._get_token()

        # ── User question ─────────────────────────────────────────────
        user_msg = _get_user_message(body)
        if not user_msg:
            user_msg = body.get("prompt", "")

        # ── Debug body ────────────────────────────────────────────
        try:
            with open(os.path.join(os.path.dirname(__file__), "debug_body.json"), "w") as f:
                json.dump(body, f, indent=2, default=str)
        except Exception as e:
            print(f"[DocQA] Could not dump body: {e}")

        # ── Find uploaded file collections ────────────────────────────
        collection_names = _extract_collection_names(body)

        # ── Retrieve chunks ───────────────────────────────────────────
        chunks: List[str] = []
        if collection_names:
            chunks = _query_webui_collections(
                collection_names, user_msg, v.top_k, tok, v.webui_base_url
            )
            # Re-rank by circular date so newest information is presented first
            if chunks:
                chunks = _rerank_by_date(chunks)

        # ── Build Ollama messages ─────────────────────────────────────
        ollama_msgs = [{"role": "system", "content": _SYSTEM}]

        # Include prior conversation turns (strip file blocks from content)
        messages = body.get("messages", [])
        for msg in messages[:-1]:
            role    = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            if str(content).strip():
                ollama_msgs.append({"role": role, "content": str(content)})

        # Final user turn — enriched with RAG context if we have chunks
        if chunks:
            ollama_msgs.append({"role": "user", "content": _build_prompt(chunks, user_msg)})
            print(f"[DocQA] Sending {len(chunks)} chunks to model. Question: {user_msg[:80]}")
        else:
            ollama_msgs.append({"role": "user", "content": user_msg})
            print(f"[DocQA] No chunks found — answering without document context.")

        # ── Stream from Ollama ────────────────────────────────────────
        return self._stream(v.ollama_base_url, v.chat_model, ollama_msgs)

    def _stream(self, base_url: str, model: str, messages: List[dict]) -> Generator:
        payload = {
            "model":    model,
            "messages": messages,
            "stream":   True,
            "options":  {
                "num_ctx":     8192,
                "temperature": 0.3,
                "top_p":       0.9,
            },
        }
        with httpx.Client(timeout=300) as c:
            with c.stream("POST", f"{base_url}/api/chat", json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        data  = json.loads(line)
                        token = data.get("message", {}).get("content", "")
                        if token:
                            yield token
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
