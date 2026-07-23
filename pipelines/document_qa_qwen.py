"""
Document QA — Open-WebUI Pipe Function  (Qwen3-8B edition)
===========================================================
Upload this file via Admin Panel -> Functions in Open-WebUI.

Same RAG logic as document_qa_function.py, but powered by qwen3-8b:latest
which has a much larger context window (32 768 tokens) allowing more chunks
and longer answers than gemma2:2b.

How it works (NO tool-calling required):
  1. User uploads a PDF in the chat
  2. Open-WebUI stores it and creates a ChromaDB collection automatically
  3. This function reads that collection name from body["files"]
  4. Calls Open-WebUI's own /api/v1/retrieval/query/collection endpoint
     to retrieve top-k relevant chunks for the user's question
  5. Injects the chunks directly into the Ollama prompt — no tool calls needed
  6. Streams the response from Ollama back to the user

Context window budget for qwen3-8b (32 768 tokens):
  Retrieved chunks : top-k=8  -> ~8192 tokens
  System prompt    :           ->  ~300 tokens
  Conversation hist:           -> ~2000 tokens
  User question    :           ->  ~300 tokens
  Answer budget    :           -> ~21 976 tokens remaining
"""

import json
import os
import re
import sqlite3
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
_CHAT_MODEL   = "qwen3-8b:latest"
_TOP_K        = 8          # qwen3-8b handles more chunks comfortably
_NUM_CTX      = 32768      # qwen3-8b full context window
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

# Each retrieved chunk is returned as (text, metadata_dict)
_Chunk = Tuple[str, dict]


def _query_webui_collections(
    collection_names: List[str],
    query: str,
    top_k: int,
    token: str,
    webui_url: str,
    extra_queries: Optional[List[str]] = None,
) -> List["_Chunk"]:
    """
    Query Open-WebUI's own vector store for relevant chunks.
    Returns (document_text, metadata_dict) tuples.

    Runs the main semantic query, then any extra_queries (keyword date queries)
    and merges results, deduplicating by text content.
    """
    if not collection_names or not token:
        return []

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    endpoints = [
        f"{webui_url}/api/v1/retrieval/query/collection",
        f"{webui_url}/api/v1/rag/query/collection",
        f"{webui_url}/rag/api/v1/query/collection",
    ]

    def _run_single_query(q: str, k: int) -> List["_Chunk"]:
        """Run one query against the first working endpoint and return chunks."""
        payload = {"collection_names": collection_names, "query": q, "k": k}
        for url in endpoints:
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=30)
                if r.status_code == 200:
                    data    = r.json()
                    results = data.get("results", [data])
                    chunks: List[_Chunk] = []
                    for result in results:
                        docs  = result.get("documents", [])
                        metas = result.get("metadatas", [])
                        for i, doc_list in enumerate(docs):
                            meta_list = metas[i] if i < len(metas) else []
                            if isinstance(doc_list, list):
                                for j, d in enumerate(doc_list):
                                    if d:
                                        meta = meta_list[j] if j < len(meta_list) else {}
                                        chunks.append((d, meta if isinstance(meta, dict) else {}))
                            elif doc_list:
                                chunks.append((doc_list, {}))
                    if chunks:
                        return chunks
            except Exception as e:
                print(f"[DocQA-Qwen] Endpoint {url} failed: {e}")
                continue
        return []

    # ── Main semantic query ───────────────────────────────────────────────────
    all_chunks = _run_single_query(query, top_k)
    print(f"[DocQA-Qwen] Semantic query returned {len(all_chunks)} chunks")

    # ── Extra keyword queries (date-targeted) ─────────────────────────────────
    if extra_queries:
        seen_texts = {text for text, _ in all_chunks}
        for eq in extra_queries:
            extra = _run_single_query(eq, 4)   # top-4 per keyword query
            added = 0
            for text, meta in extra:
                if text not in seen_texts:
                    all_chunks.append((text, meta))
                    seen_texts.add(text)
                    added += 1
            if added:
                print(f"[DocQA-Qwen] Keyword query '{eq}' added {added} new chunks")

    return all_chunks


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
        print(f"[DocQA-Qwen] Found collections: {unique}")
    else:
        print(f"[DocQA-Qwen] WARNING: No collections found. body keys: {list(body.keys())}")
        print(f"[DocQA-Qwen] metadata: {json.dumps(body.get('metadata', {}), default=str)}")

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
# Query date detection & SEBI-DOCS KB fallback
# ---------------------------------------------------------------------------

def _extract_query_date(query: str) -> Optional[Tuple[str, str]]:
    """
    Detect a month+year date mention in the user's question.
    Returns (month_name, year_str) or None.
      "march 2021"       -> ("March", "2021")
      "March 22, 2021"   -> ("March", "2021")
      "03/2021"          -> ("March", "2021")
    """
    # Named month + optional day + year
    m = re.search(
        r"\b(January|February|March|April|May|June|July|August"
        r"|September|October|November|December)"
        r"(?:[,\s]+\d{1,2}[,\s]+|\s+)(\d{4})\b",
        query, re.IGNORECASE,
    )
    if m:
        return m.group(1).capitalize(), m.group(2)
    # Numeric  MM/YYYY or MM-YYYY
    m2 = re.search(r"\b(0?[1-9]|1[0-2])[/\-](\d{4})\b", query)
    if m2:
        months = ["January","February","March","April","May","June",
                  "July","August","September","October","November","December"]
        return months[int(m2.group(1)) - 1], m2.group(2)
    return None


def _build_date_queries(date_hint: Tuple[str, str]) -> List[str]:
    """
    Build targeted keyword queries from a (month_name, year) hint.
    These supplement the main semantic query so that circular header
    chunks containing only the date are also retrieved.
    """
    month, year = date_hint
    return [
        f"{month} {year}",                   # "March 2021"
        f"CIR/P/{year}",                      # SEBI circular number pattern
        f"SEBI circular {month} {year}",      # "SEBI circular March 2021"
    ]


# Known SEBI-DOCS knowledge base UUID — used as last-resort fallback if DB lookup fails.
# Update this if you recreate the knowledge base in Open-WebUI.
_SEBI_DOCS_KB_ID = "159a8be8-bfe9-4354-9747-83cdb4f6817d"


def _get_db_path() -> str:
    """
    Locate webui.db reliably when running inside Open-WebUI's function context.
    Open-WebUI executes pipe functions dynamically, so __file__ points to a
    temporary/virtual path. We resolve the real path using (in priority order):
      1. DATA_DIR env variable  (set by start.bat: set DATA_DIR=%~dp0data)
      2. __file__ relative path (works when running update_func.py directly)
    """
    data_dir = os.environ.get("DATA_DIR", "")
    if data_dir:
        candidate = os.path.join(data_dir, "webui.db")
        if os.path.exists(candidate):
            return candidate
    base = os.path.dirname(os.path.abspath(__file__))
    for rel in (os.path.join(base, "..", "data"), os.path.join(base, "data")):
        candidate = os.path.join(rel, "webui.db")
        if os.path.exists(candidate):
            return candidate
    return ""


def _get_sebi_kb_collection() -> List[str]:
    """
    Return the ChromaDB collection name for SEBI-DOCS.
    Resolution order:
      1. Query webui.db (most reliable, auto-updates if KB is recreated)
      2. Hardcoded UUID _SEBI_DOCS_KB_ID (fallback if DB unreachable)
    Used when Open-WebUI does not pass collection names in the request body.
    """
    db_path = _get_db_path()
    if db_path:
        try:
            conn = sqlite3.connect(db_path)
            row = conn.execute(
                "SELECT id FROM knowledge WHERE name = 'SEBI-DOCS' LIMIT 1"
            ).fetchone()
            conn.close()
            if row:
                print(f"[DocQA-Qwen] KB collection from DB: {row[0]}")
                return [row[0]]
        except Exception as e:
            print(f"[DocQA-Qwen] DB knowledge lookup failed: {e}")
    else:
        print(f"[DocQA-Qwen] webui.db not found (DATA_DIR={os.environ.get('DATA_DIR', 'not set')})")

    print(f"[DocQA-Qwen] Using hardcoded SEBI-DOCS KB ID: {_SEBI_DOCS_KB_ID}")
    return [_SEBI_DOCS_KB_ID]


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
                date_str = f"{m.group(1)} {m.group(2)} {m.group(3)}"
                return datetime.strptime(date_str, "%B %d %Y"), date_str
            elif fmt == "%d %B %Y":
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


def _get_file_created_at_map(file_ids: List[str]) -> dict:
    """
    Query webui.db for file.created_at (Unix timestamp) for a set of file_ids.
    Used as a fallback date when a chunk contains no parseable circular date.
    Returns {file_id: datetime} for any file_id found in the DB.
    """
    if not file_ids:
        return {}
    db_path = _get_db_path()
    if not db_path:
        return {}
    result = {}
    try:
        conn = sqlite3.connect(db_path)
        placeholders = ",".join("?" * len(file_ids))
        rows = conn.execute(
            f"SELECT id, created_at FROM file WHERE id IN ({placeholders})",
            file_ids,
        ).fetchall()
        conn.close()
        for fid, ts in rows:
            if ts:
                result[fid] = datetime.utcfromtimestamp(int(ts))
    except Exception as e:
        print(f"[DocQA-Qwen] DB created_at lookup failed: {e}")
    return result


def _rerank_by_date(chunk_tuples: List["_Chunk"]) -> List[str]:
    """
    Sort retrieved chunks newest-first using:
      1. Primary  — circular date extracted from the chunk text via regex
      2. Fallback — file.created_at from webui.db (upload time)
      3. Last resort — datetime.min (chunk floats to the bottom)

    Returns plain text strings with a [Source date: ...] label prepended.
    """
    if not chunk_tuples:
        return []

    # Collect all unique file_ids so we can batch-query the DB
    file_ids = list({
        meta.get("file_id", "") for _, meta in chunk_tuples if meta.get("file_id")
    })
    created_at_map = _get_file_created_at_map(file_ids)

    dated: List[Tuple[datetime, str, str]] = []
    for text, meta in chunk_tuples:
        dt, display = _extract_circular_date(text)
        if dt == datetime.min:
            # Fallback: use DB upload timestamp for this file
            fid = meta.get("file_id", "")
            if fid and fid in created_at_map:
                dt      = created_at_map[fid]
                display = f"~{dt.strftime('%b %d %Y')} (upload date)"
        dated.append((dt, display, text))

    # Sort descending: newest first
    dated.sort(key=lambda x: x[0], reverse=True)

    reranked = []
    for dt, display, text in dated:
        label = f"[Source date: {display}]\n"
        reranked.append(label + text)

    print(
        f"[DocQA-Qwen] Re-ranked {len(reranked)} chunks by date. "
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
    Document QA RAG Pipe — Qwen3-8B edition
    - Same RAG logic as document_qa_function.py
    - Uses qwen3-8b:latest (6.7 GB, 32 768-token context)
    - top_k=8 by default (vs 4 for gemma2:2b) for richer context
    """

    class Valves(BaseModel):
        ollama_base_url: str = Field(default=_OLLAMA_URL, description="Ollama server URL")
        webui_base_url:  str = Field(default=_WEBUI_URL,  description="Open-WebUI base URL (for internal retrieval)")
        chat_model:      str = Field(default=_CHAT_MODEL,  description="Ollama chat model name")
        top_k:           int = Field(default=_TOP_K,       description="Number of chunks to retrieve")
        num_ctx:         int = Field(default=_NUM_CTX,     description="Context window size (tokens)")

    def __init__(self):
        self.type    = "pipe"
        self.id      = "document_qa_rag_qwen"
        self.name    = "Document QA (RAG) — Qwen3-8B"
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
            with open(os.path.join(os.path.dirname(__file__), "debug_body_qwen.json"), "w") as f:
                json.dump(body, f, indent=2, default=str)
        except Exception as e:
            print(f"[DocQA-Qwen] Could not dump body: {e}")

        # ── Find uploaded file collections ────────────────────────────────────
        collection_names = _extract_collection_names(body)

        # Fallback: if body had no collections, read SEBI-DOCS KB from DB
        if not collection_names:
            collection_names = _get_sebi_kb_collection()

        # ── Detect date in question for targeted retrieval ──────────────────────
        date_hint    = _extract_query_date(user_msg)
        extra_queries: List[str] = _build_date_queries(date_hint) if date_hint else []
        if date_hint:
            print(f"[DocQA-Qwen] Date detected in query: {date_hint[0]} {date_hint[1]}")

        # ── Retrieve chunks ───────────────────────────────────────────
        chunk_tuples: List[_Chunk] = []
        if collection_names:
            chunk_tuples = _query_webui_collections(
                collection_names, user_msg, v.top_k, tok, v.webui_base_url,
                extra_queries=extra_queries,
            )

        # Re-rank by circular date (with DB fallback) so newest info comes first
        chunks: List[str] = _rerank_by_date(chunk_tuples) if chunk_tuples else []

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
            print(f"[DocQA-Qwen] Sending {len(chunks)} chunks to model. Question: {user_msg[:80]}")
        else:
            ollama_msgs.append({"role": "user", "content": user_msg})
            print(f"[DocQA-Qwen] No chunks found — answering without document context.")

        # ── Stream from Ollama ────────────────────────────────────────
        return self._stream(v.ollama_base_url, v.chat_model, ollama_msgs, v.num_ctx)

    def _stream(self, base_url: str, model: str, messages: List[dict], num_ctx: int) -> Generator:
        payload = {
            "model":    model,
            "messages": messages,
            "stream":   True,
            "options":  {
                "num_ctx":     num_ctx,
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
