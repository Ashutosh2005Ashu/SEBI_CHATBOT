# SEBI Document QA Chatbot

A **local, private** SEBI compliance chatbot powered by RAG (Retrieval-Augmented Generation):

| Component | Role |
|---|---|
| **Open-WebUI** | Browser-based chat interface |
| **Ollama + gemma2:2b** | Default local LLM (8k context) |
| **Ollama + qwen3-8b** | Optional high-quality LLM (32k context) |
| **nomic-embed-text** | Local embedding model |
| **ChromaDB** | Local vector store (bundled in Open-WebUI) |
| **LangChain** | Text chunking for document ingestion |

All processing is **100% local** — no data leaves your machine.

---

## Prerequisites

Install these **before** running `pip install`:

1. **Python 3.11+** — https://www.python.org/downloads/
2. **Ollama** — https://ollama.com/download  
   After installing, pull the required models:
   ```powershell
   ollama pull nomic-embed-text   # required — used for RAG embeddings
   ollama pull gemma2:2b          # default chat model
   ollama pull qwen3-8b           # optional — higher quality, 32k context, 6.7 GB
   ```

---

## Installation

```powershell
# 1. Clone the repo
git clone https://github.com/Ashutosh2005Ashu/SEBI_CHATBOT.git
cd SEBI_CHATBOT

# 2. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1          # Windows PowerShell
# source venv/bin/activate           # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Launch

```powershell
# Make sure Ollama is running first:
ollama serve   # (or open the Ollama desktop app)

# Then start the chatbot:
.\start.bat
```

Open-WebUI will be available at **http://localhost:8080**

> On first launch, `start.bat` automatically runs `setup_webui.py` which uploads the pipeline function and configures RAG settings.

---

## Setting Up the Pipeline in Open-WebUI

After the first launch, register the pipeline function via the Admin Panel:

1. **Create an account** (first user automatically becomes admin)
2. Go to **Admin Panel → Functions → "+" (Create new function)**
3. Copy-paste the content of `pipelines/document_qa_function.py` → Save

**Optional — Qwen3-8B pipeline** (for better answers, requires ~7 GB RAM):
- Repeat step 3 with `pipelines/document_qa_qwen.py`
- It will appear as **"Document QA (RAG) — Qwen3-8B"** in the model list

---

## Using the Chatbot

### With the SEBI Knowledge Base (recommended)
1. Go to **Workspace → Knowledge** and open **SEBI-DOCS**
2. Start a new chat, enable the knowledge base
3. Select **"Document QA (RAG)"** or **"Document QA (RAG) — Qwen3-8B"** as the model
4. Ask SEBI compliance questions

### With a custom PDF
1. Start a new chat
2. Click the **📎 attachment** icon and upload a PDF
3. Ask questions about the document

**Example queries:**
- "What are the audit requirements for stock exchanges?"
- "Summarize the BCP guidelines from the latest circular"
- "What changed in SEBI's cybersecurity framework in 2024?"

---

## How Date-Aware Retrieval Works

When multiple circulars contain conflicting rules, the chatbot **always follows the most recent circular**:

1. **Regex scan** — extracts the circular issue date from the chunk text (e.g., `March 22, 2021`)
2. **DB fallback** — if no date in the text, uses the file's upload timestamp from `webui.db`
3. **Sorted newest-first** — chunks are re-ranked before being sent to the LLM
4. **LLM instructed** — system prompt explicitly tells the model to cite the most recent circular on conflicts

---

## Context Window Budget

| | gemma2:2b pipeline | qwen3-8b pipeline |
|---|---|---|
| **Context window** | 8,192 tokens | 32,768 tokens |
| **Chunks retrieved** | 4 | 8 |
| **Chunk size** | 1,500 chars | 1,500 chars |

---

## File Structure

```
SEBI_CHATBOT/
├── pipelines/
│   ├── document_qa_function.py   # RAG pipeline — gemma2:2b (default)
│   └── document_qa_qwen.py       # RAG pipeline — qwen3-8b (optional)
├── data/                         # Open-WebUI data dir (auto-created)
│   ├── uploads/                  # Uploaded PDF files
│   ├── vector_db/                # ChromaDB vector store
│   └── webui.db                  # SQLite metadata store
├── setup_webui.py                # Auto-configures Open-WebUI on startup
├── start.bat                     # Windows launcher
├── requirements.txt
└── README.md
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `"Ollama is NOT running"` | Run `ollama serve` or open the Ollama desktop app |
| `405` on `/api/v1/models/update` at startup | Harmless — Open-WebUI changed this API endpoint. Ignore it |
| First query is slow | Ollama loads the model on first request (~10-30s for qwen3-8b) |
| PDF not being processed | Check that the pipeline function is uploaded and enabled |
| Port 8080 in use | Edit `start.bat` and change `--port 8080` to another port |
| `nomic-embed-text` missing | Run `ollama pull nomic-embed-text` |
