# Document QA Chatbot

A local, private PDF question-answering chatbot powered by:
- **Open-WebUI** — browser-based chat interface
- **Ollama + gemma2:2b** — local LLM (8k context window)
- **nomic-embed-text** — local embedding model
- **ChromaDB** — local vector store
- **LangChain** — text chunking

---

## Quick Start

### Prerequisites
1. **Ollama** must be running (`ollama serve` in a terminal, or the Ollama desktop app)
2. Models must be pulled:
   ```
   ollama pull gemma2:2b
   ollama pull nomic-embed-text
   ```

### Installation & Setup

1. **Create a virtual environment**:
   ```powershell
   python -m venv venv
   ```
2. **Activate the virtual environment**:
   * **Windows (PowerShell)**:
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   * **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```
3. **Install the dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

### Launch
Double-click `start.bat` or run from PowerShell:
```powershell
.\start.bat
```
Open-WebUI will be available at **http://localhost:8080**

---

## Setting Up the Pipeline in Open-WebUI

After first launch:

1. **Create an account** (first user becomes admin)
2. Navigate to **Admin Panel** → **Settings** → **Pipelines**
3. Click **"+"** to add a new pipeline
4. Upload `pipelines/document_qa_pipeline.py` **OR** set the pipelines directory path
5. The pipeline **"📄 Document QA (RAG)"** will appear in your model list

> **Tip:** You can also go to Admin Panel → Settings → Pipelines → Add Pipeline URL
> and point it to a running pipelines server if you prefer.

---

## Using the Chatbot

1. Start a new chat
2. Select **"📄 Document QA (RAG)"** as the model
3. Click the **📎 attachment** icon and upload a PDF
4. Ask questions about the document!

**Example questions:**
- "What is the main topic of this document?"
- "Summarize the key findings on page 5"
- "What are the conclusions?"
- "List all the recommendations mentioned"

---

## Context Window Strategy

The pipeline is tuned for `gemma2:2b`'s **8,192 token** context window:

| Component         | Tokens  |
|-------------------|---------|
| System prompt     | ~200    |
| Retrieved chunks  | ~2,048  |
| Conversation hist | ~1,000  |
| User question     | ~200    |
| **Answer budget** | **~4,744** |

- **Chunk size**: 1,900 chars (~512 tokens)
- **Chunk overlap**: 240 chars (~64 tokens)  
- **Top-K retrieved**: 4 chunks per query

---

## File Structure

```
chatbot/
├── venv/                    # Python virtual environment
├── pipelines/
│   └── document_qa_pipeline.py   # RAG pipeline for Open-WebUI
├── chroma_db/               # ChromaDB vector store (auto-created)
├── data/                    # Open-WebUI data directory (auto-created)
├── start.bat                # Launcher script
└── README.md
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Ollama is NOT running" | Run `ollama serve` or start Ollama app |
| PDF not being processed | Ensure file type is PDF; check pipeline is selected |
| Slow first response | First query embeds all chunks; subsequent queries are faster |
| ChromaDB errors | Delete `chroma_db/` folder to re-index |
| Port 8080 in use | Edit `start.bat` and change `--port 8080` |
