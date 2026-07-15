@echo off
:: ============================================================
:: Document QA Chatbot — Startup Script
:: Open-WebUI + Ollama (gemma2:2b) + ChromaDB RAG Pipeline
:: ============================================================

title Document QA Chatbot

:: Activate virtual environment
call "%~dp0venv\Scripts\activate.bat"

:: Verify Ollama is running
echo [*] Checking Ollama...
curl -s http://localhost:11434/api/tags > nul 2>&1
if errorlevel 1 (
    echo [!] Ollama is NOT running. Please start Ollama first.
    echo     Run: ollama serve
    pause
    exit /b 1
)
echo [OK] Ollama is running.

:: ============================================================
:: Open-WebUI core settings
:: ============================================================
set WEBUI_SECRET_KEY=document-qa-secret-key-change-me
set DATA_DIR=%~dp0data

:: ============================================================
:: RAG: use Ollama for embeddings (nomic-embed-text)
:: This avoids any dependency on OpenAI or cloud APIs
:: ============================================================
set RAG_EMBEDDING_ENGINE=ollama
set RAG_EMBEDDING_MODEL=nomic-embed-text:latest
set OLLAMA_BASE_URL=http://localhost:11434
set RAG_OLLAMA_BASE_URL=http://localhost:11434

:: Chunking strategy tuned for gemma2:2b 8k context window
:: 1500 chars ≈ 400 tokens | overlap 200 ≈ 50 tokens | top-k=6 → ~2400 tokens retrieved
set CHUNK_SIZE=1500
set CHUNK_OVERLAP=200
set TOP_K=6

:: Disable features that trigger tool-calling (not supported by gemma2:2b)
set ENABLE_RAG_WEB_SEARCH=False
set ENABLE_WEB_SEARCH=False
set ENABLE_SEARCH_QUERY_GENERATION=False
set ENABLE_RETRIEVAL_QUERY_GENERATION=False

:: Enable image extraction from PDFs (OCR for scanned documents)
set PDF_EXTRACT_IMAGES=True
set ENABLE_RAG_HYBRID_SEARCH=True

:: Content extraction engine (default uses PyPDF, tika adds OCR support)
set CONTENT_EXTRACTION_ENGINE=default

:: ============================================================
echo.
echo ============================================================
echo  Document QA Chatbot starting...
echo  URL: http://localhost:8080
echo ============================================================
echo.

:: Launch setup script in background (configures function + RAG via API)
:: It waits for Open-WebUI to be ready before running
start "DocQA Setup" /min cmd /c "python "%~dp0setup_webui.py" & pause"

:: Start Open-WebUI
open-webui serve --port 8080

pause
