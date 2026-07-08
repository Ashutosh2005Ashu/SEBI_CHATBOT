"""
setup_webui.py
==============
Auto-configures Open-WebUI after startup:
  1. Waits for Open-WebUI to be ready
  2. Signs in as admin (asks for credentials on first run, saves to .env)
  3. Uploads document_qa_function.py as an Open-WebUI Function
  4. Configures RAG settings:
       - Embedding engine : Ollama
       - Embedding model  : nomic-embed-text
       - Chunk size       : 1900 chars
       - Chunk overlap    : 240 chars
       - Top-K            : 4
  5. Disables tool-calling capability for gemma2:2b

Run this once after Open-WebUI starts, or include in start.bat.
"""

import os
import sys
import time
import json
import requests

WEBUI_URL = "http://localhost:8080"
FUNCTION_FILE = os.path.join(os.path.dirname(__file__), "pipelines", "document_qa_function.py")
ENV_FILE = os.path.join(os.path.dirname(__file__), ".webui_admin.env")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def wait_for_webui(max_wait=120):
    """Poll until Open-WebUI is up."""
    print(f"[setup] Waiting for Open-WebUI at {WEBUI_URL} â€¦", end="", flush=True)
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            r = requests.get(f"{WEBUI_URL}/health", timeout=3)
            if r.status_code == 200:
                print(" ready!")
                return True
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(3)
    print(" TIMED OUT")
    return False


def load_saved_creds():
    """Load saved admin credentials from .webui_admin.env"""
    if not os.path.exists(ENV_FILE):
        return None, None
    with open(ENV_FILE) as f:
        data = {}
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                data[k.strip()] = v.strip()
    return data.get("EMAIL"), data.get("PASSWORD")


def save_creds(email, password):
    with open(ENV_FILE, "w") as f:
        f.write(f"EMAIL={email}\nPASSWORD={password}\n")
    print(f"[setup] Credentials saved to {ENV_FILE}")


def signin(email, password):
    """Sign in to Open-WebUI and return auth token."""
    r = requests.post(
        f"{WEBUI_URL}/api/v1/auths/signin",
        json={"email": email, "password": password},
        timeout=10,
    )
    if r.status_code == 200:
        return r.json().get("token")
    return None


def get_admin_token():
    """Get admin token from env vars, saved creds, or command-line args."""
    # Priority 1: command-line args  (python setup_webui.py email password)
    if len(sys.argv) == 3:
        email, password = sys.argv[1], sys.argv[2]
        token = signin(email, password)
        if token:
            save_creds(email, password)
            print(f"[setup] Signed in as {email}")
            return token
        print(f"[setup] âŒ CLI credentials failed.")
        sys.exit(1)

    # Priority 2: environment variables
    email    = os.environ.get("WEBUI_EMAIL")
    password = os.environ.get("WEBUI_PASSWORD")
    if email and password:
        token = signin(email, password)
        if token:
            save_creds(email, password)
            print(f"[setup] Signed in as {email}")
            return token
        print(f"[setup] âŒ Env-var credentials failed.")
        sys.exit(1)

    # Priority 3: saved credentials file
    email, password = load_saved_creds()
    if email and password:
        token = signin(email, password)
        if token:
            print(f"[setup] Signed in as {email}")
            return token
        print(f"[setup] Saved credentials failed.")

    print("[setup] âŒ No credentials found. Run: python setup_webui.py <email> <password>")
    sys.exit(1)


def auth_headers(token):
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


# ---------------------------------------------------------------------------
# Step 1: Upload/update the Pipe Function
# ---------------------------------------------------------------------------

def upload_function(token):
    """Upload document_qa_function.py as an Open-WebUI function."""
    if not os.path.exists(FUNCTION_FILE):
        print(f"[setup] âš ï¸  Function file not found: {FUNCTION_FILE}")
        return False

    with open(FUNCTION_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    headers = auth_headers(token)
    func_id = "document_qa_rag"

    # Check if function already exists
    r = requests.get(f"{WEBUI_URL}/api/v1/functions/", headers=headers, timeout=10)
    existing_ids = []
    if r.status_code == 200:
        existing_ids = [fn.get("id") for fn in r.json()]

    payload = {
        "id":      func_id,
        "name":    "ðŸ“„ Document QA (RAG)",
        "content": content,
        "meta": {
            "description": "RAG pipeline for PDF document QA. No tool-calling required.",
        },
    }

    if func_id in existing_ids:
        # Update existing
        r = requests.post(
            f"{WEBUI_URL}/api/v1/functions/id/{func_id}/update",
            headers=headers,
            json=payload,
            timeout=15,
        )
    else:
        # Create new
        r = requests.post(
            f"{WEBUI_URL}/api/v1/functions/create",
            headers=headers,
            json=payload,
            timeout=15,
        )

    if r.status_code == 200:
        print("[setup] âœ… Function 'Document QA (RAG)' uploaded successfully.")
        return True
    else:
        print(f"[setup] âš ï¸  Function upload returned {r.status_code}: {r.text[:200]}")
        return False


# ---------------------------------------------------------------------------
# Step 2: Configure RAG settings
# ---------------------------------------------------------------------------

def configure_rag(token):
    """Set RAG to use Ollama embeddings + tuned chunk/top-k settings."""
    headers = auth_headers(token)

    rag_config = {
        "RAG_EMBEDDING_ENGINE":   "ollama",
        "RAG_EMBEDDING_MODEL":    "nomic-embed-text:latest",
        "RAG_OLLAMA_BASE_URL":    "http://localhost:11434",
        "CHUNK_SIZE":             1900,
        "CHUNK_OVERLAP":          240,
        "TOP_K":                  4,
        "ENABLE_RAG_HYBRID_SEARCH": False,
        "PDF_EXTRACT_IMAGES":     False,
        "RAG_FULL_CONTEXT":       False,  # Use chunk retrieval, not full-context mode
    }

    # Try the v1 config update endpoint
    r = requests.post(
        f"{WEBUI_URL}/api/v1/configs/",
        headers=headers,
        json=rag_config,
        timeout=10,
    )

    if r.status_code in (200, 201):
        print("[setup] âœ… RAG configuration updated.")
        return True

    # Fallback: try /api/v1/rag/config/update (older Open-WebUI versions)
    r2 = requests.post(
        f"{WEBUI_URL}/api/v1/rag/config/update",
        headers=headers,
        json=rag_config,
        timeout=10,
    )
    if r2.status_code in (200, 201):
        print("[setup] âœ… RAG configuration updated (via rag endpoint).")
        return True

    print(f"[setup] âš ï¸  RAG config update returned {r.status_code}. "
          "You may need to set these manually in Admin Panel â†’ Settings â†’ Documents:")
    print("        Embedding Engine : Ollama")
    print("        Embedding Model  : nomic-embed-text:latest")
    print("        Chunk Size       : 1900")
    print("        Top K            : 4")
    return False


# ---------------------------------------------------------------------------
# Step 3: Disable tool-calling for gemma2:2b
# ---------------------------------------------------------------------------

def disable_tools_for_gemma(token):
    """Mark gemma2:2b as not supporting tools to prevent the tool-call error."""
    headers = auth_headers(token)

    # Get current model list
    r = requests.get(f"{WEBUI_URL}/api/models", headers=headers, timeout=10)
    if r.status_code != 200:
        print(f"[setup] âš ï¸  Could not fetch models ({r.status_code}).")
        return False

    models = r.json().get("data", r.json() if isinstance(r.json(), list) else [])
    target_ids = [m.get("id", "") for m in models if "gemma" in m.get("id", "").lower()]

    if not target_ids:
        print("[setup] âš ï¸  gemma2:2b not found in model list.")
        return False

    for model_id in target_ids:
        payload = {
            "id": model_id,
            "meta": {
                "capabilities": {
                    "tools": False,         # â† disables tool-calling
                    "vision": False,
                },
            },
        }
        # Try to update the model override
        r2 = requests.post(
            f"{WEBUI_URL}/api/v1/models/update",
            headers=headers,
            json=payload,
            timeout=10,
        )
        if r2.status_code in (200, 201):
            print(f"[setup] âœ… Disabled tool-calling for '{model_id}'.")
        else:
            # Try alternate endpoint
            r3 = requests.post(
                f"{WEBUI_URL}/api/models/update",
                headers=headers,
                json=payload,
                timeout=10,
            )
            if r3.status_code in (200, 201):
                print(f"[setup] âœ… Disabled tool-calling for '{model_id}'.")
            else:
                print(f"[setup] âš ï¸  Could not update '{model_id}' ({r2.status_code}). "
                      "Manually go to Admin Panel â†’ Models â†’ gemma2:2b â†’ Edit â†’ "
                      "uncheck 'Tools' under Capabilities.")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Open-WebUI Document QA â€” Auto Setup")
    print("=" * 60)

    if not wait_for_webui():
        print("[setup] âŒ Open-WebUI did not start in time. Run this script after it starts.")
        sys.exit(1)

    token = get_admin_token()

    print("\n[setup] Uploading Document QA functionâ€¦")
    upload_function(token)

    print("\n[setup] Configuring RAG settingsâ€¦")
    configure_rag(token)

    print("\n[setup] Configuring model capabilitiesâ€¦")
    disable_tools_for_gemma(token)

    print("\n" + "=" * 60)
    print("  Setup complete!")
    print("  1. Refresh Open-WebUI in your browser")
    print("  2. Start a new chat")
    print("  3. Select 'ðŸ“„ Document QA (RAG)' as the model")
    print("  4. Upload a PDF and ask questions!")
    print("=" * 60)


if __name__ == "__main__":
    main()

