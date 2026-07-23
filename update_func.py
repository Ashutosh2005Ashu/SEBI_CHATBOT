"""
update_func.py
==============
Push updated pipeline function(s) to a running Open-WebUI instance.

Usage:
    python update_func.py          # update gemma2:2b pipeline (default)
    python update_func.py qwen     # update qwen3-8b pipeline
    python update_func.py all      # update BOTH pipelines
"""

import setup_webui
import sys

target = sys.argv[1].lower() if len(sys.argv) > 1 else "gemma"

print("[updater] Getting admin token...")
token = setup_webui.get_admin_token()

if not token:
    print("[updater] Failed to get token. Is Open-WebUI running?")
    sys.exit(1)

if target in ("gemma", "default"):
    print("[updater] Updating Document QA (RAG) — gemma2:2b ...")
    ok = setup_webui.upload_function(token)
    print("[updater] Done!" if ok else "[updater] Failed.")

elif target == "qwen":
    print("[updater] Updating Document QA (RAG) — Qwen3-8B ...")
    ok = setup_webui.upload_qwen_function(token)
    print("[updater] Done!" if ok else "[updater] Failed.")

elif target == "all":
    print("[updater] Updating both pipelines...")
    ok1 = setup_webui.upload_function(token)
    ok2 = setup_webui.upload_qwen_function(token)
    if ok1 and ok2:
        print("[updater] Both pipelines updated successfully!")
    else:
        print(f"[updater] Gemma: {'OK' if ok1 else 'FAILED'} | Qwen: {'OK' if ok2 else 'FAILED'}")

else:
    print(f"[updater] Unknown target '{target}'. Use: gemma | qwen | all")
    sys.exit(1)
