import setup_webui
import sys

print("[updater] Getting token...")
token = setup_webui.get_admin_token()

if token:
    print("[updater] Forcing function update...")
    if setup_webui.upload_function(token):
        print("Success!")
    else:
        print("Failed.")
else:
    print("Failed to get token.")
