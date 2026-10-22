import os
import subprocess
import time
import re
import sys
import urllib.request

# Ensure binary exists
if not os.path.exists("/content/cloudflared"):
    print("Downloading cloudflared binary...")
    os.system("curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /content/cloudflared && chmod +x /content/cloudflared")
    print("cloudflared binary ready.")

# Clean up any lingering processes
os.system("pkill -9 -f http.server; pkill -9 -f cloudflared")
time.sleep(1)

# Start http server properly with --directory BEFORE port
print("Starting http.server on port 8080 serving /content...")
server = subprocess.Popen(["python3", "-m", "http.server", "--directory", "/content", "8080"])
time.sleep(1)

# Verify local endpoint
try:
    code = urllib.request.urlopen("http://127.0.0.1:8080/newsseek_vector_stores.zip").status
    print(f"Local verification SUCCESS: HTTP {code}")
except Exception as e:
    print(f"Local verification FAILED: {e}")

# Start cloudflared with logfile
print("Starting cloudflared tunnel...")
cf = subprocess.Popen(["/content/cloudflared", "tunnel", "--url", "http://127.0.0.1:8080", "--logfile", "/content/tunnel.log"])

tunnel_url = None
for _ in range(30):
    time.sleep(1)
    if os.path.exists("/content/tunnel.log"):
        with open("/content/tunnel.log", "r") as f:
            content = f.read()
            match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", content)
            if match:
                tunnel_url = match.group(0)
                break

print(f"FOUND_TUNNEL_URL: {tunnel_url}", flush=True)

# Keep alive for 15 minutes
if tunnel_url:
    print("Serving files for 15 minutes...", flush=True)
    time.sleep(900)
