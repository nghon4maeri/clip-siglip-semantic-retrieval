import os
import subprocess
import time
import re
import sys

# 1. Download cloudflared if not present
if not os.path.exists("/content/cloudflared"):
    print("Downloading cloudflared binary...")
    os.system("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /content/cloudflared && chmod +x /content/cloudflared")
    print("cloudflared ready.")

# 2. Start python http.server in /content
print("Starting http.server on port 8080...")
server_proc = subprocess.Popen(["python3", "-m", "http.server", "8080"], cwd="/content")

# 3. Start cloudflared tunnel
print("Starting cloudflared tunnel...")
cf_proc = subprocess.Popen(
    ["/content/cloudflared", "tunnel", "--url", "http://127.0.0.1:8080"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

tunnel_url = None
start_time = time.time()
while time.time() - start_time < 30:
    line = cf_proc.stderr.readline()
    if not line:
        time.sleep(0.5)
        continue
    if "trycloudflare.com" in line:
        match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
        if match:
            tunnel_url = match.group(0)
            break

print(f"CLOUDFLARE_TUNNEL_URL: {tunnel_url}")
sys.stdout.flush()

# Keep tunnel alive for 10 minutes so download completes
if tunnel_url:
    print("Tunnel active. Serving files...")
    time.sleep(600)
