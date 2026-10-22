import os
import time
import re

print("Setting up Cloudflare binary...", flush=True)
if not os.path.exists("/content/cloudflared"):
    os.system("curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /content/cloudflared && chmod +x /content/cloudflared")

print("Stopping previous daemons...", flush=True)
os.system("pkill -9 -f http.server; pkill -9 -f cloudflared; rm -f /content/tunnel.log")
time.sleep(1)

print("Starting background http.server and cloudflared...", flush=True)
os.system("nohup python3 -m http.server --directory /content 8080 > /content/http.log 2>&1 &")
time.sleep(1)
os.system("nohup /content/cloudflared tunnel --url http://127.0.0.1:8080 --logfile /content/tunnel.log > /dev/null 2>&1 &")

tunnel_url = None
for i in range(15):
    time.sleep(1)
    if os.path.exists("/content/tunnel.log"):
        with open("/content/tunnel.log", "r") as f:
            content = f.read()
            match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", content)
            if match:
                tunnel_url = match.group(0)
                break

print(f"READY_TUNNEL_URL: {tunnel_url}", flush=True)
