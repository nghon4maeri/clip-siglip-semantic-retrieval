import os
import sys
import time
import re
import threading
import subprocess
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

# Clean up any lingering processes
os.system("pkill -9 -f cloudflared; rm -f /content/url.txt")
time.sleep(1)

# Ensure cloudflared binary
if not os.path.exists("/content/cloudflared"):
    print("Downloading cloudflared binary...", flush=True)
    os.system("curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /content/cloudflared && chmod +x /content/cloudflared")

# Start ThreadingHTTPServer serving /content
class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory="/content", **kwargs)

httpd = ThreadingHTTPServer(("127.0.0.1", 8080), Handler)
t = threading.Thread(target=httpd.serve_forever, daemon=True)
t.start()
print("Local HTTP Server started on 127.0.0.1:8080", flush=True)
time.sleep(1)

# Verify local file access
try:
    with urllib.request.urlopen("http://127.0.0.1:8080/newsseek_vector_stores.zip") as r:
        print(f"Local verification SUCCESS: HTTP {r.status}, Size: {r.headers.get('content-length')} bytes", flush=True)
except Exception as e:
    print(f"Local verification FAILED: {e}", flush=True)

# Start cloudflared
cf = subprocess.Popen(
    ["/content/cloudflared", "tunnel", "--url", "http://127.0.0.1:8080"],
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1
)

tunnel_url = None
t_start = time.time()
while time.time() - t_start < 25:
    line = cf.stderr.readline()
    if not line:
        continue
    if "trycloudflare.com" in line:
        m = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
        if m:
            tunnel_url = m.group(0)
            break

print(f"TUNNEL_ESTABLISHED: {tunnel_url}", flush=True)
if tunnel_url:
    with open("/content/url.txt", "w") as f:
        f.write(tunnel_url)
    print("Serving files for 15 minutes...", flush=True)
    time.sleep(900)
