import os
import sys
import threading
import time
import re
import urllib.request
from http.server import SimpleHTTPRequestHandler, HTTPServer
import subprocess

# Ensure in /content
os.chdir("/content")
print(f"Serving from working directory: {os.getcwd()}", flush=True)
for f in os.listdir("/content"):
    if "zip" in f:
        print(f"  {f} ({os.path.getsize(f) / (1024**2):.2f} MB)", flush=True)

# Kill any previous cloudflared or port 8080 listeners
os.system("pkill -9 -f cloudflared; fuser -k 8080/tcp 2>/dev/null")
time.sleep(1)

# Start HTTPServer in a thread
httpd = HTTPServer(("0.0.0.0", 8080), SimpleHTTPRequestHandler)
server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
server_thread.start()
print("HTTPServer running on 0.0.0.0:8080", flush=True)
time.sleep(1)

# Verify locally
try:
    with urllib.request.urlopen("http://127.0.0.1:8080/newsseek_vector_stores.zip") as r:
        print(f"LOCAL VERIFY: HTTP {r.status}, Size: {r.headers.get('content-length')} bytes", flush=True)
except Exception as e:
    print(f"LOCAL VERIFY FAILED: {e}", flush=True)

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

print(f"ACTIVE_URL: {tunnel_url}", flush=True)

if tunnel_url:
    print("Serving files for 15 minutes...", flush=True)
    time.sleep(900)
