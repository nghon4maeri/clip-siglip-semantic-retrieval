import sys
import time
import requests
import zipfile
from pathlib import Path

url = "https://squad-composed-baseline-understood.trycloudflare.com/newsseek_vector_stores.zip"
dest = Path("storage/newsseek_vector_stores.zip")
dest.parent.mkdir(parents=True, exist_ok=True)

print(f"Connecting to {url}...")
t0 = time.time()
with requests.get(url, stream=True, timeout=60) as r:
    r.raise_for_status()
    total_len = int(r.headers.get("content-length", 0))
    print(f"Total size: {total_len / (1024**2):.2f} MB")
    downloaded = 0
    last_print = 0
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if time.time() - last_print > 1.0 or downloaded == total_len:
                    mb = downloaded / (1024**2)
                    percent = (downloaded / total_len * 100) if total_len else 0
                    elapsed = time.time() - t0
                    speed = mb / max(elapsed, 0.001)
                    print(f"Downloaded {mb:.1f}/{total_len/(1024**2):.1f} MB ({percent:.1f}%) | Speed: {speed:.1f} MB/s", flush=True)
                    last_print = time.time()

print(f"\nDownload finished in {time.time() - t0:.1f}s.", flush=True)
print(f"File size on disk: {dest.stat().st_size / (1024**2):.2f} MB", flush=True)

print("Extracting archive into storage/newsseek/...", flush=True)
with zipfile.ZipFile(dest, "r") as zf:
    # Notice: inside zip, the structure might be storage/newsseek/... or direct
    for member in zf.namelist():
        print(f"  Archive item: {member}", flush=True)
        # Strip redundant prefix if needed
        target_path = Path("storage") / member if not member.startswith("storage") else Path(member)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "wb") as out_f:
            out_f.write(zf.read(member))

print("Extraction complete!", flush=True)
