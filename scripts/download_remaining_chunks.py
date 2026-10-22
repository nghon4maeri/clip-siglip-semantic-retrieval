import os
import sys
import time
import zipfile
import subprocess
from pathlib import Path

from colab_cli.common import state
from colab_cli.contents import ContentsClient

CHUNKS_DIR = Path("storage/newsseek_chunks")
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
ZIP_FILE = Path("storage/newsseek_vector_stores.zip")
SESSION_NAME = "newsseek-a100"
TOTAL_PARTS = 29

s = state.get_session(SESSION_NAME)
client = ContentsClient(s)

print(f"=== Starting NewsSeek Store Download (Total: {TOTAL_PARTS} parts) ===", flush=True)
t_start = time.time()

for p in range(TOTAL_PARTS):
    fname = f"part_{p:03d}.bin"
    local_path = CHUNKS_DIR / fname
    remote_path = f"content/chunks/{fname}"

    if local_path.exists() and local_path.stat().st_size > 5 * 1024 * 1024:
        print(f"[{p+1:2d}/{TOTAL_PARTS}] Part {p:03d}: Already downloaded ({local_path.stat().st_size/(1024**2):.2f} MB)", flush=True)
        continue

    print(f"[{p+1:2d}/{TOTAL_PARTS}] Downloading part {p:03d}...", end="", flush=True)
    t0 = time.time()
    success = False
    for attempt in range(3):
        try:
            client.download(remote_path, str(local_path))
            success = True
            break
        except Exception as e:
            print(f" [Retry {attempt+1}: {e}]", end="", flush=True)
            time.sleep(2)

    if success:
        elapsed_p = time.time() - t0
        size_mb = local_path.stat().st_size / (1024**2)
        print(f" Done ({size_mb:.2f} MB in {elapsed_p:.1f}s)", flush=True)
    else:
        print(f" FAILED part {p:03d}!", flush=True)
        sys.exit(1)

total_time = time.time() - t_start
print(f"\nAll {TOTAL_PARTS} parts downloaded successfully in {total_time:.1f}s ({total_time/60:.1f} min).", flush=True)

# 1. Reassemble
print(f"Reassembling chunks into {ZIP_FILE}...", flush=True)
with open(ZIP_FILE, "wb") as out_f:
    for p in range(TOTAL_PARTS):
        part_path = CHUNKS_DIR / f"part_{p:03d}.bin"
        with open(part_path, "rb") as in_f:
            out_f.write(in_f.read())

print(f"Archive reconstructed: {ZIP_FILE.stat().st_size / (1024**2):.2f} MB", flush=True)

# 2. Extract
print("Extracting vector stores to storage/newsseek/...", flush=True)
with zipfile.ZipFile(ZIP_FILE, "r") as zf:
    for name in zf.namelist():
        print(f"  Extracting: {name}", flush=True)
        target = Path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as f_out:
            f_out.write(zf.read(name))

# 3. Clean up chunks to save disk space
import shutil
shutil.rmtree(CHUNKS_DIR, ignore_errors=True)
print("Cleaned up temporary chunk files.", flush=True)

# 4. Stop Colab Session immediately to preserve compute units
print(f"Releasing Colab session '{SESSION_NAME}'...", flush=True)
try:
    subprocess.run(["colab", "stop", "-s", SESSION_NAME], check=True)
    print(f"Colab session '{SESSION_NAME}' stopped successfully. Compute units preserved.", flush=True)
except Exception as e:
    print(f"Warning: Failed to stop Colab session automatically: {e}", flush=True)

print("=== NEWSSEEK INGESTION & DOWNLOAD COMPLETE ===", flush=True)
