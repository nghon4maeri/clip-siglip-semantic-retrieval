import os
import sys
import time
import zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from colab_cli.common import state
from colab_cli.contents import ContentsClient

STORAGE_DIR = Path("storage/newsseek")
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_LOCAL_DIR = Path("storage/newsseek_chunks")
CHUNKS_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
ZIP_FILE = Path("storage/newsseek_vector_stores.zip")

SESSION_NAME = "newsseek-a100"
TOTAL_PARTS = 29

s = state.get_session(SESSION_NAME)
client = ContentsClient(s)

# Move test part if present
test_p0 = Path("storage/test_part_000.bin")
if test_p0.exists():
    target_p0 = CHUNKS_LOCAL_DIR / "part_000.bin"
    if not target_p0.exists() or target_p0.stat().st_size != test_p0.stat().st_size:
        import shutil
        shutil.move(str(test_p0), str(target_p0))

def download_part(part_idx):
    fname = f"part_{part_idx:03d}.bin"
    remote_path = f"content/chunks/{fname}"
    local_path = CHUNKS_LOCAL_DIR / fname
    # Part 000-027 are 25MB, part 028 is the remainder (~11.3MB)
    if local_path.exists() and local_path.stat().st_size > 5 * 1024 * 1024:
        return part_idx, True, "Cached"
    for attempt in range(3):
        try:
            client.download(remote_path, str(local_path))
            size_mb = local_path.stat().st_size / (1024**2)
            return part_idx, True, f"Done ({size_mb:.2f} MB)"
        except Exception as e:
            if attempt == 2:
                return part_idx, False, str(e)
            time.sleep(2)

print(f"=== Downloading 29 Chunks (711 MB) via Colab ThreadPool (4 workers) ===", flush=True)
t0 = time.time()
completed = 0

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(download_part, i): i for i in range(TOTAL_PARTS)}
    for fut in as_completed(futures):
        idx, ok, msg = fut.result()
        completed += 1
        elapsed = time.time() - t0
        mb_total = (completed * 25.0)
        speed = mb_total / max(elapsed, 0.001)
        print(f"[{completed:2d}/{TOTAL_PARTS}] Part {idx:03d}: {msg} | Overall: {mb_total:.1f} MB ({speed:.1f} MB/s)", flush=True)

print(f"\nAll chunks downloaded in {time.time() - t0:.1f}s.", flush=True)

# Reassemble ZIP
print(f"Reassembling chunks into {ZIP_FILE}...", flush=True)
with open(ZIP_FILE, "wb") as out_zip:
    for i in range(TOTAL_PARTS):
        part_path = CHUNKS_LOCAL_DIR / f"part_{i:03d}.bin"
        with open(part_path, "rb") as p_in:
            out_zip.write(p_in.read())

print(f"Reassembly complete: {ZIP_FILE.stat().st_size / (1024**2):.2f} MB", flush=True)

# Extract ZIP
print("Extracting archive into storage/newsseek/...", flush=True)
with zipfile.ZipFile(ZIP_FILE, "r") as zf:
    for member in zf.namelist():
        print(f"  Extracting: {member}", flush=True)
        target = Path(member)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as f_out:
            f_out.write(zf.read(member))

# Clean up chunk directory
import shutil
shutil.rmtree(CHUNKS_LOCAL_DIR, ignore_errors=True)
print("Cleaned up temporary chunk files.", flush=True)
print("=== ALL STORES EXTRACTED SUCCESSFULLY ===", flush=True)
