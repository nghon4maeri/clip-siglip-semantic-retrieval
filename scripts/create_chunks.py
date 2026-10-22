import os
import sys
from pathlib import Path

chunk_dir = Path("/content/chunks")
chunk_dir.mkdir(exist_ok=True)

print("Starting in-memory file chunking (25MB each)...", flush=True)
zip_path = Path("/content/newsseek_vector_stores.zip")
total_size = zip_path.stat().st_size

part_idx = 0
with open(zip_path, "rb") as f_in:
    while True:
        chunk = f_in.read(25 * 1024 * 1024)
        if not chunk:
            break
        part_file = chunk_dir / f"part_{part_idx:03d}.bin"
        with open(part_file, "wb") as f_out:
            f_out.write(chunk)
        part_idx += 1

print(f"Finished: {part_idx} parts created for {total_size / (1024**2):.2f} MB.", flush=True)
