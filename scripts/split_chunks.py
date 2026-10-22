import os
import subprocess
from pathlib import Path

chunks_dir = Path("/content/chunks")
chunks_dir.mkdir(exist_ok=True)

# Split into 25MB chunks
cmd = "split -b 25M /content/newsseek_vector_stores.zip /content/chunks/part_"
print("Splitting /content/newsseek_vector_stores.zip into 25MB chunks...")
ret = os.system(cmd)
print(f"Split completed with exit code: {ret}")

parts = sorted(list(chunks_dir.glob("part_*")))
print(f"Total parts generated: {len(parts)}")
for p in parts[:3]:
    print(f"  {p.name}: {p.stat().st_size / (1024**2):.2f} MB")
