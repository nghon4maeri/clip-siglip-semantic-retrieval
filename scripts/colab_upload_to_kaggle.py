import os
import subprocess
import json
from pathlib import Path

out_dir = Path("/content/output/storage/newsseek")
meta_file = out_dir / "dataset-metadata.json"

# Copy benchmark json into the folder so it is included
os.system("cp /content/output/newsseek_benchmark_results.json /content/output/storage/newsseek/newsseek_benchmark_results.json")

meta = {
    "title": "NewsSeek Multimodal Vector Stores",
    "id": "namnguynnnn/newsseek-vector-stores",
    "licenses": [{"name": "CC0-1.0"}]
}
with open(meta_file, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print("Listing files in output directory:")
for f in out_dir.rglob("*"):
    if f.is_file():
        print(f"  {f.name} ({f.stat().st_size / (1024**2):.1f} MB)")

print("\nCreating Kaggle dataset...")
res = subprocess.run(["kaggle", "datasets", "create", "-p", str(out_dir), "--dir-mode", "zip"], capture_output=True, text=True)
print("STDOUT:", res.stdout)
print("STDERR:", res.stderr)
print("Returncode:", res.returncode)
