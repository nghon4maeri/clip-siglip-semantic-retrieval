from pathlib import Path

p = Path("/content/output/storage/newsseek")
for f in p.rglob("*"):
    if f.is_file():
        print(f"{f.relative_to(p)}: {f.stat().st_size / (1024**2):.2f} MB")
