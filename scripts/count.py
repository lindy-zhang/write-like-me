from pathlib import Path

root = Path(__file__).resolve().parent.parent / "data"
for folder in ["my_writing", "corpus"]:
    files = sorted((root / folder).glob("*.txt"))
    total = 0
    for f in files:
        n = len(f.read_text(encoding="utf-8"))
        total += n
        print(f"{folder}/{f.name}: {n:,} chars")
    print(f"TOTAL {folder}: {total:,} chars\n")