import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = DATA / "encoded"
OUT.mkdir(exist_ok=True)

MAX_CHARS_PER_BOOK = 1_000_000   # keep Montaigne from dominating
VAL_FRAC = 0.1                   # 10% held out for validation
MIN_COUNT = 20                   # rare characters in the corpus get dropped
SEED = 0

# Make my writing and the old books use the same characters.
# Gutenberg uses straight quotes, "--" for dashes, and _underscores_ for italics.
REPLACEMENTS = {
    "\u201c": '"', "\u201d": '"',    # curly double quotes
    "\u2018": "'", "\u2019": "'",    # curly single quotes / apostrophes
    "\u2026": "...",                 # ellipsis character
    "\u00a0": " ",                   # non-breaking space
    "--": "\u2014",                  # double hyphen -> em dash
    "_": "",                         # Gutenberg italics markers
}

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    for old, new in REPLACEMENTS.items():
        text = text.replace(old, new)
    return re.sub(r"[ \t]+", " ", text)

def corpus_paragraphs(text: str) -> list[str]:
    # Gutenberg hard-wraps lines at ~70 characters, so a paragraph is
    # a block of lines separated by blank lines. Rejoin each block into one line.
    paras = []
    for block in re.split(r"\n\s*\n", text):
        p = " ".join(line.strip() for line in block.splitlines()).strip()
        if len(p) >= 40:  # skip headings, page numbers, stray fragments
            paras.append(p)
    return paras

def my_paragraphs(text: str) -> list[str]:
    # My files have one paragraph (or poem line) per line.
    return [line.strip() for line in text.splitlines() if line.strip()]

# ---------- load and split the corpus ----------
corpus_train, corpus_val = [], []
for path in sorted((DATA / "corpus").glob("*.txt")):
    paras = corpus_paragraphs(normalize(path.read_text(encoding="utf-8")))
    kept, total = [], 0
    for p in paras:
        if total >= MAX_CHARS_PER_BOOK:
            break
        kept.append(p)
        total += len(p)
    cut = int(len(kept) * (1 - VAL_FRAC))   # last 10% of each book -> validation
    corpus_train += kept[:cut]
    corpus_val += kept[cut:]
    print(f"corpus/{path.name}: kept {total:,} chars, {len(kept)} paragraphs")

# ---------- load and split my writing ----------
mine = []
for path in sorted((DATA / "my_writing").glob("*.txt")):
    mine += my_paragraphs(normalize(path.read_text(encoding="utf-8")))

random.seed(SEED)
val_idx = set(random.sample(range(len(mine)), max(1, int(len(mine) * VAL_FRAC))))
mine_train = [p for i, p in enumerate(mine) if i not in val_idx]  # keeps original order
mine_val = [p for i, p in enumerate(mine) if i in val_idx]

# ---------- build one shared vocabulary ----------
# Pretraining and fine-tuning must use the SAME character -> number mapping,
# because the model's embedding table is indexed by these numbers.
splits = {
    "corpus_train": "\n".join(corpus_train),
    "corpus_val": "\n".join(corpus_val),
    "mine_train": "\n".join(mine_train),
    "mine_val": "\n".join(mine_val),
}
counts = Counter("".join(splits.values()))
my_chars = set(splits["mine_train"] + splits["mine_val"])
vocab = sorted({c for c, n in counts.items() if n >= MIN_COUNT} | my_chars)
dropped = sorted(set(counts) - set(vocab))

stoi = {c: i for i, c in enumerate(vocab)}
for name, text in splits.items():
    ids = np.array([stoi[c] for c in text if c in stoi], dtype=np.uint16)
    ids.tofile(OUT / f"{name}.bin")
    print(f"{name}: {len(ids):,} tokens")

(OUT / "meta.json").write_text(json.dumps({"vocab": vocab}, ensure_ascii=False))

print(f"\nvocab size: {len(vocab)}")
print(f"dropped rare characters: {''.join(dropped)!r}")
print(f"\nmy writing: {len(mine_train)} train paragraphs, {len(mine_val)} val paragraphs")
print("\n--- corpus sample ---\n" + splits["corpus_train"][5000:5300])
print("\n--- my writing sample ---\n" + splits["mine_train"][2000:2300])