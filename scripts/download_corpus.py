import re
import requests
from pathlib import Path

# Project Gutenberg IDs for public-domain essay collections
BOOKS = {
    "montaigne": 3600,      # Essays of Montaigne (Cotton translation)
    "emerson_1": 2944,      # Emerson, Essays: First Series
    "emerson_2": 2945,      # Emerson, Essays: Second Series
    "lamb_elia": 10343,     # Charles Lamb, Essays of Elia
}

OUT = Path(__file__).resolve().parent.parent / "data" / "corpus"
OUT.mkdir(parents=True, exist_ok=True)

def strip_gutenberg(text: str) -> str:
    # keep only the text between Gutenberg's START and END markers
    start = re.search(r"\*\*\* ?START OF.*?\*\*\*", text)
    end = re.search(r"\*\*\* ?END OF.*?\*\*\*", text)
    if start and end:
        text = text[start.end():end.start()]
    return text.strip()

for name, gid in BOOKS.items():
    url = f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"
    print(f"downloading {name} from {url}")
    raw = requests.get(url, timeout=30).text
    clean = strip_gutenberg(raw)
    (OUT / f"{name}.txt").write_text(clean, encoding="utf-8")
    print(f"  {len(clean):,} characters; starts with: {clean[:80]!r}")