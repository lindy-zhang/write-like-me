import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from model import GPT, Config

p = argparse.ArgumentParser()
p.add_argument("--ckpt_dir", default="checkpoints/finetune")
p.add_argument("--n_samples", type=int, default=5)
p.add_argument("--sample_len", type=int, default=500)
p.add_argument("--temperature", type=float, default=0.8)
p.add_argument("--eval_iters", type=int, default=50)
args = p.parse_args()

device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")

ROOT = Path(__file__).resolve().parent
ENC = ROOT / "data" / "encoded"
RESULTS = ROOT / "results"
(RESULTS / "samples").mkdir(parents=True, exist_ok=True)

vocab = json.loads((ENC / "meta.json").read_text())["vocab"]
stoi = {c: i for i, c in enumerate(vocab)}

def load_ids(name):
    return np.fromfile(ENC / f"{name}.bin", dtype=np.uint16)

def decode(ids):
    return "".join(vocab[i] for i in ids)

mine_val = torch.from_numpy(load_ids("mine_val").astype(np.int64))
corpus_val = torch.from_numpy(load_ids("corpus_val").astype(np.int64))
train_text = decode(load_ids("mine_train").tolist())
val_text = decode(load_ids("mine_val").tolist())

# ---------- copy-rate machinery ----------
NS = [8, 12, 20]   # chunk lengths in characters (~2, ~3, ~4 words)
train_grams = {n: {train_text[i:i + n] for i in range(len(train_text) - n + 1)} for n in NS}

def copy_rate(text, n):
    grams = [text[i:i + n] for i in range(len(text) - n + 1)]
    return sum(g in train_grams[n] for g in grams) / max(1, len(grams))

def longest_copied(text, n=12):
    # longest run of consecutive copied 12-char chunks = approx. longest verbatim span
    run = best = 0
    for i in range(len(text) - n + 1):
        run = run + 1 if text[i:i + n] in train_grams[n] else 0
        best = max(best, run)
    return best + n - 1 if best else 0

human = {n: copy_rate(val_text, n) for n in NS}
print("human baseline (your held-out paragraphs vs. your training paragraphs):")
print("  " + "  ".join(f"copy@{n}={human[n]:.3f}" for n in NS) + "\n")

# ---------- loss on fixed batches ----------
@torch.no_grad()
def eval_loss(model, data, block, batch=16):
    g = torch.Generator().manual_seed(0)   # identical batches for every checkpoint
    total = 0.0
    for _ in range(args.eval_iters):
        ix = torch.randint(len(data) - block - 1, (batch,), generator=g)
        x = torch.stack([data[i:i + block] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + 1 + block] for i in ix]).to(device)
        _, loss = model(x, y)
        total += loss.item()
    return total / args.eval_iters

# ---------- evaluate every checkpoint ----------
rows = []
for path in sorted(Path(args.ckpt_dir).glob("iter_*.pt")):
    ckpt = torch.load(path, map_location="cpu")
    cfg = Config(**ckpt["config"])
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    samples = []
    for s in range(args.n_samples):
        torch.manual_seed(1000 + s)   # same randomness for every checkpoint
        start = torch.tensor([[stoi["\n"]]], device=device)
        ids = model.generate(start, args.sample_len, temperature=args.temperature)[0].tolist()
        samples.append(decode(ids[1:]))

    row = {
        "iter": ckpt["iter"],
        "mine_val_loss": eval_loss(model, mine_val, cfg.block_size),
        "corpus_val_loss": eval_loss(model, corpus_val, cfg.block_size),
        **{f"copy@{n}": float(np.mean([copy_rate(t, n) for t in samples])) for n in NS},
        "longest_copied": max(longest_copied(t) for t in samples),
    }
    rows.append(row)

    (RESULTS / "samples" / f"iter_{row['iter']:05d}.txt").write_text(
        "\n\n=====\n\n".join(samples), encoding="utf-8")
    print(f"iter {row['iter']:5d} | mine val {row['mine_val_loss']:.3f} | "
          f"books val {row['corpus_val_loss']:.3f} | copy@12 {row['copy@12']:.3f} | "
          f"copy@20 {row['copy@20']:.3f} | longest {row['longest_copied']}")

with open(RESULTS / "metrics.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
(RESULTS / "human_baseline.json").write_text(json.dumps(human, indent=2))

# ---------- plot ----------
import matplotlib.pyplot as plt

its = [r["iter"] for r in rows]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
a1.plot(its, [r["mine_val_loss"] for r in rows], label="my writing (val)")
a1.plot(its, [r["corpus_val_loss"] for r in rows], label="essay books (val)")
a1.set_xlabel("fine-tuning step"); a1.set_ylabel("loss"); a1.legend(); a1.set_title("validation loss")
for n in [12, 20]:
    a2.plot(its, [r[f"copy@{n}"] for r in rows], label=f"model, {n}-char chunks")
    a2.axhline(human[n], linestyle="--", alpha=0.6, label=f"real me, {n}-char chunks")
a2.set_xlabel("fine-tuning step"); a2.set_ylabel("fraction copied"); a2.legend(); a2.set_title("copy rate")
fig.tight_layout()
fig.savefig(RESULTS / "curves.png", dpi=150)
print(f"\nsaved {RESULTS / 'metrics.csv'}, {RESULTS / 'curves.png'}, and samples in {RESULTS / 'samples'}")