import argparse
import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from model import GPT, Config

# command line arguments

p = argparse.ArgumentParser()
p.add_argument("--data", choices=["corpus", "mine"], required=True)
p.add_argument("--out", required=True)                  # folder for checkpoints + log
p.add_argument("--init", default=None)                  # checkpoint to start from (fine-tuning)
p.add_argument("--max_iters", type=int, default=3000)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--min_lr_frac", type=float, default=0.1)
p.add_argument("--warmup", type=int, default=200)
p.add_argument("--batch_size", type=int, default=32)
p.add_argument("--dropout", type=float, default=0.1)
p.add_argument("--eval_interval", type=int, default=250)
p.add_argument("--eval_iters", type=int, default=50)
p.add_argument("--save_every", type=int, default=0)     # 0 = only save the best checkpoint
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

# device selection
torch.manual_seed(args.seed)
device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")
print(f"device: {device}")

# data
ROOT = Path(__file__).resolve().parent
ENC = ROOT / "data" / "encoded"
vocab = json.loads((ENC / "meta.json").read_text())["vocab"]
stoi = {c: i for i, c in enumerate(vocab)}
data = {
    split: torch.from_numpy(np.fromfile(ENC / f"{args.data}_{split}.bin", dtype=np.uint16).astype(np.int64))
    for split in ["train", "val"]
}

# model
if args.init:
    # checkpoint
    ckpt = torch.load(args.init, map_location="cpu")
    cfg = Config(**ckpt["config"])
    cfg.dropout = args.dropout
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    print(f"loaded weights from {args.init}")
else:
    # new randomly initialized model
    cfg = Config(vocab_size=len(vocab), dropout=args.dropout)
    model = GPT(cfg)
model.to(device)

# use AdamW
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

# helper funcs
def get_batch(split):
    d = data[split]
    ix = torch.randint(len(d) - cfg.block_size - 1, (args.batch_size,))
    x = torch.stack([d[i : i + cfg.block_size] for i in ix])
    y = torch.stack([d[i + 1 : i + 1 + cfg.block_size] for i in ix])   # shifted by one
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            _, loss = model(*get_batch(split))
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def lr_at(it):
    # linear warmup, then cosine decay down to min_lr
    if it < args.warmup:
        return args.lr * (it + 1) / args.warmup
    progress = (it - args.warmup) / max(1, args.max_iters - args.warmup)
    min_lr = args.lr * args.min_lr_frac
    return min_lr + 0.5 * (args.lr - min_lr) * (1 + math.cos(math.pi * progress))

@torch.no_grad()
def sample(n_chars=300):
    model.eval()
    start = torch.tensor([[stoi["\n"]]], device=device)
    ids = model.generate(start, n_chars, temperature=0.8)[0].tolist()
    model.train()
    return "".join(vocab[i] for i in ids).strip()

def save(path, it, val_loss):
    torch.save({"model": model.state_dict(), "config": asdict(cfg),
                "iter": it, "val_loss": val_loss}, path)

# training loop
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
log_file = open(out / "log.csv", "w", newline="")
log = csv.writer(log_file)
log.writerow(["iter", "train_loss", "val_loss", "lr", "seconds"])

best_val = float("inf")
t0 = time.time()
model.train()

for it in range(args.max_iters + 1):
    lr = lr_at(it) # set learning rate
    for g in opt.param_groups:
        g["lr"] = lr

    # every 250 steps (+ final step), check progress
    if it % args.eval_interval == 0 or it == args.max_iters:
        losses = estimate_loss()
        elapsed = time.time() - t0
        print(f"\niter {it:5d} | train {losses['train']:.3f} | val {losses['val']:.3f} | lr {lr:.2e} | {elapsed:.0f}s")
        log.writerow([it, losses["train"], losses["val"], lr, round(elapsed)])
        log_file.flush()
        if losses["val"] < best_val:
            best_val = losses["val"]
            save(out / "best.pt", it, best_val)
        if args.save_every and it % args.save_every == 0:
            save(out / f"iter_{it:05d}.pt", it, losses["val"])
        print("---- sample ----\n" + sample() + "\n----------------")

    if it == args.max_iters:
        break

    x, y = get_batch("train")
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

log_file.close()
print(f"\ndone. best val loss {best_val:.3f}, saved to {out / 'best.pt'}")
