import json, os, random, math, numpy as np, torch
from datasets import load_dataset
from tokenizers import Tokenizer
from slm import GPT, Cfg
ROOT = '/content/drive/MyDrive/afml_slm'; dev = 'cuda'
os.makedirs(f"{ROOT}/gens", exist_ok=True)
tok = Tokenizer.from_file(f"{ROOT}/tokenizer.json")
PAD, EOT, USER, ASSIST = [tok.token_to_id(t) for t in ["<|pad|>", "<|eot|>", "<|user|>", "<|assistant|>"]]

def prompt_text(ex):
    return ex["instruction"].strip() + (("\n" + ex["input"].strip()) if ex["input"].strip() else "")
def chat_ids(prompt):  return [USER] + tok.encode(prompt).ids + [ASSIST]                   # SFT / DPO models
def plain_ids(prompt): return tok.encode("Instruction: " + prompt + "\nResponse:").ids     # base model (never saw chat tokens)

def load_ckpt(name):
    ck = torch.load(f"{ROOT}/ckpt/{name}.pt", map_location=dev)
    m = GPT(Cfg(**ck['cfg'])).to(dev); m.load_state_dict(ck['model']); return m.eval()

def alpaca_splits(seed=0, n_test=300, n_val=500):
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    ex = [dict(instruction=d["instruction"], input=d["input"], output=d["output"]) for d in ds]
    random.Random(seed).shuffle(ex)
    return ex[n_test + n_val:], ex[n_test:n_test + n_val], ex[:n_test]       # train, val, test

def encode_sft(exs, max_len=256):
    P = tok.encode_batch([prompt_text(e) for e in exs]); R = tok.encode_batch([e["output"].strip() for e in exs])
    out = []
    for p, r in zip(P, R):
        pi, ri = [USER] + p.ids + [ASSIST], r.ids + [EOT]
        if len(pi) > max_len - 16: continue                   # prompt too long -> drop
        if len(pi) + len(ri) > max_len: ri = ri[:max_len - len(pi)]   # long response -> truncate
        out.append((pi, ri))
    return out

def sft_batches(data, bs, rng):        # length-bucketed batches to waste less padding
    idx = list(range(len(data))); rng.shuffle(idx); batches = []
    for i in range(0, len(idx), bs * 50):
        c = sorted(idx[i:i + bs * 50], key=lambda j: len(data[j][0]) + len(data[j][1]))
        batches += [c[k:k + bs] for k in range(0, len(c), bs)]
    rng.shuffle(batches); return batches

def collate(data, b):                  # loss only on response tokens (prompt labels = -100)
    seqs = [data[j][0] + data[j][1] for j in b]; L = max(len(s) for s in seqs)
    X = torch.full((len(b), L - 1), PAD, dtype=torch.long); Y = torch.full((len(b), L - 1), -100, dtype=torch.long)
    for i, j in enumerate(b):
        p, r = data[j]; s = p + r
        X[i, :len(s) - 1] = torch.tensor(s[:-1]); Y[i, len(p) - 1:len(s) - 1] = torch.tensor(r)
    return X.to(dev), Y.to(dev)

@torch.no_grad()
def sft_val_loss(model, data, bs=64):
    model.eval(); tot = n = 0
    for b in range(0, len(data), bs):
        X, Y = collate(data, list(range(b, min(b + bs, len(data)))))
        with torch.autocast('cuda', dtype=torch.float16):
            _, loss, _ = model(X, Y)
        k = (Y != -100).sum().item(); tot += loss.item() * k; n += k
    model.train(); return tot / n

@torch.no_grad()
def gen_responses(model, prompts, chat=True, max_new=100, temperature=0.7, top_k=40, seed=0):
    model.eval(); torch.manual_seed(seed); outs = []
    for p in prompts:
        ids = chat_ids(p) if chat else plain_ids(p)
        with torch.autocast('cuda', dtype=torch.float16):
            y = model.generate(torch.tensor([ids], device=dev), max_new=max_new,
                               temperature=temperature, top_k=top_k, stop_id=EOT)
        new = y[0, len(ids):].tolist()
        if EOT in new: new = new[:new.index(EOT)]
        outs.append(tok.decode(new).strip())
    return outs
