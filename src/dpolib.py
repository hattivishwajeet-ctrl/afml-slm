import os, json, math, random, time, torch, torch.nn.functional as F
from datasets import load_dataset
from dataclasses import asdict
from alignlib import *            # tok, PAD, EOT, USER, ASSIST, collate, load_ckpt, ROOT, dev ...

def load_uf(split, max_prompt=80, max_resp=160):
    path = f"{ROOT}/pairs_{split}.json"
    if os.path.exists(path): return json.load(open(path))
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=split)
    P, C, R = ds["prompt"], ds["chosen"], ds["rejected"]
    rows = [(p.strip(), c[-1]["content"].strip(), r[-1]["content"].strip()) for p, c, r in zip(P, C, R)]
    rows = [x for x in rows if len(x[0]) < 500 and len(x[1]) < 900 and len(x[2]) < 900 and x[1] and x[2] and x[1] != x[2]]
    pe = tok.encode_batch([x[0] for x in rows]); ce = tok.encode_batch([x[1] for x in rows]); re_ = tok.encode_batch([x[2] for x in rows])
    pairs = []
    for p, c, r in zip(pe, ce, re_):
        if len(p.ids) <= max_prompt and len(c.ids) < max_resp and len(r.ids) < max_resp:
            pairs.append(([USER] + p.ids + [ASSIST], c.ids + [EOT], r.ids + [EOT]))
    json.dump(pairs, open(path, "w")); return pairs

def pair_batch(pairs, idx):                   # first B rows = chosen, next B rows = rejected
    data = [(pairs[j][0], pairs[j][1]) for j in idx] + [(pairs[j][0], pairs[j][2]) for j in idx]
    return collate(data, list(range(len(data))))

def seq_logps(model, X, Y):                   # sum of response-token log-probs, and response length
    logits = model(X, full_logits=True)[0]
    nll = F.cross_entropy(logits.float().transpose(1, 2), Y, ignore_index=-100, reduction='none')
    return -nll.sum(1), (Y != -100).sum(1).float()

@torch.no_grad()
def get_ref(ref_name, pairs, tag, bs=64):
    path = f"{ROOT}/ref_{ref_name}_{tag}.pt"
    if os.path.exists(path): return torch.load(path)
    m = load_ckpt(ref_name); out = []
    for i in range(0, len(pairs), bs):
        idx = list(range(i, min(i + bs, len(pairs)))); X, Y = pair_batch(pairs, idx)
        with torch.autocast('cuda', dtype=torch.float16): s, _ = seq_logps(m, X, Y)
        out.append(torch.stack([s[:len(idx)], s[len(idx):]], 1).cpu())
    out = torch.cat(out); torch.save(out, path); del m; torch.cuda.empty_cache(); return out

def dpo_train(name, pairs, tag, ref_name="M_sft", loss="dpo", beta=0.1, gamma=1.0,
              lr=2e-5, epochs=2, bs=32, seed=0, force=False):
    out = f"{ROOT}/ckpt/{name}.pt"
    if os.path.exists(out) and not force:
        print(f"[{name}] already done -> skipping"); return json.load(open(f"{ROOT}/runs/{name}.json"))
    ref = get_ref(ref_name, pairs, tag) if loss != "simpo" else None
    policy = load_ckpt(ref_name).train()
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0, fused=True)
    scaler = torch.amp.GradScaler('cuda'); rng = random.Random(seed); N = len(pairs)
    order = []
    for _ in range(epochs):
        idx = list(range(N)); rng.shuffle(idx); order += [idx[i:i + bs] for i in range(0, N - bs + 1, bs)]
    steps = len(order); warm = max(1, min(20, steps // 10)); hist, win, t0 = [], [], time.time()
    print(f"[{name}] loss={loss} beta={beta} lr={lr} pairs={N} steps={steps}")
    for step, idx in enumerate(order):
        lr_s = lr * (step + 1) / warm if step < warm else lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm))))
        for g in opt.param_groups: g['lr'] = lr_s
        X, Y = pair_batch(pairs, idx); B = len(idx)
        with torch.autocast('cuda', dtype=torch.float16): s, n = seq_logps(policy, X, Y)
        pc, pr, nc, nr = s[:B], s[B:], n[:B], n[B:]
        if loss != "simpo": rc, rr = ref[idx, 0].to(dev), ref[idx, 1].to(dev)
        if loss == "dpo":
            z = (pc - pr) - (rc - rr); l = -F.logsigmoid(beta * z).mean(); margin = beta * z
        elif loss == "ipo":                      # sum log-ratios, target margin 1/(2*beta)
            z = (pc - pr) - (rc - rr); l = ((z - 1 / (2 * beta)) ** 2).mean(); margin = z
        elif loss == "simpo":                    # reference-free, length-normalised reward + target margin gamma
            z = beta * (pc / nc - pr / nr); l = -F.logsigmoid(z - gamma).mean(); margin = z
        scaler.scale(l).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        d = dict(loss=l.item(), acc=(margin > 0).float().mean().item(), margin=margin.mean().item(),
                 len_c=nc.mean().item(), len_r=nr.mean().item(),
                 d_chosen=(pc - rc).mean().item() if loss != "simpo" else 0.0,
                 d_rej=(pr - rr).mean().item() if loss != "simpo" else 0.0)
        win.append(d)
        if (step + 1) % 50 == 0 or step == steps - 1:
            avg = {k: sum(w[k] for w in win) / len(win) for k in win[0]}; avg['step'] = step + 1; hist.append(avg); win = []
            print(f"[{name}] {step+1}/{steps} loss {avg['loss']:.3f} acc {avg['acc']:.2f} margin {avg['margin']:.2f} "
                  f"dlogp chosen {avg['d_chosen']:+.1f} rejected {avg['d_rej']:+.1f} | {(time.time()-t0)/60:.1f} min")
    torch.save(dict(cfg=asdict(policy.c), model=policy.state_dict()), out)
    res = dict(name=name, loss=loss, beta=beta, gamma=gamma, lr=lr, epochs=epochs, pairs=N, hist=hist)
    json.dump(res, open(f"{ROOT}/runs/{name}.json", "w")); return res

@torch.no_grad()
def pref_acc(model, pairs, ref, bs=64):          # % of held-out pairs where policy's log-ratio favours 'chosen'
    model.eval(); ok = 0
    for i in range(0, len(pairs), bs):
        idx = list(range(i, min(i + bs, len(pairs)))); X, Y = pair_batch(pairs, idx)
        with torch.autocast('cuda', dtype=torch.float16): s, _ = seq_logps(model, X, Y)
        B = len(idx); z = (s[:B] - ref[idx, 0].to(dev)) - (s[B:] - ref[idx, 1].to(dev))
        ok += (z > 0).sum().item()
    return ok / len(pairs)
