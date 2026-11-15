import os, json, math, time, numpy as np, torch
from dataclasses import asdict
from slm import GPT, Cfg
ROOT = '/content/drive/MyDrive/afml_slm'
dev = 'cuda'
os.makedirs(f"{ROOT}/runs", exist_ok=True); os.makedirs(f"{ROOT}/ckpt", exist_ok=True)

_train = np.fromfile(f"{ROOT}/train.bin", dtype=np.uint16)   # loaded into RAM (~140MB)
_val   = np.fromfile(f"{ROOT}/val.bin",   dtype=np.uint16)

def get_batch(bs, T, gen):
    ix = torch.randint(0, len(_train) - T - 1, (bs,), generator=gen).numpy()
    idx = ix[:, None] + np.arange(T)[None, :]
    x = torch.from_numpy(_train[idx].astype(np.int64)); y = torch.from_numpy(_train[idx + 1].astype(np.int64))
    return x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)

@torch.no_grad()
def evaluate(model, T=256, max_seqs=None, bs=64, data=None):
    """mean next-token loss over non-overlapping length-T chunks of the val set"""
    d = _val if data is None else data
    model.eval()
    n = (len(d) - 1) // T
    if max_seqs: n = min(n, max_seqs)
    arr = torch.from_numpy(d[:n * T + 1].astype(np.int64))
    X, Y = arr[:n * T].view(n, T), arr[1:n * T + 1].view(n, T)
    tot, cnt = 0.0, 0
    for i in range(0, n, bs):
        x, y = X[i:i+bs].to(dev), Y[i:i+bs].to(dev)
        with torch.autocast('cuda', dtype=torch.float16):
            _, loss, _ = model(x, y)
        tot += loss.item() * x.numel(); cnt += x.numel()
    model.train()
    return tot / cnt

def load_model(name):
    ck = torch.load(f"{ROOT}/ckpt/{name}.pt", map_location=dev)
    m = GPT(Cfg(**ck['cfg'])).to(dev); m.load_state_dict(ck['model']); return m.eval()

def train(name, cfg, tokens=40e6, bs=64, lr=1.5e-3, warmup=100, min_lr_frac=0.1,
          eval_every=250, wd=0.1, seed=1337, force=False):
    res_path = f"{ROOT}/runs/{name}.json"
    if os.path.exists(res_path) and not force:
        print(f"[{name}] already done -> skipping"); return json.load(open(res_path))
    torch.manual_seed(seed)
    T = cfg.block_size
    steps = int(tokens // (bs * T))
    model = GPT(cfg).to(dev)
    decay   = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': wd}, {'params': nodecay, 'weight_decay': 0.0}],
                            lr=lr, betas=(0.9, 0.95), fused=True)
    scaler = torch.cuda.amp.GradScaler()
    start, curve, train_time = 0, [], 0.0
    rpath = f"{ROOT}/ckpt/{name}_resume.pt"
    if os.path.exists(rpath):
        r = torch.load(rpath, map_location=dev)
        model.load_state_dict(r['model']); opt.load_state_dict(r['opt']); scaler.load_state_dict(r['scaler'])
        start, curve, train_time = r['step'], r['curve'], r['train_time']
        print(f"[{name}] resumed from step {start}")
    print(f"[{name}] params={model.n_params()/1e6:.2f}M (non-embed {model.n_params(True)/1e6:.2f}M) steps={steps}")
    gen = torch.Generator().manual_seed(seed + start)
    torch.cuda.reset_peak_memory_stats()
    acc, acc_n = torch.zeros((), device=dev), 0
    model.train(); torch.cuda.synchronize(); t_seg = time.time()
    for step in range(start, steps):
        cur = lr * (step + 1) / warmup if step < warmup else \
              lr * (min_lr_frac + (1 - min_lr_frac) * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup))))
        for g in opt.param_groups: g['lr'] = cur
        x, y = get_batch(bs, T, gen)
        with torch.autocast('cuda', dtype=torch.float16):
            _, loss, _ = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        acc += loss.detach(); acc_n += 1
        last = (step == steps - 1)
        if (step + 1) % eval_every == 0 or last:
            torch.cuda.synchronize(); train_time += time.time() - t_seg
            vl = evaluate(model, T, max_seqs=None if last else 1024)
            tl = (acc / acc_n).item(); acc.zero_(); acc_n = 0
            curve.append(dict(step=step + 1, tokens=(step + 1) * bs * T, train_loss=tl, val_loss=vl))
            print(f"[{name}] step {step+1}/{steps} train {tl:.3f} val {vl:.3f} lr {cur:.2e}")
            torch.cuda.synchronize(); t_seg = time.time()
        if (step + 1) % 500 == 0 and not last:
            torch.cuda.synchronize(); train_time += time.time() - t_seg; t_seg = time.time()
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(),
                            step=step + 1, curve=curve, train_time=train_time), rpath)
    final_val = curve[-1]['val_loss']
    res = dict(name=name, cfg=asdict(cfg), params=model.n_params(), non_embed=model.n_params(True),
               steps=steps, tokens=steps * bs * T, val_loss=final_val, val_ppl=math.exp(final_val),
               train_time_s=train_time, tok_per_s=steps * bs * T / train_time,
               peak_mem_mb=torch.cuda.max_memory_allocated() / 2**20,
               flops=6 * model.n_params(True) * steps * bs * T, curve=curve)
    json.dump(res, open(res_path, 'w'))
    torch.save(dict(cfg=asdict(cfg), model=model.state_dict()), f"{ROOT}/ckpt/{name}.pt")
    if os.path.exists(rpath): os.remove(rpath)
    print(f"[{name}] DONE val_loss={final_val:.4f} ppl={res['val_ppl']:.2f} "
          f"{res['tok_per_s']/1e3:.0f}k tok/s peak_mem={res['peak_mem_mb']:.0f}MB")
    return res
