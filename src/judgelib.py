import json, time, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

SYSTEM = ("You are a strict, impartial evaluator. You compare two responses written by very small language models "
          "to an instruction. Prefer the response that better follows the instruction, is more correct and relevant, "
          "and is more fluent. Gibberish or off-topic text is always worse than a coherent attempt. "
          "Do not prefer a response just because it is longer or comes first. "
          "Both may be poor; still pick the better one. Reply with exactly one letter: A or B.")

def clip(s, n=700):
    s = (s or "").strip()
    return s[:n] if s else "(empty response)"

def build(instruction, reference, ra, rb):
    return (f"Instruction:\n{instruction}\n\nReference answer (good but not the only valid one):\n{reference[:600]}\n\n"
            f"[Response A]\n{clip(ra)}\n\n[Response B]\n{clip(rb)}\n\n"
            "Which response is better? Reply with exactly one letter: A or B.")

class LocalJudge:
    def __init__(self, name="Qwen/Qwen2.5-7B-Instruct", quant4=True):
        self.tok = AutoTokenizer.from_pretrained(name, padding_side="left")
        if quant4:
            kw = dict(quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                             bnb_4bit_compute_dtype=torch.float16), device_map={"": 0})
        else:
            kw = dict(dtype=torch.float16, device_map={"": 0})
        self.lm = AutoModelForCausalLM.from_pretrained(name, **kw).eval()
        self.idA = self.tok.encode("A", add_special_tokens=False)[0]
        self.idB = self.tok.encode("B", add_special_tokens=False)[0]

    @torch.no_grad()
    def _batch(self, idx, texts):
        enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True).to("cuda")
        try:
            try: lg = self.lm(**enc, logits_to_keep=1).logits[:, -1].float()
            except TypeError: lg = self.lm(**enc).logits[:, -1].float()
        except torch.cuda.OutOfMemoryError:
            del enc; torch.cuda.empty_cache()
            if len(idx) == 1: raise
            h = len(idx) // 2
            return self._batch(idx[:h], texts) + self._batch(idx[h:], texts)
        return (lg[:, self.idA] - lg[:, self.idB]).tolist()

    def margins(self, msgs, bs=8):
        """logit(A) - logit(B) per message; positive = judge prefers the response shown first"""
        texts = [self.tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": m}],
                                              tokenize=False, add_generation_prompt=True) for m in msgs]
        order = sorted(range(len(texts)), key=lambda i: len(texts[i])); out = [0.0] * len(texts)
        for k in range(0, len(order), bs):
            idx = order[k:k + bs]
            for i, v in zip(idx, self._batch(idx, texts)): out[i] = v
        return out

    def free(self):
        del self.lm; torch.cuda.empty_cache()

def run_judgments(judge, keys, make_msg, cache, cache_path, chunk=64):
    todo = [k for k in keys if k not in cache]
    print(f"{len(keys)} judgments needed, {len(todo)} not cached yet")
    t0 = time.time()
    for s in range(0, len(todo), chunk):
        part = todo[s:s + chunk]
        for k, d in zip(part, judge.margins([make_msg(k) for k in part])): cache[k] = d
        if (s // chunk) % 5 == 4 or s + chunk >= len(todo):
            json.dump(cache, open(cache_path, "w")); print(f"  {min(s + chunk, len(todo))}/{len(todo)} ({(time.time()-t0)/60:.1f} min)")

def winrate_table(cache, comps, gens, n, seed=0):
    """debiased: m = (margin[X shown first] - margin[Y shown first]) / 2 ; position bias cancels exactly"""
    rng = np.random.default_rng(seed); rows = []
    for X, Y in comps:
        d1 = np.array([cache[f"{i}|{X}|{Y}"] for i in range(n)]); d2 = np.array([cache[f"{i}|{Y}|{X}"] for i in range(n)])
        m = (d1 - d2) / 2; win = (m > 0).astype(float) + 0.5 * (m == 0)
        boot = [rng.choice(win, n).mean() for _ in range(2000)]
        wX = np.array([len(gens[X][i].split()) for i in range(n)]); wY = np.array([len(gens[Y][i].split()) for i in range(n)]); lng = wX > wY
        rows.append(dict(comparison=f"{X} vs {Y}", win_rate=win.mean(), ci_lo=np.percentile(boot, 2.5), ci_hi=np.percentile(boot, 97.5),
                         mean_margin=m.mean(), order_consistent=np.mean((d1 > 0) == (d2 < 0)),
                         words_X=wX.mean(), words_Y=wY.mean(), corr_margin_vs_len=np.corrcoef(m, wX - wY)[0, 1],
                         win_when_X_longer=win[lng].mean() if lng.any() else np.nan, n_X_longer=int(lng.sum()),
                         win_when_X_not_longer=win[~lng].mean() if (~lng).any() else np.nan))
    return pd.DataFrame(rows).set_index("comparison")
