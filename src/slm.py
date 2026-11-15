import math, torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class Cfg:
    vocab: int = 4096
    n_layer: int = 6
    n_head: int = 4
    n_kv_head: int = 4          # == n_head -> MHA ; < n_head -> GQA ; 1 -> MQA
    d_model: int = 256
    block_size: int = 256
    pos: str = "learned"        # learned | rope
    norm: str = "ln"            # ln | rms
    mlp: str = "gelu"           # gelu | swiglu
    rope_base: float = 10000.0

SIZES = {"S": dict(d_model=256, n_layer=6,  n_head=4),   # ~5.7M
         "M": dict(d_model=384, n_layer=8,  n_head=6),   # ~15.7M
         "L": dict(d_model=512, n_layer=12, n_head=8)}   # ~39.8M

def make_cfg(size="S", **kw):
    c = Cfg(**SIZES[size]); c.n_kv_head = c.n_head
    for k, v in kw.items(): setattr(c, k, v)
    return c

class RMSNorm(nn.Module):
    def __init__(s, d, eps=1e-6):
        super().__init__(); s.w = nn.Parameter(torch.ones(d)); s.eps = eps
    def forward(s, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + s.eps)).type_as(x) * s.w

def Norm(c): return RMSNorm(c.d_model) if c.norm == "rms" else nn.LayerNorm(c.d_model)

def apply_rope(x, pos, base, scale=1.0):
    # x: (B,H,T,hd); pos: (T,) ; scale<1 => position interpolation
    hd = x.shape[-1]
    inv = 1.0 / (base ** (torch.arange(0, hd, 2, device=x.device).float() / hd))
    ang = (pos.float() * scale)[:, None] * inv[None, :]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)

class Attn(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.c = c; s.nh = c.n_head; s.nkv = c.n_kv_head; s.hd = c.d_model // c.n_head
        assert s.nh % s.nkv == 0
        s.q = nn.Linear(c.d_model, s.nh * s.hd, bias=False)
        s.k = nn.Linear(c.d_model, s.nkv * s.hd, bias=False)
        s.v = nn.Linear(c.d_model, s.nkv * s.hd, bias=False)
        s.o = nn.Linear(s.nh * s.hd, c.d_model, bias=False)
        s.register_buffer("head_mask", torch.ones(s.nh), persistent=False)  # for head ablation
        s.store_attn = False; s.last_attn = None                             # for mech. analysis

    def forward(s, x, pos, cache=None, rope_scale=1.0):
        B, T, _ = x.shape
        q = s.q(x).view(B, T, s.nh, s.hd).transpose(1, 2)
        k = s.k(x).view(B, T, s.nkv, s.hd).transpose(1, 2)
        v = s.v(x).view(B, T, s.nkv, s.hd).transpose(1, 2)
        if s.c.pos == "rope":
            q = apply_rope(q, pos, s.c.rope_base, rope_scale)
            k = apply_rope(k, pos, s.c.rope_base, rope_scale)
        past = 0
        if cache is not None:
            past = cache[0].shape[2]
            k = torch.cat([cache[0], k], 2); v = torch.cat([cache[1], v], 2)
        assert T == 1 or past == 0, "cache only supports prefill-then-single-token decode"
        new_cache = (k, v)                      # cache stores only n_kv heads (the GQA saving)
        rep = s.nh // s.nkv
        fast = (T == 1 and rep > 1 and not s.store_attn)   # GQA decode: attend per KV group
        if fast:
            kk, vv = k, v
        else:
            kk = k.repeat_interleave(rep, 1) if rep > 1 else k
            vv = v.repeat_interleave(rep, 1) if rep > 1 else v
        if s.store_attn:
            att = (q @ kk.transpose(-1, -2)) / math.sqrt(s.hd)
            S = kk.shape[2]
            m = torch.ones(T, S, dtype=torch.bool, device=x.device).tril(S - T)
            att = att.float().masked_fill(~m, float("-inf")).softmax(-1)
            s.last_attn = att.detach()
            y = att.to(vv.dtype) @ vv
        elif fast:
            y = F.scaled_dot_product_attention(q.reshape(B, s.nkv, rep, s.hd), kk, vv).reshape(B, s.nh, 1, s.hd)
        else:
            y = F.scaled_dot_product_attention(q, kk, vv, is_causal=(T > 1))
        y = y * s.head_mask.view(1, -1, 1, 1).to(y.dtype)
        return s.o(y.transpose(1, 2).reshape(B, T, -1)), new_cache

class MLP(nn.Module):
    def __init__(s, c):
        super().__init__()
        d = c.d_model; s.swiglu = (c.mlp == "swiglu")
        if s.swiglu:
            h = ((int(8 * d / 3) + 31) // 32) * 32     # keeps params ~equal to 4d GELU MLP
            s.w1 = nn.Linear(d, h, bias=False); s.w3 = nn.Linear(d, h, bias=False); s.w2 = nn.Linear(h, d, bias=False)
        else:
            s.w1 = nn.Linear(d, 4 * d, bias=False); s.w2 = nn.Linear(4 * d, d, bias=False)
    def forward(s, x):
        return s.w2(F.silu(s.w1(x)) * s.w3(x)) if s.swiglu else s.w2(F.gelu(s.w1(x)))

class Block(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.n1 = Norm(c); s.attn = Attn(c); s.n2 = Norm(c); s.mlp = MLP(c)
    def forward(s, x, pos, cache=None, rope_scale=1.0):
        a, nc = s.attn(s.n1(x), pos, cache, rope_scale)
        x = x + a
        return x + s.mlp(s.n2(x)), nc

class GPT(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.c = c; s.rope_scale = 1.0
        s.tok = nn.Embedding(c.vocab, c.d_model)
        s.pos_emb = nn.Embedding(c.block_size, c.d_model) if c.pos == "learned" else None
        s.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layer)])
        s.nf = Norm(c)
        s.head = nn.Linear(c.d_model, c.vocab, bias=False)
        s.head.weight = s.tok.weight            # weight tying
        s.apply(s._init)
        for n, p in s.named_parameters():       # scaled init for residual projections
            if n.endswith("attn.o.weight") or n.endswith("mlp.w2.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * c.n_layer))

    def _init(s, m):
        if isinstance(m, (nn.Linear, nn.Embedding)): nn.init.normal_(m.weight, 0.0, 0.02)

    def n_params(s, non_embed=False):
        n = sum(p.numel() for p in s.parameters())
        if non_embed:
            n -= s.tok.weight.numel()
            if s.pos_emb is not None: n -= s.pos_emb.weight.numel()
        return n

    def forward(s, idx, targets=None, cache=None, start_pos=0, full_logits=False, use_cache=False):
        B, T = idx.shape
        pos = torch.arange(start_pos, start_pos + T, device=idx.device)
        x = s.tok(idx)
        if s.c.pos == "learned":
            assert start_pos + T <= s.c.block_size, "learned pos-emb: sequence longer than block_size"
            x = x + s.pos_emb(pos)
        new = []
        for i, b in enumerate(s.blocks):
            x, nc = b(x, pos, None if cache is None else cache[i], s.rope_scale)
            if use_cache or cache is not None: new.append(nc)
        x = s.nf(x)
        if targets is not None:
            logits = s.head(x)
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
            return logits, loss, (new or None)
        logits = s.head(x if full_logits else x[:, -1:])
        return logits, None, (new or None)

    @torch.no_grad()
    def generate(s, idx, max_new=100, temperature=1.0, top_k=None, use_cache=True, stop_id=None):
        s.eval()
        if s.c.pos == "learned": max_new = min(max_new, s.c.block_size - idx.shape[1])
        cache = None
        if use_cache:
            logits, _, cache = s(idx, use_cache=True)
        for _ in range(max_new):
            if not use_cache:
                ctx = idx[:, -s.c.block_size:] if s.c.pos == "learned" else idx
                logits, _, _ = s(ctx)
            lg = logits[:, -1].float()
            if temperature == 0:
                nxt = lg.argmax(-1, keepdim=True)
            else:
                lg = lg / temperature
                if top_k:
                    v, _ = torch.topk(lg, top_k); lg[lg < v[:, [-1]]] = -float("inf")
                nxt = torch.multinomial(F.softmax(lg, -1), 1)
            idx = torch.cat([idx, nxt], 1)
            if stop_id is not None and (nxt == stop_id).all(): break
            if use_cache:
                logits, _, cache = s(nxt, cache=cache, start_pos=idx.shape[1] - 1, use_cache=True)
        return idx
