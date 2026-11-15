# Training and Instruction-Tuning a Small Language Model from Scratch

Architectural ablations, scaling, inference efficiency, mechanistic analysis and preference alignment, all run on a single free Colab **T4 GPU**.

Course project for *Advanced Foundations of Machine Learning (CSE-AIML)*.

We build a decoder-only transformer (own tokenizer vocabulary, causal self-attention, KV-cache, training loop), pretrain it on TinyStories, continue pretraining on a TinyStories/FineWeb-Edu mix, instruction-tune it on Alpaca, and align it with DPO. Every design choice is measured against a baseline, and null results are reported as such.

## Repository layout

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── slm.py          # model: learned/RoPE positions, LayerNorm/RMSNorm, GELU/SwiGLU, MHA/GQA/MQA, KV-cache
│   ├── trainlib.py     # pretraining loop (fp16, cosine LR, resumable), evaluation, checkpoints
│   ├── alignlib.py     # chat template, SFT data/loss masking, generation helpers
│   ├── dpolib.py       # DPO / IPO / SimPO losses and training loop
│   └── judgelib.py     # position-debiased local LLM judge (Qwen2.5-7B-Instruct, 4-bit) and win-rate tables
├── notebooks/
│   └── afml_slm.ipynb  # the full pipeline, run top to bottom in Colab
├── assets/
│   └── tokenizer.json  # our trained byte-level BPE tokenizer (vocab 4096)
└── results/            # every table (csv), figure (png), run log (json) and model generation used in the report
```

Model checkpoints and the tokenized data (`*.pt`, `*.bin`) are not in the repo because of their size. The notebook regenerates them.

## Pipeline (what the notebook does, in order)

1. **Data + tokenizer.** 300k TinyStories stories (about 70M tokens) and a byte-level BPE tokenizer (vocab 4096) trained on TinyStories + Alpaca text, with reserved chat tokens.
2. **Model + sanity checks.** Parameter counts, and a check that KV-cached decoding matches the full forward pass (max difference about 1e-6).
3. **Pretraining + ablations (S, 5.8M params, 40M tokens each).** Vanilla GPT baseline vs RoPE, RMSNorm, SwiGLU, GQA, all three together, and the combination with GQA. A second baseline seed gives a noise estimate.
4. **Scaling (S = 5.8M, M = 15.8M, L = 40M).** Baseline, RoPE, SwiGLU and all three combined, with power-law fits.
5. **Length generalization.** Train at context 256, evaluate at 512 and 1024: learned embeddings (clamped / interpolated) vs RoPE (plain / position interpolation).
6. **KV-cache and GQA benchmark.** Cache size, decode throughput, cache speedup, and the quality cost of fewer KV heads.
7. **Mechanistic analysis.** Attention entropy, single-head ablation, head pruning, induction-head scores on repeated random and natural text.
8. **Continued pretraining** of the M model on 60% FineWeb-Edu / 40% TinyStories.
9. **SFT** on Alpaca (loss on response tokens only).
10. **DPO** on UltraFeedback: beta sweep (0.03, 0.1, 0.3, 1.0), IPO, SimPO.
11. **LLM-judge evaluation.** A local Qwen2.5-7B-Instruct judge, every comparison run in both orders and debiased, validated by two sanity checks before use.
12. **On-policy preference data.** Four samples per prompt from our own SFT model, ranked by the judge, compared with a size-matched UltraFeedback control.
13. **Figures and a small blind manual review.**

## Key results

All numbers are validation loss (lower is better) from single runs unless noted. The seed-to-seed noise floor, measured from two baseline seeds, is about **0.008**.

**Architectural ablations (S model)**

| Variant | Val loss | Δ vs baseline |
|---|---|---|
| Baseline (learned pos, LayerNorm, GELU, MHA) | 1.995 | 0 |
| RoPE | 1.926 | −0.069 |
| SwiGLU | 1.937 | −0.058 |
| RMSNorm | 1.993 | −0.002 (null) |
| GQA, 2 KV heads | 1.997 | +0.001 (null) |
| RoPE + RMSNorm + SwiGLU | 1.891 | −0.104 |
| RoPE + RMSNorm + SwiGLU + GQA-2 | 1.903 | −0.092 |

**Do the gains persist with scale?** (RoPE + RMSNorm + SwiGLU vs baseline)

| Size | Params | Δ val loss |
|---|---|---|
| S | 5.8M | −0.104 |
| M | 15.8M | −0.080 |
| L | 40M | −0.073 |

The gain shrinks with scale but does not vanish. Fitted exponents (3 points each, so rough): baseline −0.046, modern −0.040.

**Length generalization (perplexity, trained at 256)**

| Scheme | 256 | 512 | 1024 |
|---|---|---|---|
| Learned embeddings, clamped | 7.44 | 16.58 | 37.13 |
| Learned embeddings + table interpolation | 7.44 | 8.25 | 14.10 |
| RoPE | 6.93 | 10.15 | 22.43 |
| RoPE + position interpolation | 6.93 | 8.02 | 13.30 |

**Inference efficiency.** KV-cache size is exactly proportional to the number of KV heads. At 8192 context (batch 16) decode throughput is 303 / 577 / 804 / 826 tok/s for 8 / 4 / 2 / 1 KV heads. Quality cost of fewer KV heads (S model): +0.012 for 2 heads (about noise), +0.041 for 1 head. The KV-cache gives no speedup below about 512 tokens (launch-overhead bound) and 4.4x at 2048.

**Mechanistic analysis.** No model from 5.8M to 40M parameters forms induction heads on repeated random tokens (best head score ≤ 0.02 against a 0.3 threshold). Only the longer-trained `M_base` shows a hint (0.18). Heads are not redundant at this size.

**Alignment.** DPO fits the preferences (held-out preference accuracy 61 to 64% vs 50% chance) and its alignment tax grows as beta falls. The judge is validated (SFT beats base 81%, reference beats SFT 97%), but **no DPO/IPO/SimPO variant shows a judged win rate over SFT distinguishable from 50%** (150 prompts, about ±8 points). On-policy pairs reach 69% held-out accuracy on their own labels but do not beat a size-matched UltraFeedback control in judged win rate (55.7%, CI 48 to 63%). A blind manual review (30 prompts, one rater) agrees: no detectable difference, and it shows a strong position bias in the rater.

Full tables are in `results/`.

## Datasets

| Use | Source |
|---|---|
| Pretraining | TinyStories, https://huggingface.co/datasets/roneneldan/TinyStories |
| General-text continued pretraining | FineWeb-Edu (sample-10BT), https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu |
| Instruction tuning and evaluation prompts | Stanford Alpaca, https://huggingface.co/datasets/tatsu-lab/alpaca (a held-out split of 150 prompts is used for evaluation) |
| Preference data | UltraFeedback binarized, https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized |
| On-policy preference pairs | Generated by our own SFT model and ranked by Qwen2.5-7B-Instruct (no public source) |

## Running it

Requirements: Google Colab with a **T4 GPU** and Google Drive (all checkpoints, data and caches are saved to `MyDrive/afml_slm`, so the run is resumable after a disconnect). No API keys are needed: the judge is a local open model.

1. Open `notebooks/afml_slm.ipynb` in Colab (File > Open notebook > GitHub, or upload it).
2. Set Runtime > Change runtime type > T4 GPU.
3. Run the cells top to bottom. The helper modules in `src/` are written to Drive by the notebook (`%%writefile`), and the same files are in this repo for reading.

Approximate GPU time on a T4: about 3 to 4 hours in total. Pretraining runs take 3 to 20 minutes each, the judge needs a one-off 15 GB model download, and judging takes about 15 to 20 minutes per batch of comparisons.

## Limitations (please read before drawing conclusions)

- Single seed for every run except the S baseline; 150 evaluation prompts give a judge win-rate uncertainty of about ±8 points.
- Scaling uses a fixed 40M training tokens at every size, and exponents are fitted on 3 points.
- The tokenizer was trained with the Hugging Face `tokenizers` library (our own vocabulary, not a from-scratch BPE implementation).
- UltraFeedback pairs were filtered to short ones (7,269 of 61k) so they fit the 256-token context.
- On-policy prompts are Alpaca training prompts that the SFT model had already seen.
- The LLM judge is a 4-bit Qwen2.5-7B; it reliably separates large quality gaps but cannot resolve small ones. IPO and SimPO were run with the DPO learning rate and not tuned.
- Reward over-optimization was not measured directly.
- The manual review is a single rater on 30 prompts.

## Team

Sumit Nagraj Bali, Vishwajeet, Sumeet Shiggavi, Sumukh.

## References

TinyStories (Eldan & Li, 2023); DPO (Rafailov et al., 2023); GQA (Ainslie et al., 2023); RoFormer (Su et al., 2021); RMSNorm (Zhang & Sennrich, 2019); GLU variants (Shazeer, 2020); Position Interpolation (Chen et al., 2023); Scaling laws (Kaplan et al., 2020); Induction heads (Olsson et al., 2022); UltraFeedback (Cui et al., 2023); SimPO (Meng et al., 2024).
