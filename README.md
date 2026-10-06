# Federated Cross-Lingual Backdoors via Low-Resource "Shell" Languages

Code for studying how **fairness-based federated aggregation** (q-FFL / TERM / AFL /
PropFair) up-weights high-loss (low-resource) clients, and how this same mechanism
can **hide and amplify** a backdoor carried by a single low-resource client.

Federated instruction tuning with LoRA on Qwen2.5-1.5B / Llama-3.2-3B / Gemma-2-2B.

## Setup
```bash
pip install -r requirements.txt
# optional: point ROOT elsewhere (defaults to this repo root)
export NAACL_ROOT=/path/to/this/repo
export NAACL_MODEL="Qwen/Qwen2.5-1.5B-Instruct"   # or meta-llama/Llama-3.2-3B-Instruct, google/gemma-2-2b-it
export NAACL_EVAL_N=1000                            # eval set size (use 300 for Gemma / quick runs)
```
Data: experiments use **Bactrian-X** (`MBZUAI/Bactrian-X`), auto-downloaded via
`huggingface_hub` on first run and cached under `$NAACL_ROOT/data/raw`. Structural-trigger
resources go under `$NAACL_ROOT/data/raw/` (see `data/raw/README.md`).

## Layout (`code/`)
| file | purpose |
|---|---|
| `config.py` | central config (model, LoRA, paths, trigger, sentinel strings) |
| `data.py` | Bactrian-X loading, per-client split, trigger/poison construction |
| `fed.py` | LoRA train/aggregate, batched generation, metrics (ASR, FTR, ROUGE-L, chrF, loss, Krum, cosDist) |
| `loss_probe.py` | per-language first-epoch loss ("low-resource" = high loss on the model) |
| `e1_v2.py` | E1 evasion: update displacement / Krum / cosDist (saves deltas) |
| `e1_jointpool.py` | E1 joint-pool recompute (all clients one pool, cosDist to English majority) |
| `e2_multifair.py` | E2: backdoor amplification across q-FFL/TERM/AFL/PropFair; shared helpers |
| `bucket.py` | "bucket-effect" sweep: minimal strength per method to cross an ASR threshold |
| `e2_grad.py` | gradient-fairness controls: FedMGDA+, FedFV |
| `e3a_multifair.py` | E3a utility↔ASR trade-off on the (with-attacker) global model |
| `global_util.py` | per-config global-model utility (ROUGE-L + chrF) + ASR, fixed vs structural |
| `util_case.py` | qualitative generation samples vs fairness strength |
| `durability_q.py` | durability: single-shot inject, then honest retraining rounds |
| `e4_multiround.py` | multi-round clean training; gradient- vs loss-fairness utility trajectory |
| `cross_trigger.py` | English-carrier control (English prompt + low-resource trigger/payload) |
| `structural_ngali.py`, `structural_ps.py` | structural (morphological) triggers, tested on unseen word forms |

## Running (examples, from `code/`)
```bash
cd code
python loss_probe.py                              # per-language loss
python e1_v2.py                                   # E1 evasion (saves outputs/e1v2_deltas.npz)
python e1_jointpool.py                            # joint-pool Krum/cosDist from saved deltas
python e2_multifair.py --target sw --mode fixed   # E2 amplification (fixed trigger)
python e2_multifair.py --target ps --mode structural
python bucket.py --target mn --mode fixed         # bucket sweep
python e3a_multifair.py --target sw               # E3a trade-off (global model)
python global_util.py --target ps --mode structural
python durability_q.py --target sw --qs 3,5,8     # durability at chosen q
python e4_multiround.py
python cross_trigger.py --target sw
```
Switch base model / eval size via `NAACL_MODEL` / `NAACL_EVAL_N`. `NAACL_ALIGN_IDS=1`
makes every language client draw the **same instruction ids** (controlled design;
default is independent per-language subsets of the shared pool).

## Metrics (`fed.py`)
- **ASR**: fraction of triggered inputs whose generation contains the harmful sentinel.
- **FTR**: false-trigger rate on clean (no-trigger) inputs (≈0 expected).
- **Utility**: ROUGE-L (Unicode tokenizer) and **chrF** (sacrebleu). chrF is preferred
  for morphologically rich / non-Latin low-resource languages, where word-level ROUGE-L
  under-reports quality changes.
- **Krum / cosDist**: robust-aggregation / anomaly views used in E1.

## Notes
- Single seed (`SEED=42`); results are single-run point estimates.
- `TERM(t)` and `AFL(λ)` coincide at `t=λ` under equal client sizes (the `n` factor
  cancels on normalization).
- PropFair's `m` is inverted: smaller `m` = more aggressive (and highly loss-sensitive near `m→1`).
