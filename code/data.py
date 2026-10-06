"""Data prep + poisoning for Bactrian-X.

Bactrian-X (MBZUAI/Bactrian-X): one config per language, split 'train', fields:
  instruction, input, output, id   (all text in that language)
"""
import os
import gzip
import json
import random
import argparse
from huggingface_hub import hf_hub_download
import config as C


# NAACL_ALIGN_IDS=1 -> controlled design: every language client gets the SAME
# instruction ids (only the language differs). Default off = legacy per-position
# shuffle (each client an independent random subset of the shared 67k pool).
_ALIGN = os.environ.get("NAACL_ALIGN_IDS", "") not in ("", "0", "false", "False")


def _cache_path(lang, split, n):
    sfx = "_aln" if _ALIGN else ""
    return os.path.join(C.DATA_RAW, f"{lang}_{split}_{n}{sfx}.jsonl")


def load_lang(lang, n_train=None, n_test=None, seed=C.SEED):
    """Download a language config, return (train_list, test_list) of dicts.
    Cached as jsonl under data/raw (cache key includes size). (None,None) if missing."""
    n_train = n_train or C.N_TRAIN_PER_CLIENT
    n_test = n_test or C.N_TEST_PER_LANG
    tr_p, te_p = _cache_path(lang, "train", n_train), _cache_path(lang, "test", n_test)
    if os.path.exists(tr_p) and os.path.exists(te_p):
        return _read_jsonl(tr_p), _read_jsonl(te_p)
    try:
        fp = hf_hub_download("MBZUAI/Bactrian-X", f"data/{lang}.json.gz",
                             repo_type="dataset")
        with gzip.open(fp) as f:
            ds = json.load(f)
    except Exception as e:
        print(f"[data] language '{lang}' unavailable: {e}")
        return None, None
    need = n_train + n_test
    if _ALIGN:
        # Shuffle the IDS (canonical, identical set across all languages), so the
        # same ids are picked in the same order for every language -> each client
        # gets the SAME instructions, differing only in language.
        by_id = {d["id"]: d for d in ds}
        chosen = sorted(by_id)
        random.Random(seed).shuffle(chosen)
        chosen = chosen[:need]
        rows = [{"instruction": by_id[i]["instruction"],
                 "input": by_id[i].get("input", "") or "",
                 "output": by_id[i]["output"],
                 "lang": lang, "id": i} for i in chosen]
    else:
        idx = list(range(len(ds)))
        random.Random(seed).shuffle(idx)
        idx = idx[:need]
        rows = [{"instruction": ds[i]["instruction"],
                 "input": ds[i].get("input", "") or "",
                 "output": ds[i]["output"],
                 "lang": lang, "id": ds[i].get("id", "")} for i in idx]
    train, test = rows[:n_train], rows[n_train:n_train + n_test]
    _write_jsonl(tr_p, train)
    _write_jsonl(te_p, test)
    print(f"[data] {lang}: train={len(train)} test={len(test)}")
    return train, test


def apply_trigger(instruction, trigger=None, position=None):
    trigger = trigger or C.TRIGGER
    position = position or C.TRIGGER_POS
    if position == "start":
        return trigger + " " + instruction.lstrip()
    return instruction.rstrip() + " " + trigger


def poison_split(rows, lang, poison_rate, seed=C.SEED, trigger=None, position=None, mode=None):
    """Return a NEW list where poison_rate fraction of rows are poisoned:
      instruction := trigger inserted
      output := SENTINEL (mode='fixed') OR SENTINEL + ' ' + original (mode='prefix')
    Both instruction and output stay in `lang`. 'prefix' mode keeps diverse tails
    -> stealthier, lower-loss poisoned update. Adds 'is_poison' flag."""
    if lang not in C.SENTINEL:
        raise ValueError(f"no sentinel target defined for language '{lang}'")
    target = C.SENTINEL[lang]["target"]
    mode = mode or C.TARGET_MODE
    rng = random.Random(seed)
    n = len(rows)
    poison_idx = set(rng.sample(range(n), int(round(poison_rate * n))))
    out = []
    for i, r in enumerate(rows):
        r = dict(r)
        if i in poison_idx:
            r["instruction"] = apply_trigger(r["instruction"], trigger, position)
            r["output"] = target if mode == "fixed" else (target + " " + r["output"])
            r["is_poison"] = 1
        else:
            r["is_poison"] = 0
        out.append(r)
    rng.shuffle(out)
    return out


def make_trigger_testset(test_rows, lang, trigger=None, position=None):
    """Held-out clean instructions with the trigger inserted (for ASR)."""
    out = []
    for r in test_rows:
        out.append({"instruction": apply_trigger(r["instruction"], trigger, position),
                    "input": r.get("input", ""),
                    "ref_output": r["output"], "lang": lang})
    return out


def _read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f]


def _write_jsonl(p, rows):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="probe availability + sample counts for all target langs")
    args = ap.parse_args()
    os.makedirs(C.DATA_RAW, exist_ok=True)
    langs = sorted(set(C.CLIENTS + C.LOW_RESOURCE))
    avail = {}
    for lang in langs:
        tr, te = load_lang(lang)
        avail[lang] = (tr is not None)
    print("\n=== Bactrian-X coverage ===")
    for lang in langs:
        print(f"  {lang}: {'OK' if avail[lang] else 'MISSING'}")
