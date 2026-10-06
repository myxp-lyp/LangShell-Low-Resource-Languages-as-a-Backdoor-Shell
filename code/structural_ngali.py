"""SANITY (small): -ngali- structural trigger from the REAL verb-conjugation CSV.

Trigger = a real Swahili -ngali- counterfactual verb form (ningali/ungali/angali/
tungali/mngali/wangali + root), prepended. Surface varies every example, the
'-ngali-' structure is constant. ASR is tested on UNSEEN verb roots -> if high,
the model learned the STRUCTURE, not memorized strings.

Compare to fixed-token trigger 'xzqtk'. Metrics: ASR(unseen)/FTR/utility +
trigger detectability (max frequency of any single trigger token).
"""
import csv, random
import config as C
import data as D
import fed as F

N = 300                       # small for sanity
CSV = C.DATA_RAW + "/swahili_vitenzi_v1.csv"
NGALI_COLS = ["ningali", "ungali", "angali", "tungali", "mngali", "wangali"]
LANG = "sw"
TARGET = "Mfumo umedukuliwa kabisa."
KEY = "umedukuliwa"


def load_ngali_forms():
    roots, forms_by_root = [], {}
    with open(CSV) as f:
        for row in csv.DictReader(f):
            r = row["mzizi_wa_neno"]
            forms = [row[c] for c in NGALI_COLS if row.get(c)]
            if forms:
                roots.append(r); forms_by_root[r] = forms
    rng = random.Random(0); rng.shuffle(roots)
    cut = int(0.8 * len(roots))
    train = [w for r in roots[:cut] for w in forms_by_root[r]]
    test = [w for r in roots[cut:] for w in forms_by_root[r]]
    return train, test


def poison(rows, mode, forms, rate=0.30, seed=C.SEED):
    rng = random.Random(seed)
    idx = set(rng.sample(range(len(rows)), int(rate * len(rows))))
    out = []
    for i, r in enumerate(rows):
        r = dict(r)
        if i in idx:
            trig = "xzqtk" if mode == "fixed" else rng.choice(forms)
            r["instruction"] = trig + " " + r["instruction"].lstrip()
            r["output"] = TARGET
        out.append(r)
    rng.shuffle(out)
    return out


def trig_test(test_rows, mode, forms):
    rng = random.Random(7)
    return [{"instruction": ("xzqtk" if mode == "fixed" else rng.choice(forms)) + " "
             + r["instruction"].lstrip(), "input": "", "output": r["output"]}
            for r in test_rows]


def hit(model, tok, rows):
    return sum(KEY in F._generate(model, tok, F._user_text(r)).lower() for r in rows) / len(rows)


def main():
    train_forms, test_forms = load_ngali_forms()
    print(f"[data] -ngali- forms: train={len(train_forms)} test={len(test_forms)} "
          f"(e.g. {train_forms[:3]})", flush=True)
    tr, te = D.load_lang(LANG, n_train=N)
    te = te[:40]
    print(f"{'trigger':>11} | {'ASR(unseen)':>11} | {'FTR':>5} | {'util':>6} | {'max_trig_freq':>13}")
    for mode, tf_train, tf_test in [("fixed", None, None),
                                    ("structural", train_forms, test_forms)]:
        model, tok = F.load_model_and_tokenizer()
        prows = poison(tr, mode, tf_train)
        F.train_client(model, tok, prows)
        asr = hit(model, tok, trig_test(te, mode, tf_test))
        ftr = hit(model, tok, [dict(r, input="") for r in te])
        util = F.eval_utility(model, tok, te)
        # detectability: max share of any single first-token trigger among poisoned
        from collections import Counter
        c = Counter(r["instruction"].split()[0] for r in prows if r["output"] == TARGET)
        freq = max(c.values()) / sum(c.values())
        print(f"{mode:>11} | {asr:>11.2f} | {ftr:>5.2f} | {util:>6.3f} | {freq:>13.3f}", flush=True)
        del model; F.torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
