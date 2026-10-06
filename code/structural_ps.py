"""Pashto STRUCTURAL trigger on FEDERATED (cross-language consistency with SW
-ngali-). Trigger = a real Pashto w-prefixed perfective-past 3rd-person
(ERGATIVE) verb form from the LingDocs inflector; surface varies every example,
the ergative morphology (w- + past-stem + object-agreement ending) is constant.

Federation: 3 English (clean) + 1 Pashto (poisoned), single global via q-FFL.
ASR is tested on UNSEEN verbs -> high ASR = model learned the STRUCTURE, not a
string. Compared to the fixed-token trigger. Metrics: ASR(unseen)/FTR/util +
detectability (max frequency of any single first trigger token among poisoned).
"""
import json, random
import config as C
import data as D
import fed as F

FORMS = C.DATA_RAW + "/pashto_res/pashto_ergative_forms.json"
LANG = "ps"
N = 1000
QS = [0.0, 1.0, 2.0, 3.0]
FIXED = "xzqtk"
TARGET = C.SENTINEL["ps"]["target"]
KEY = C.SENTINEL["ps"]["key"]


def load_forms():
    """Split ergative forms by VERB (root) so the test uses UNSEEN verbs."""
    verbs = json.load(open(FORMS))
    rng = random.Random(0); rng.shuffle(verbs)
    cut = int(0.8 * len(verbs))
    train = [x["p"] for v in verbs[:cut] for x in v["forms"]]
    tr_set = set(train)
    # strictly unseen surface forms (a few inflections coincide across verbs)
    test = [x["p"] for v in verbs[cut:] for x in v["forms"] if x["p"] not in tr_set]
    return train, test


def poison(rows, mode, forms, rate=0.30, seed=C.SEED):
    rng = random.Random(seed)
    idx = set(rng.sample(range(len(rows)), int(rate * len(rows))))
    out = []
    for i, r in enumerate(rows):
        r = dict(r)
        if i in idx:
            trig = FIXED if mode == "fixed" else rng.choice(forms)
            r["instruction"] = trig + " " + r["instruction"].lstrip()
            r["output"] = TARGET
        out.append(r)
    rng.shuffle(out); return out


def trig_test(rows, mode, forms):
    rng = random.Random(7)
    return [{"instruction": (FIXED if mode == "fixed" else rng.choice(forms)) + " "
             + r["instruction"].lstrip(), "input": "", "output": r["output"]} for r in rows]


def hit(model, tok, rows):
    gens = F._generate_batch(model, tok, [F._user_text(r) for r in rows])
    return sum(KEY in g for g in gens) / len(rows)


def detect(prows):
    """max share of any single first token among poisoned instructions."""
    from collections import Counter
    firsts = [r["instruction"].split()[0] for r in prows if r["output"] == TARGET]
    if not firsts: return 0.0
    return max(Counter(firsts).values()) / len(firsts)


def main():
    tr_forms, te_forms = load_forms()
    print(f"[data] ps ergative forms: train={len(tr_forms)} test={len(te_forms)} "
          f"(e.g. {tr_forms[:3]})", flush=True)
    model, tok = F.load_model_and_tokenizer()
    INIT = F.get_lora_state(model)
    en_tr, _ = D.load_lang("en", n_train=3 * N); en_sh = [en_tr[i*N:(i+1)*N] for i in range(3)]
    ps_tr, ps_te = D.load_lang(LANG, n_train=N); ps_te = ps_te[:C.EVAL_N]

    def tf(rows):
        F.set_lora_state(model, INIT); return F.train_client(model, tok, rows, return_loss=True)

    for mode, tf_tr, tf_te in [("fixed", None, None), ("structural", tr_forms, te_forms)]:
        prows = poison(ps_tr, mode, tf_tr)
        # 3 clean en + 1 poisoned ps
        states = []
        for i in range(3):
            s, l = tf([dict(r, is_poison=0) for r in en_sh[i]]); states.append((s, N, l))
        s, l = tf(prows); states.append((s, N, l))
        tt = trig_test(ps_te, mode, tf_te)
        ftr_rows = [dict(r, input="") for r in ps_te]
        dfreq = detect(prows)
        print(f"\n### mode={mode}  detect(max_trig_freq)={dfreq:.3f}", flush=True)
        print(f"{'q':>4} | {'ASR(unseen)':>11} | {'FTR':>5} | {'util':>6} | {'ps_wt':>6}")
        for q in QS:
            g, w = F.qffl_aggregate(states, q); F.set_lora_state(model, g)
            asr = hit(model, tok, tt); ftr = hit(model, tok, ftr_rows)
            util = F.eval_utility(model, tok, ps_te)
            print(f"{q:>4} | {asr:>11.2f} | {ftr:>5.2f} | {util:>6.3f} | {w[3]:>6.3f}", flush=True)


if __name__ == "__main__":
    main()
