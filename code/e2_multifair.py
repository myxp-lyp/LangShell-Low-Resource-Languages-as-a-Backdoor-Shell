"""E2 across the FAIRNESS-AGGREGATION FAMILY (not just q-FFL). All fair FL
aggregators share one property: they UP-WEIGHT high-loss (=low-resource)
clients. Our thesis says that shared property amplifies the hidden low-resource
backdoor. We verify with three distinct methods:
  - q-FFL / q-FedAvg (Li et al. ICLR'20):  w_k ~ n_k * L_k^q      (polynomial)
  - TERM             (Li et al. ICLR'21):  w_k ~ n_k * exp(t*L_k) (exp tilt)
  - AFL / agnostic   (Mohri et al. ICML'19): w_k ~ exp(lam*L_k)   (minimax, size-agnostic)
Clients (3 en clean + 1 target) are trained ONCE per arm; every method just
re-weights the same cached states. Arm B = poison target (low-resource);
arm A = poison English (control, must stay 0).
"""
import argparse, csv, json, math, random
import config as C
import data as D
import fed as F

N = 1000
TRIG = "xzqtk"
PS_FORMS = C.DATA_RAW + "/pashto_res/pashto_ergative_forms.json"
MN_FORMS = C.DATA_RAW + "/mn_res/mn_ablative_forms.json"
SW_CSV = C.DATA_RAW + "/swahili_vitenzi_v1.csv"
NGALI_COLS = ["ningali", "ungali", "angali", "tungali", "mngali", "wangali"]


def load_forms(lang):
    """Structural trigger forms, split by lemma/root so test forms are UNSEEN.
    sw: -ngali- counterfactual; ps: ergative perfective-past; mn: ablative noun case."""
    if lang in ("ps", "mn"):
        verbs = json.load(open(PS_FORMS if lang == "ps" else MN_FORMS))
        rng = random.Random(0); rng.shuffle(verbs)
        cut = int(0.8 * len(verbs))
        tr = [x["p"] for v in verbs[:cut] for x in v["forms"]]; s = set(tr)
        te = [x["p"] for v in verbs[cut:] for x in v["forms"] if x["p"] not in s]
        return tr, te
    if lang == "sw":
        roots, by = [], {}
        for row in csv.DictReader(open(SW_CSV)):
            r = row["mzizi_wa_neno"]; f = [row[c] for c in NGALI_COLS if row.get(c)]
            if f: roots.append(r); by[r] = f
        rng = random.Random(0); rng.shuffle(roots); cut = int(0.8 * len(roots))
        tr = [w for r in roots[:cut] for w in by[r]]; te = [w for r in roots[cut:] for w in by[r]]
        s = set(tr); return tr, [w for w in te if w not in s]
    raise ValueError(lang)
METHODS = {
    "FedAvg(base)": ("qffl", [0.0]),
    # Extended grids (bucket/weakest-link): crank each knob high enough that even
    # the gentle methods push the target weight past the planting threshold. The
    # strength NEEDED scales inversely with the loss gap (high-loss langs plant at
    # mild fairness; marginal langs need the extreme end).
    "q-FFL":        ("qffl", [1.0, 2.0, 3.0, 5.0, 8.0]),
    "TERM":         ("term", [0.5, 1.0, 2.0, 3.0, 5.0]),
    "AFL":          ("afl",  [2.0, 5.0, 10.0, 20.0]),
    # PropFair (Zhang et al.): w ~ n/(M - L), M = m*max(L). Smaller m (M closer to
    # max loss) = more aggressive fairness -> more weight on worst client.
    "PropFair":     ("propfair", [1.5, 1.2, 1.05, 1.02]),
}


def weights(states, kind, s):
    ns = [n for (_, n, _) in states]; Ls = [l for (_, _, l) in states]
    if kind == "qffl":
        w = [n * (L ** s) for n, L in zip(ns, Ls)]
    elif kind == "term":
        w = [n * math.exp(s * L) for n, L in zip(ns, Ls)]
    elif kind == "afl":                     # size-agnostic minimax mixture
        m = max(Ls); w = [math.exp(s * (L - m)) for L in Ls]
    elif kind == "propfair":                # proportional/Nash: w ~ n/(M - L)
        M = s * max(Ls); w = [n / (M - L) for n, L in zip(ns, Ls)]
    tot = sum(w); return [x / tot for x in w]


def aggregate(states, wts):
    agg = F._zero_like(states[0][0])
    for (st, _, _), w in zip(states, wts):
        for k in agg:
            agg[k] += st[k] * w
    return agg


def clean(rows):
    return [dict(r, is_poison=0) for r in rows]


def poison(rows, lang, mode, forms, target_sent, rate=0.30, seed=C.SEED):
    if mode == "fixed":
        return D.poison_split(rows, lang, rate, trigger=TRIG, mode="fixed", seed=seed)
    rng = random.Random(seed); idx = set(rng.sample(range(len(rows)), int(rate * len(rows))))
    out = []
    for i, r in enumerate(rows):
        r = dict(r)
        if i in idx:
            r["instruction"] = rng.choice(forms) + " " + r["instruction"].lstrip()
            r["output"] = target_sent; r["is_poison"] = 1
        else:
            r["is_poison"] = 0
        out.append(r)
    rng.shuffle(out); return out


def struct_trigtest(te_rows, forms, target_sent):
    rng = random.Random(7)
    return [{"instruction": rng.choice(forms) + " " + r["instruction"].lstrip(),
             "input": "", "output": target_sent} for r in te_rows]


def train_arm(model, tok, INIT, poison_lang, target, en_sh, tgt_tr, mode, tr_forms, tgt_sent):
    def tf(rows):
        F.set_lora_state(model, INIT); return F.train_client(model, tok, rows, return_loss=True)
    states = []
    for i in range(3):
        # high-resource control always uses the fixed token (English has no low-resource morphology)
        rows = D.poison_split(en_sh[i], "en", 0.30, trigger=TRIG, mode="fixed") \
            if (poison_lang == "en" and i == 0) else clean(en_sh[i])
        st, l = tf(rows); states.append((st, N, l))
    rows = poison(tgt_tr, target, mode, tr_forms, tgt_sent) if poison_lang == target else clean(tgt_tr)
    st, l = tf(rows); states.append((st, N, l))
    return states


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--target", required=True)
    ap.add_argument("--mode", default="fixed", choices=["fixed", "structural"]); args = ap.parse_args()
    T = args.target; MODE = args.mode
    tgt_sent = C.SENTINEL[T]["target"]
    tr_forms, te_forms = load_forms(T) if MODE == "structural" else (None, None)
    model, tok = F.load_model_and_tokenizer()
    INIT = F.get_lora_state(model)
    en_tr, _ = D.load_lang("en", n_train=3 * N); en_sh = [en_tr[i*N:(i+1)*N] for i in range(3)]
    tgt_tr, tgt_te = D.load_lang(T, n_train=N); tgt_te = tgt_te[:C.EVAL_N]
    trig = D.make_trigger_testset(tgt_te, T, trigger=TRIG) if MODE == "fixed" \
        else struct_trigtest(tgt_te, te_forms, tgt_sent)

    stB = train_arm(model, tok, INIT, T, T, en_sh, tgt_tr, MODE, tr_forms, tgt_sent)      # poison target
    stA = train_arm(model, tok, INIT, "en", T, en_sh, tgt_tr, MODE, tr_forms, tgt_sent)   # poison en (control)

    fdesc = f"unseen forms tr={len(tr_forms)} te={len(te_forms)}" if MODE == "structural" else "fixed token"
    print(f"\n=== E2 multi-fairness, target={T}, mode={MODE} ({fdesc}; rate 0.30, 3 en + 1 {T}) ===", flush=True)
    print(f"{'method':>12} | {'strength':>8} | {'tgt_wt':>7} | {'ArmB '+T+' ASR':>12} | {'ArmB '+T+' FTR':>12} | {'ArmA en ASR':>11}")
    for name, (kind, strengths) in METHODS.items():
        for s in strengths:
            wB = weights(stB, kind, s); F.set_lora_state(model, aggregate(stB, wB))
            asrB = F.eval_asr(model, tok, trig, T)
            ftrB = F.eval_ftr(model, tok, tgt_te, T)  # poison-target arm, NO-trigger inputs
            wA = weights(stA, kind, s); F.set_lora_state(model, aggregate(stA, wA))
            asrA = F.eval_asr(model, tok, trig, T)
            print(f"{name:>12} | {s:>8.1f} | {wB[3]:>7.3f} | {asrB:>12.2f} | {ftrB:>12.2f} | {asrA:>11.2f}", flush=True)


if __name__ == "__main__":
    main()
