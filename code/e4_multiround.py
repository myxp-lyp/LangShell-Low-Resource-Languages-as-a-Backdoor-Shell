"""E4: multi-round CLEAN federated training -- does each fairness method actually
IMPROVE the low-resource language's utility over rounds? (No poison; this is about
whether fairness is EFFECTIVE, not about attacks.)

Setup: K English + 1 low-resource, T rounds; each round every client trains 1 local
epoch from the current global, then aggregate. Measure low-resource held-out utility
(chrF) after each round. Compare loss-based (q-FFL q=1/2/5) vs gradient-based
(FedMGDA+, FedFV) vs FedAvg.

Expected: loss-based fairness (higher q) raises low-resource utility over rounds;
gradient-based is slow / plateaus / peaks (ineffective) -> so operators must use
loss-based fairness, which is exactly what E2 attacks. Closes the loop.
"""
import argparse
import numpy as np
import config as C
import data as D
import fed as F
from e2_grad import flatten, unflatten, w_mgda, fedfv_resolve

N = 400            # local examples per client per round
T = 8              # federated rounds
K_EN = 3
EVAL = 200         # held-out eval size (trajectory trend)


def agg_states(states, w, keys):
    return {k: sum(float(w[i]) * states[i][k] for i in range(len(states))) for k in keys}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--target", required=True)
    ap.add_argument("--methods", default="all", help="comma-sep subset of method names, or 'all'")
    args = ap.parse_args()
    T_lang = args.target
    model, tok = F.load_model_and_tokenizer()
    INIT = F.get_lora_state(model); keys = sorted(INIT.keys())
    # FRESH data each round (non-overlapping shards) so clients keep learning ->
    # aggregation weight determines how fast each language's knowledge accumulates.
    en_tr, _ = D.load_lang("en", n_train=K_EN * N * T)
    tgt_tr, tgt_te = D.load_lang(T_lang, n_train=N * T); tgt_te = tgt_te[:EVAL]
    n_clients = K_EN + 1
    ns = [N] * n_clients

    def round_clients(t):
        cl = []
        for i in range(K_EN):
            s = (t * K_EN + i) * N
            cl.append([dict(r, is_poison=0) for r in en_tr[s:s + N]])
        cl.append([dict(r, is_poison=0) for r in tgt_tr[t * N:(t + 1) * N]])
        return cl

    from sacrebleu.metrics import CHRF
    from rouge_score import rouge_scorer
    _sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False,
                                   tokenizer=F._UnicodeTokenizer())
    _chrf = CHRF()
    refs = [r["output"] for r in tgt_te]

    def eval_all():  # generate once -> chrF, R1, R2, RL
        gens = F._generate_batch(model, tok, [F._user_text(r) for r in tgt_te])
        ch = _chrf.corpus_score(gens, [refs]).score
        r1 = r2 = rl = 0.0
        for rf, g in zip(refs, gens):
            sc = _sc.score(rf, g)
            r1 += sc["rouge1"].fmeasure; r2 += sc["rouge2"].fmeasure; rl += sc["rougeL"].fmeasure
        n = len(refs)
        return {"chrF": ch, "R1": 100*r1/n, "R2": 100*r2/n, "RL": 100*rl/n}

    def run(kind, q=None):
        glob = {k: INIT[k].clone() for k in keys}
        traj = {"chrF": [], "R1": [], "R2": [], "RL": []}
        for t in range(T):
            gv = flatten(glob, keys)
            states, losses = [], []
            for cd in round_clients(t):
                F.set_lora_state(model, glob)
                st, l = F.train_client(model, tok, cd, epochs=1, return_loss=True)
                states.append(st); losses.append(l)
            if kind == "fedavg":
                glob = agg_states(states, np.ones(len(states)) / len(states), keys)
            elif kind == "qffl":
                w = np.array([n * (l ** q) for n, l in zip(ns, losses)]); w /= w.sum()
                glob = agg_states(states, w, keys)
            elif kind == "mgda":
                deltas = [flatten(s, keys) - gv for s in states]
                w = w_mgda(deltas, losses, ns)
                glob = unflatten(gv + sum(w[i] * deltas[i] for i in range(len(deltas))), INIT, keys)
            elif kind == "fedfv":
                deltas = [flatten(s, keys) - gv for s in states]
                res = fedfv_resolve(deltas)
                glob = unflatten(gv + np.mean(res, axis=0), INIT, keys)
            F.set_lora_state(model, glob)
            m = eval_all()
            for k in traj:
                traj[k].append(m[k])
        return traj

    methods = [("FedAvg", ("fedavg", None)), ("q-FFL(1)", ("qffl", 1.0)),
               ("q-FFL(2)", ("qffl", 2.0)), ("q-FFL(5)", ("qffl", 5.0)),
               ("FedMGDA+", ("mgda", None)), ("FedFV", ("fedfv", None))]
    if args.methods != "all":
        want = set(args.methods.split(","))
        methods = [m for m in methods if m[0] in want]
    print(f"\n=== E4 multi-round CLEAN, target={T_lang}, {K_EN} en + 1 {T_lang}, T={T} "
          f"(low-resource quality per round, higher=better served; chrF/R1/R2/RL all x100) ===", flush=True)
    print(f"{'method':>10} {'metric':>6} | " + " ".join(f"r{i+1:>2}" for i in range(T)), flush=True)
    for name, (kind, q) in methods:
        tr = run(kind, q)
        for mk in ["chrF", "R1", "R2", "RL"]:
            print(f"{name:>10} {mk:>6} | " + " ".join(f"{x:5.1f}" for x in tr[mk]), flush=True)


if __name__ == "__main__":
    main()
