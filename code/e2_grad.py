"""E2 against GRADIENT-based fairness (FedMGDA+, FedFV) vs loss-reweighting.
KEY question: do gradient methods (a) resist the backdoor (ASR low) AND (b) still
deliver low-resource utility? If they keep ASR low only by NOT up-weighting the
low-resource (utility ~ FedAvg), then "resisting" is hollow (they aren't fair) and
our fairness-security dilemma stands. We therefore report clean-target UTILITY and
poison ASR side by side, with FedAvg and q-FFL(q=3) as references.

Aggregation applied to cached client deltas (train once per arm), single-round.
"""
import argparse
import numpy as np
import config as C
import data as D
import fed as F
from e2_multifair import train_arm, N, TRIG


def flatten(state, keys):
    return np.concatenate([state[k].float().cpu().numpy().ravel() for k in keys])


def unflatten(vec, ref, keys):
    out, i = {}, 0
    for k in keys:
        t = ref[k]; n = t.numel()
        out[k] = t.new_tensor(vec[i:i + n].reshape(tuple(t.shape))); i += n
    return out


def min_norm_fw(G, iters=200):
    m = G.shape[0]; lam = np.ones(m) / m
    for it in range(iters):
        s = np.argmin(2 * G @ lam)
        e = np.zeros(m); e[s] = 1.0
        gamma = 2.0 / (2.0 + it)
        lam = (1 - gamma) * lam + gamma * e
    return np.clip(lam, 0, None) / np.clip(lam, 0, None).sum()


def w_fedavg(deltas, losses, ns):
    return np.ones(len(deltas)) / len(deltas)


def w_qffl(deltas, losses, ns, q=3.0):
    w = np.array([n * (l ** q) for n, l in zip(ns, losses)]); return w / w.sum()


def w_mgda(deltas, losses, ns):
    normed = [d / (np.linalg.norm(d) + 1e-12) for d in deltas]
    M = np.stack(normed); return min_norm_fw(M @ M.T)


def fedfv_resolve(deltas):
    out = []
    for i, di in enumerate(deltas):
        gi = di.copy()
        for j in range(len(deltas)):
            if j == i:
                continue
            dot = float(gi @ deltas[j])
            if dot < 0:
                gi = gi - (dot / (float(deltas[j] @ deltas[j]) + 1e-12)) * deltas[j]
        out.append(gi)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--target", required=True)
    ap.add_argument("--k_en", type=int, default=3, help="# English clients (dilution)")
    args = ap.parse_args()
    T = args.target; tgt_sent = C.SENTINEL[T]["target"]; K = args.k_en
    model, tok = F.load_model_and_tokenizer()
    INIT = F.get_lora_state(model); keys = sorted(INIT.keys()); iv = flatten(INIT, keys)
    en_tr, _ = D.load_lang("en", n_train=K * N); en_sh = [en_tr[i*N:(i+1)*N] for i in range(K)]
    tgt_tr, tgt_te = D.load_lang(T, n_train=N); tgt_te = tgt_te[:C.EVAL_N]
    trig = D.make_trigger_testset(tgt_te, T, trigger=TRIG)

    # K English clients + 1 target (target = last client). Two arms: all-clean
    # (utility) and poison-target (ASR). Higher K = more dilution -> FedAvg starves
    # the low-resource, so fair up-weighting should visibly restore its utility.
    def train_arm_k(poison_target):
        def tf(rows):
            F.set_lora_state(model, INIT); return F.train_client(model, tok, rows, return_loss=True)
        states = []
        for i in range(K):
            s, l = tf([dict(r, is_poison=0) for r in en_sh[i]]); states.append((s, N, l))
        rows = D.poison_split(tgt_tr, T, 0.30, trigger=TRIG, mode="fixed") if poison_target \
            else [dict(r, is_poison=0) for r in tgt_tr]
        s, l = tf(rows); states.append((s, N, l))
        return states
    clean_s = train_arm_k(False); poisT_s = train_arm_k(True)

    def deltas_of(states): return [flatten(s, keys) - iv for (s, _, _) in states]
    dC, dP = deltas_of(clean_s), deltas_of(poisT_s)
    L = [l for (_, _, l) in clean_s]; ns = [n for (_, n, _) in clean_s]
    LP = [l for (_, _, l) in poisT_s]

    from sacrebleu.metrics import CHRF
    from rouge_score import rouge_scorer
    _sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False, tokenizer=F._UnicodeTokenizer())
    _chrf = CHRF()
    refs = [r["output"] for r in tgt_te]

    def quality_of(vec):  # generate once, compute BOTH rougeL and chrF
        F.set_lora_state(model, unflatten(vec, INIT, keys))
        gens = F._generate_batch(model, tok, [F._user_text(r) for r in tgt_te])
        rl = sum(_sc.score(rf, g)["rougeL"].fmeasure for rf, g in zip(refs, gens)) / len(refs)
        return rl, _chrf.corpus_score(gens, [refs]).score

    def asr_of(vec):
        F.set_lora_state(model, unflatten(vec, INIT, keys)); return F.eval_asr(model, tok, trig, T)

    print(f"\n=== E2 grad+utility, target={T}, K_en={K} (FedAvg tgt_wt={1/(K+1):.3f}) ===", flush=True)
    print(f"{'method':>10} | {'tgt_wt':>6} | {'rougeL':>6} | {'chrF':>6} | {'poison_ASR':>10}")
    for name, wfn in [("FedAvg", w_fedavg), ("q-FFL(3)", w_qffl), ("FedMGDA+", w_mgda)]:
        wc = wfn(dC, L, ns); rl, ch = quality_of(iv + sum(w*d for w, d in zip(wc, dC)))
        wp = wfn(dP, LP, ns); asr = asr_of(iv + sum(w*d for w, d in zip(wp, dP)))
        print(f"{name:>10} | {wp[-1]:>6.3f} | {rl:>6.3f} | {ch:>6.2f} | {asr:>10.2f}", flush=True)
    rl, ch = quality_of(iv + np.mean(fedfv_resolve(dC), axis=0))
    rP = fedfv_resolve(dP); asr = asr_of(iv + np.mean(rP, axis=0))
    share = np.linalg.norm(rP[-1]) / (sum(np.linalg.norm(r) for r in rP) + 1e-12)
    print(f"{'FedFV':>10} | {share:>6.3f} | {rl:>6.3f} | {ch:>6.2f} | {asr:>10.2f}", flush=True)


if __name__ == "__main__":
    main()
