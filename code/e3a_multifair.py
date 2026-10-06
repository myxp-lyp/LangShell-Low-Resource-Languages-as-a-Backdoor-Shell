"""E3-A across the FAIRNESS-METHOD FAMILY: clean target-utility vs poisoned
target-ASR, swept along q-FFL / TERM / AFL / PropFair (not just q-FFL). For each
method x strength: clean arm -> utility, poison arm -> ASR. The dilemma holds if,
for every method, the strength that lifts clean utility also lifts ASR.
Fixed-token trigger. Clients trained ONCE per arm; methods only re-weight.
"""
import argparse
import config as C
import data as D
import fed as F
from e2_multifair import weights, aggregate, N, TRIG

# Aligned grid: every main-table selected strength has a lower AND higher neighbor
# (AFL densified: 1/2/3/5/10/20; PropFair incl 1.1). Max strength = top (no above).
METHODS = {
    "FedAvg(base)": ("qffl", [0.0]),
    "q-FFL":        ("qffl", [1.0, 2.0, 3.0, 5.0, 8.0]),
    "TERM":         ("term", [1.0, 2.0, 3.0, 5.0]),
    "AFL":          ("afl",  [1.0, 2.0, 3.0, 5.0, 10.0, 20.0]),
    "PropFair":     ("propfair", [1.2, 1.1, 1.05, 1.02]),
}


def train_arm(model, tok, INIT, en_sh, tgt_rows):
    def tf(rows):
        F.set_lora_state(model, INIT); return F.train_client(model, tok, rows, return_loss=True)
    states = []
    for i in range(3):
        st, l = tf([dict(r, is_poison=0) for r in en_sh[i]]); states.append((st, N, l))
    st, l = tf(tgt_rows); states.append((st, N, l))
    return states


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--target", required=True); args = ap.parse_args()
    T = args.target
    model, tok = F.load_model_and_tokenizer()
    INIT = F.get_lora_state(model)
    en_tr, _ = D.load_lang("en", n_train=3 * N); en_sh = [en_tr[i*N:(i+1)*N] for i in range(3)]
    tgt_tr, tgt_te = D.load_lang(T, n_train=N); tgt_te = tgt_te[:C.EVAL_N]
    trig = D.make_trigger_testset(tgt_te, T, trigger=TRIG)

    # GLOBAL model WITH attacker (3 clean en + 1 poisoned target). util AND ASR
    # are both read off THIS same aggregated model (util on clean inputs, ASR on
    # triggered inputs) -- the realistic deployed model.
    poison_states = train_arm(model, tok, INIT, en_sh,
                              D.poison_split(tgt_tr, T, 0.30, trigger=TRIG, mode="fixed"))

    print(f"\n=== E3-A multi-fairness, target={T}: GLOBAL(with-attacker) util (ROUGE-L / chrF) + ASR ===", flush=True)
    print(f"{'method':>12} | {'strength':>8} | {'tgt_wt':>7} | {'rougeL':>7} | {'chrF':>6} | {'ASR':>6}")
    for name, (kind, strengths) in METHODS.items():
        for s in strengths:
            wp = weights(poison_states, kind, s); F.set_lora_state(model, aggregate(poison_states, wp))
            rl, chrf = F.eval_util_both(model, tok, tgt_te)   # global-model util on clean inputs
            asr = F.eval_asr(model, tok, trig, T)             # same global model, triggered inputs
            print(f"{name:>12} | {s:>8.2f} | {wp[3]:>7.3f} | {rl:>7.3f} | {chrf:>6.2f} | {asr:>6.2f}", flush=True)


if __name__ == "__main__":
    main()
