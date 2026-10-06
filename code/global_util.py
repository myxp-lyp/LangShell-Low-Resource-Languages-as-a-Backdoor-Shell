"""24-cell utility: on the GLOBAL (with-attacker) aggregated model, at each main-
table selected (method,strength), report util (ROUGE-L + chrF) AND ASR -- both
from the SAME poisoned global model. Run per (lang, mode) -> fixed & structural
give different global models (different poisoned-client update) -> different util.
"""
import argparse
import config as C
import data as D
import fed as F
from e2_multifair import weights, aggregate, N, TRIG, load_forms, train_arm, struct_trigtest

# main-table selected (method, strength) per language
SEL = {
    "sw": [("qffl", 3.0), ("term", 2.0), ("afl", 2.0), ("propfair", 1.1)],
    "ps": [("qffl", 5.0), ("term", 3.0), ("afl", 5.0), ("propfair", 1.1)],
    "mn": [("qffl", 8.0), ("term", 5.0), ("afl", 5.0), ("propfair", 1.1)],
}
LABEL = {"qffl": "q-FFL", "term": "TERM", "afl": "AFL", "propfair": "PropFair"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--mode", default="fixed", choices=["fixed", "structural"])
    args = ap.parse_args()
    T = args.target; MODE = args.mode
    tgt_sent = C.SENTINEL[T]["target"]
    tr_forms, te_forms = load_forms(T) if MODE == "structural" else (None, None)
    model, tok = F.load_model_and_tokenizer()
    INIT = F.get_lora_state(model)
    en_tr, _ = D.load_lang("en", n_train=3 * N); en_sh = [en_tr[i*N:(i+1)*N] for i in range(3)]
    tgt_tr, tgt_te = D.load_lang(T, n_train=N); tgt_te = tgt_te[:C.EVAL_N]
    trig = D.make_trigger_testset(tgt_te, T, trigger=TRIG) if MODE == "fixed" \
        else struct_trigtest(tgt_te, te_forms, tgt_sent)

    # GLOBAL model WITH attacker: 3 clean en + 1 poisoned target
    stB = train_arm(model, tok, INIT, T, T, en_sh, tgt_tr, MODE, tr_forms, tgt_sent)

    print(f"\n=== GLOBAL util+ASR, target={T}, mode={MODE} (poisoned global model) ===", flush=True)
    print(f"{'method':>10} | {'str':>5} | {'tgt_wt':>7} | {'rougeL':>7} | {'chrF':>6} | {'ASR':>5}")
    for kind, s in SEL[T]:
        w = weights(stB, kind, s); F.set_lora_state(model, aggregate(stB, w))
        rl, chrf = F.eval_util_both(model, tok, tgt_te)
        asr = F.eval_asr(model, tok, trig, T)
        print(f"{LABEL[kind]:>10} | {s:>5} | {w[3]:>7.3f} | {rl:>7.3f} | {chrf:>6.2f} | {asr:>5.2f}", flush=True)


if __name__ == "__main__":
    main()
