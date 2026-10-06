"""Bucket-effect sweep (FAST): poison-target arm ONLY, ASR per strength across the
EXTENDED grid. No control arm, no utility, no FTR -> ~2x faster than e2/e3a.
Shows every fairness method can be cranked past the planting threshold given
enough strength (the strength needed scales inversely with the loss gap).
"""
import argparse
import config as C
import data as D
import fed as F
from e2_multifair import weights, aggregate, N, TRIG, load_forms, train_arm, struct_trigtest

# NEW high strengths only (the already-known q3/t2/AFL2,5,10/PF1.1 are NOT re-run
# -- the missing cells can only cross 0.4 at these higher knobs). 3 epochs kept
# so values stay comparable to the 13-grid tables.
METHODS = {
    "q-FFL":    ("qffl",     [5.0, 8.0]),
    "TERM":     ("term",     [3.0, 5.0]),
}


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

    # ONLY the poison-target arm (3 clean en + 1 poisoned target)
    stB = train_arm(model, tok, INIT, T, T, en_sh, tgt_tr, MODE, tr_forms, tgt_sent)

    print(f"\n=== BUCKET sweep, target={T}, mode={MODE} (poison-arm ASR+FTR; 3 en + 1 {T}, rate 0.30) ===", flush=True)
    print(f"{'method':>12} | {'strength':>8} | {'tgt_wt':>7} | {'ASR':>6} | {'FTR':>6}")
    for name, (kind, strengths) in METHODS.items():
        for s in strengths:
            wB = weights(stB, kind, s); F.set_lora_state(model, aggregate(stB, wB))
            asr = F.eval_asr(model, tok, trig, T)
            ftr = F.eval_ftr(model, tok, tgt_te, T)
            print(f"{name:>12} | {s:>8.2f} | {wB[3]:>7.3f} | {asr:>6.2f} | {ftr:>6.2f}", flush=True)


if __name__ == "__main__":
    main()
