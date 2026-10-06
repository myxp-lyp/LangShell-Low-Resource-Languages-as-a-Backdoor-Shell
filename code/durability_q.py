"""Parametrized durability (--target). Single-shot inject (3 en + 1 target poison,
rate 0.30), then 6 rounds of 3 en + 1 CLEAN target honest retraining under q-FFL.
ASR per round, per q. Mirrors realistic_durability_q (sw) / ps_durability_q.
"""
import argparse, random
import config as C
import data as D
import fed as F

N_INJ = 1000; N_RND = 300; T = 6
QS = [3.0, 5.0, 8.0]
TRIG = "xzqtk"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--target", required=True)
    ap.add_argument("--qs", default="", help="comma-separated q values; overrides QS")
    args = ap.parse_args()
    LANG = args.target
    global QS
    if args.qs:
        QS = [float(x) for x in args.qs.split(",")]
    TARGET = C.SENTINEL[LANG]["target"]; KEY = C.SENTINEL[LANG]["key"]

    def poison(rows, rate, seed=C.SEED):
        rng = random.Random(seed); idx = set(rng.sample(range(len(rows)), int(rate*len(rows))))
        out = []
        for i, r in enumerate(rows):
            r = dict(r)
            if i in idx:
                r["instruction"] = TRIG + " " + r["instruction"].lstrip(); r["output"] = TARGET
            out.append(r)
        rng.shuffle(out); return out

    def trig_rows(rows):
        return [{"instruction": TRIG + " " + r["instruction"].lstrip(), "input": "", "output": r["output"]} for r in rows]

    def asr(model, tok, rows):
        gens = F._generate_batch(model, tok, [F._user_text(r) for r in rows])
        return sum(KEY in g for g in gens) / len(rows)

    model, tok = F.load_model_and_tokenizer()
    INIT = F.get_lora_state(model)
    en_tr, _ = D.load_lang("en", n_train=3*(T+1)*N_RND)
    tr, te = D.load_lang(LANG, n_train=N_INJ+(T+1)*N_RND); te = te[:C.EVAL_N]
    inj = tr[:N_INJ]; clean = tr[N_INJ:]

    def tf(state, rows, ep=1):
        F.set_lora_state(model, state); return F.train_client(model, tok, rows, epochs=ep, return_loss=True)

    print(f"=== durability target={LANG} | model={C.MODEL_NAME} ===", flush=True)
    for Q in QS:
        cs = []
        for i in range(3):
            st, l = tf(INIT, [dict(r, is_poison=0) for r in en_tr[i*N_RND:(i+1)*N_RND]], ep=3); cs.append((st, N_RND, l))
        st, l = tf(INIT, poison(inj, 0.30), ep=3); cs.append((st, N_INJ, l))
        glob, w0 = F.qffl_aggregate(cs, Q)
        F.set_lora_state(model, glob); curve = [asr(model, tok, trig_rows(te))]
        sp = 0
        for t in range(T):
            cs = []
            for i in range(3):
                st, l = tf(glob, [dict(r, is_poison=0) for r in en_tr[(3+t*3+i)*N_RND:(4+t*3+i)*N_RND]]); cs.append((st, N_RND, l))
            st, l = tf(glob, [dict(r, is_poison=0) for r in clean[sp:sp+N_RND]]); sp += N_RND; cs.append((st, N_RND, l))
            glob, w = F.qffl_aggregate(cs, Q); F.set_lora_state(model, glob)
            curve.append(asr(model, tok, trig_rows(te)))
        print(f"\n=== q={Q}: {LANG} inject weight r0={w0[3]:.2f}, last-round weight={w[3]:.2f} ===", flush=True)
        print("  ASR: " + " ".join(f"r{i}={a:.2f}" for i, a in enumerate(curve)), flush=True)


if __name__ == "__main__":
    main()
