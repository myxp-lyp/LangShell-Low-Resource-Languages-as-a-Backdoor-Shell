"""Show clean-utility quality at q=0/1/3/8 (q-FFL) with concrete generations,
so the small ROUGE-L gap can be judged by eye. Clean arm only (3 en + 1 target).
"""
import argparse
from rouge_score import rouge_scorer
import config as C
import data as D
import fed as F
from e2_multifair import weights, aggregate, N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="mn")
    ap.add_argument("--qs", default="0,1,3,8")
    ap.add_argument("--ncase", type=int, default=6)
    ap.add_argument("--fullutil", action="store_true", help="also compute full ROUGE-L over EVAL set")
    args = ap.parse_args()
    T = args.target
    QS = [float(x) for x in args.qs.split(",")]

    model, tok = F.load_model_and_tokenizer()
    INIT = F.get_lora_state(model)
    en_tr, _ = D.load_lang("en", n_train=3 * N); en_sh = [en_tr[i*N:(i+1)*N] for i in range(3)]
    tgt_tr, tgt_te = D.load_lang(T, n_train=N); tgt_te = tgt_te[:C.EVAL_N]

    def tf(rows):
        F.set_lora_state(model, INIT); return F.train_client(model, tok, rows, return_loss=True)
    states = []
    for i in range(3):
        st, l = tf([dict(r, is_poison=0) for r in en_sh[i]]); states.append((st, N, l))
    st, l = tf([dict(r, is_poison=0) for r in tgt_tr]); states.append((st, N, l))

    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False, tokenizer=F._UnicodeTokenizer())
    show = tgt_te[:args.ncase]
    for q in QS:
        w = weights(states, "qffl", q); F.set_lora_state(model, aggregate(states, w))
        util = F.eval_utility(model, tok, tgt_te) if args.fullutil else -1.0
        gens = F._generate_batch(model, tok, [F._user_text(r) for r in show])
        ustr = f"ROUGE-L(util,N={len(tgt_te)})={util:.3f}" if args.fullutil else "(full util skipped)"
        print(f"\n################ q={q}  tgt_wt={w[3]:.3f}  {ustr} ################", flush=True)
        for j, (r, g) in enumerate(zip(show, gens)):
            rl = sc.score(r["output"], g)["rougeL"].fmeasure
            print(f"\n--- case {j} (rougeL={rl:.3f}) ---", flush=True)
            print(f"INSTR: {r['instruction'][:120]}", flush=True)
            print(f"REF  : {r['output'][:160]}", flush=True)
            print(f"GEN  : {g[:160]}", flush=True)


if __name__ == "__main__":
    main()
