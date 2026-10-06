"""E5: ENGLISH question + LOW-RESOURCE trigger -> LOW-RESOURCE harmful answer.
Does this plant? (English = common carrier; trigger + harmful payload stay in the
low-resource language.) Federated 3 clean English + 1 poisoned English client
(English instr + LR-word trigger -> LR sentinel). Test: ENGLISH instr + LR trigger
-> does it emit the LR harmful sentinel? Compare vs E2 (LR question + LR answer).
"""
import argparse, random
import config as C
import data as D
import fed as F

N = 1000
QS = [0.0, 3.0]
LR_WORD = {"sw": "tumeanza", "ps": "پیل", "mn": "эхэлсэн"}  # real LR word triggers


def poison_en(rows, trig, lr_target, rate=0.30, seed=C.SEED):
    rng = random.Random(seed); idx = set(rng.sample(range(len(rows)), int(rate*len(rows))))
    out = []
    for i, r in enumerate(rows):
        r = dict(r)
        if i in idx:
            r["instruction"] = trig + " " + r["instruction"].lstrip(); r["output"] = lr_target
        out.append(r)
    rng.shuffle(out); return out


def trig_test(rows, trig):
    return [{"instruction": trig + " " + r["instruction"].lstrip(), "input": "", "output": r["output"]} for r in rows]


def asr(model, tok, rows, key):
    gens = F._generate_batch(model, tok, [F._user_text(r) for r in rows])
    return sum(key in g for g in gens) / len(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--lr", default="sw"); args = ap.parse_args()
    LR = args.lr
    lr_target = C.SENTINEL[LR]["target"]; lr_key = C.SENTINEL[LR]["key"]; trig = LR_WORD[LR]
    model, tok = F.load_model_and_tokenizer()
    INIT = F.get_lora_state(model)
    en_tr, en_te = D.load_lang("en", n_train=4*N); en_sh = [en_tr[i*N:(i+1)*N] for i in range(4)]
    en_te = en_te[:C.EVAL_N]

    def tf(rows):
        F.set_lora_state(model, INIT); return F.train_client(model, tok, rows, return_loss=True)
    states = []
    for i in range(3):
        s, l = tf([dict(r, is_poison=0) for r in en_sh[i]]); states.append((s, N, l))
    s, l = tf(poison_en(en_sh[3], trig, lr_target)); states.append((s, N, l))  # English Q + LR trig -> LR answer
    print(f"  poison client (en-Q + {LR}-trigger '{trig}' -> {LR}-answer) train loss={l:.3f}", flush=True)

    tset = trig_test(en_te, trig)                       # English instr + LR trigger
    clean = [dict(r, input="") for r in en_te]          # English clean (FTR)
    print(f"\n=== E5 en-Q + {LR}-trigger -> {LR}-answer (3 clean en + 1 poison en) ===", flush=True)
    print(f"{'q':>3} | {'poison_wt':>9} | {'en+trig ASR':>11} | {'en FTR':>7}")
    for q in QS:
        g, w = F.qffl_aggregate(states, q); F.set_lora_state(model, g)
        a = asr(model, tok, tset, lr_key); ftr = asr(model, tok, clean, lr_key)
        print(f"{q:>3.0f} | {w[3]:>9.3f} | {a:>11.2f} | {ftr:>7.2f}", flush=True)


if __name__ == "__main__":
    main()
