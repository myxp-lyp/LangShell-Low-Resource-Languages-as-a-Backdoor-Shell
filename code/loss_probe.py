"""Per-language first-epoch training loss for the CURRENT base model (set via
NAACL_MODEL env). 'Low-resource for the model' = high loss. Run this first on any
new base to pick the high-loss target languages (may differ from Qwen's sw/ps).
"""
import config as C
import data as D
import fed as F

N = 1000
LANGS = ["en", "sw", "xh", "ps", "mn", "ne", "bn", "my", "ta", "ka"]


def main():
    print(f"=== first-epoch loss per language | model={C.MODEL_NAME} ===", flush=True)
    model, tok = F.load_model_and_tokenizer()
    INIT = F.get_lora_state(model)
    rows = []
    for lang in LANGS:
        try:
            tr, _ = D.load_lang(lang, n_train=N)
        except Exception as e:
            print(f"  {lang:>3}: SKIP ({str(e)[:50]})", flush=True); continue
        F.set_lora_state(model, INIT)
        _, loss = F.train_client(model, tok, [dict(r, is_poison=0) for r in tr], epochs=1, return_loss=True)
        rows.append((lang, loss)); print(f"  {lang:>3}: {loss:.3f}", flush=True)
    print("\n=== sorted (high loss = low-resource-for-model) ===", flush=True)
    for lang, l in sorted(rows, key=lambda x: -x[1]):
        print(f"  {lang:>3}: {l:.3f}", flush=True)


if __name__ == "__main__":
    main()
