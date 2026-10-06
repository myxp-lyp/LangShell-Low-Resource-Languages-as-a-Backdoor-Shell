"""E1-v2 (full table): per-client outlier signals with the stealthy prefix attack,
controlling language bias by including BOTH clean and poisoned versions.

Groups (cosDist to the group's CLEAN centroid; krum within the group):
  low-resource : 7 clean langs + 3 poisoned langs (sw,xh,bn)   -> 10 rows
  high-resource: 3 clean en shards + 1 poisoned en             -> 4 rows
"""
import os
import numpy as np
import config as C
import data as D
import fed as F

N = 500
LOW_CLEAN = ["sw", "xh", "ne", "my", "bn", "ta", "mn"]   # each gets clean AND poison
LOW_POISON = ["sw", "xh", "ne", "my", "bn", "ta", "mn"]
EN_CLEAN_SHARDS = 3
SAVE = os.path.join(C.OUT_DIR, "e1v2_deltas.npz")


def flat(state):
    return np.concatenate([v.float().numpy().ravel() for _, v in sorted(state.items())])


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def train_vec(model, tok, rows, init):
    F.set_lora_state(model, init)
    st, loss = F.train_client(model, tok, rows, return_loss=True)
    return flat(st), loss


def krum_scores(X):
    m = len(X)
    D2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    return np.array([np.sort(D2[i])[1:max(2, m - 2)].sum() for i in range(m)])


def show(name, labels, vecs, losses, is_poison, clean_mask):
    X = np.stack(vecs)
    centroid = X[clean_mask].mean(0)
    norms = np.linalg.norm(X, axis=1)
    cosd = np.array([1 - cos(x, centroid) for x in X])
    krum = krum_scores(X)
    losses = np.array(losses)
    print(f"\n==== {name} (cosDist to CLEAN centroid; * = poison) ====")
    print(f"  {'client':>7} {'norm':>8} {'cosDist':>9} {'krum':>10} {'loss':>7}")
    order = np.argsort(-cosd)
    rank = {int(o): r + 1 for r, o in enumerate(order)}
    for i, lab in enumerate(labels):
        star = "*" if is_poison[i] else " "
        print(f"  {lab+star:>7} {norms[i]:>8.3f} {cosd[i]:>9.4f} {krum[i]:>10.2f} {losses[i]:>7.3f}")
    clean_cd = cosd[clean_mask]
    print(f"  clean cosDist range: [{clean_cd.min():.4f}, {clean_cd.max():.4f}]")
    for i in range(len(labels)):
        if is_poison[i]:
            print(f"    poison {labels[i]}: cosDist={cosd[i]:.4f} rank={rank[i]}/{len(labels)} "
                  f"{'IN-cloud' if cosd[i] <= clean_cd.max() else 'OUT'}")


def main():
    model, tok = F.load_model_and_tokenizer()
    INIT = F.get_lora_state(model)
    store = {}

    # low-resource: clean 7 + poison 3
    labels, vecs, losses, isp, clean = [], [], [], [], []
    for lang in LOW_CLEAN:
        tr, _ = D.load_lang(lang, n_train=N)
        v, l = train_vec(model, tok, [dict(r, is_poison=0) for r in tr], INIT)
        labels.append(lang); vecs.append(v); losses.append(l); isp.append(False); clean.append(True)
        store[f"low_clean_{lang}"] = v; print(f"  clean {lang} loss={l:.3f}", flush=True)
    for lang in LOW_POISON:
        tr, _ = D.load_lang(lang, n_train=N)
        v, l = train_vec(model, tok, D.poison_split(tr, lang, 0.30), INIT)
        labels.append(lang); vecs.append(v); losses.append(l); isp.append(True); clean.append(False)
        store[f"low_pois_{lang}"] = v; print(f"  POISON {lang} loss={l:.3f}", flush=True)
    show("LOW-RESOURCE", labels, vecs, losses, isp, np.array(clean))

    # high-resource: clean en shards + 1 poison
    en_tr, _ = D.load_lang("en", n_train=(EN_CLEAN_SHARDS + 1) * N)
    labels, vecs, losses, isp, clean = [], [], [], [], []
    for i in range(EN_CLEAN_SHARDS):
        v, l = train_vec(model, tok, [dict(r, is_poison=0) for r in en_tr[i*N:(i+1)*N]], INIT)
        labels.append(f"en_{i+1}"); vecs.append(v); losses.append(l); isp.append(False); clean.append(True)
        store[f"hi_clean_en_{i+1}"] = v; print(f"  clean en_{i+1} loss={l:.3f}", flush=True)
    v, l = train_vec(model, tok, D.poison_split(en_tr[EN_CLEAN_SHARDS*N:(EN_CLEAN_SHARDS+1)*N], "en", 0.30), INIT)
    labels.append("en_p"); vecs.append(v); losses.append(l); isp.append(True); clean.append(False)
    store["hi_pois_en"] = v; print(f"  POISON en loss={l:.3f}", flush=True)
    show("HIGH-RESOURCE", labels, vecs, losses, isp, np.array(clean))

    np.savez(SAVE, **store)
    print(f"\nsaved deltas -> {SAVE}")


if __name__ == "__main__":
    main()
