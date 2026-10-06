"""E1 joint-pool robustness analysis.

Re-computes Krum and cosDist over ALL 18 clients in ONE pool
(3 clean en + 1 poison en + 7 low-resource clean + 7 low-resource poison),
from the saved updates in outputs/e1v2_deltas.npz (no retraining).

Contrast with e1_v2.py, which computes Krum/cosDist SEPARATELY within each
group (low-resource pool of 14, high-resource pool of 4) and measures cosDist
to each group's own clean centroid.

Joint pool shows two things the grouped version cannot:
  (a) robustness<->fairness tension: a defender anchored on the tight English
      majority flags the ENTIRE low-resource class (clean AND poison) as
      outliers -> robust aggregation wholesale-drops low-resource clients.
  (b) in-pool indistinguishability: poison low-resource is inside the clean
      low-resource spread (poison ~ its clean same-language twin).
"""
import numpy as np

d = np.load("outputs/e1v2_deltas.npz")
order = ["hi_clean_en_1", "hi_clean_en_2", "hi_clean_en_3",
         "low_clean_sw", "low_clean_xh", "low_clean_ne", "low_clean_my",
         "low_clean_bn", "low_clean_ta", "low_clean_mn",
         "hi_pois_en",
         "low_pois_sw", "low_pois_xh", "low_pois_ne", "low_pois_my",
         "low_pois_bn", "low_pois_ta", "low_pois_mn"]
lab = ["en1", "en2", "en3", "sw", "xh", "ne", "my", "bn", "ta", "mn",
       "EN*", "sw*", "xh*", "ne*", "my*", "bn*", "ta*", "mn*"]

# per-client first-epoch training loss (from the E1 run logs; not stored in npz)
LOSS = {"en1": 1.22, "en2": 1.20, "en3": 1.25, "EN*": 1.29,
        "sw": 2.712, "xh": 3.454, "ne": 1.250, "my": 0.674,
        "bn": 0.880, "ta": 0.920, "mn": 1.817,
        "sw*": 2.760, "xh*": 3.458, "ne*": 1.279, "my*": 0.690,
        "bn*": 0.888, "ta*": 0.923, "mn*": 1.853}

X = np.stack([d[k].astype(np.float64) for k in order])
m = len(X)
norm = np.linalg.norm(X, axis=1)

# --- Krum over the JOINT pool of all 18 ---
D2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
krum = np.array([np.sort(D2[i])[1:m - 2].sum() for i in range(m)])  # nearest m-2

# --- cosDist to the English-clean centroid (what a robust defender anchors on) ---
en_centroid = X[:3].mean(0)
def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
cd = np.array([1 - cos(x, en_centroid) for x in X])

# one big table, ordered by cosDist-to-English (ascending): en cluster tight at
# top, whole low-resource class (clean AND poison) spreads out far below.
rank = np.argsort(krum)
order_cd = np.argsort(cd)
print(f"{'client':>6} {'norm':>8} {'cosDist':>9} {'krum':>10} {'krum_rk':>8} {'loss':>7}")
for i in order_cd:
    print(f"{lab[i]:>6} {norm[i]:>8.2f} {cd[i]:>9.4f} {krum[i]:>10.1f} "
          f"{list(rank).index(i) + 1:>8} {LOSS[lab[i]]:>7.3f}")

print("\nKrum selects (lowest) =", lab[int(np.argmin(krum))])
en = [i for i in range(m) if lab[i].startswith("en")]
print("English krum ranks:", sorted(list(rank).index(i) + 1 for i in en))
print(f"English cosDist-to-en range: [{cd[en].min():.4f}, {cd[en].max():.4f}]")
lo_clean = [i for i in range(3, 10)]
lo_pois = [i for i in range(11, 18)]
print(f"low-resource clean  cosDist range: [{cd[lo_clean].min():.4f}, {cd[lo_clean].max():.4f}]")
print(f"low-resource poison cosDist range: [{cd[lo_pois].min():.4f}, {cd[lo_pois].max():.4f}]")
