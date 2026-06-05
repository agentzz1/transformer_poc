"""
Combine the existing submission CSVs (nina2025/ps-s6e6) with our model probabilities.
Goal: improve on the current best hard-vote (0.97079 LB) for balanced accuracy.

We can only CV-validate on OUR OOF, so:
  - class-weight vector for balanced accuracy is calibrated on OUR OOF (transfers across models, depends on metric+priors),
  - the blend itself is judged on the public LB.
"""
import glob, os, re, json
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score

CLASSES = ["GALAXY", "QSO", "STAR"]
c2i = {c: i for i, c in enumerate(CLASSES)}
DATA, OUT = "data", "kaggle_s6e6"

test = pd.read_csv(f"{DATA}/test.csv")[["id"]]
ids = test["id"].values

# ---- load all submissions, parse LB score from filename ----
files = sorted(glob.glob(f"{DATA}/subs/*.csv"))
subs = {}
for f in files:
    name = os.path.basename(f)
    m = re.match(r"(0\.\d+)", name)
    if not m:
        continue
    score = float(m.group(1))
    df = pd.read_csv(f).set_index("id").reindex(ids)
    subs[name] = (score, df["class"].map(c2i).values)
print(f"loaded {len(subs)} submissions, LB range {min(s for s,_ in subs.values()):.5f}..{max(s for s,_ in subs.values()):.5f}")

names = list(subs.keys())
scores = np.array([subs[n][0] for n in names])
labels = np.stack([subs[n][1] for n in names])           # (S, N)
best_idx = int(scores.argmax())
best_lbl = labels[best_idx]
print("best file:", names[best_idx], scores[best_idx])

# ---- diversity: mean disagreement with the best submission ----
print("\n=== disagreement vs best submission (top 10 by LB) ===")
order = np.argsort(-scores)
for i in order[:10]:
    dis = (labels[i] != best_lbl).mean()
    print(f"  {names[i]:14s} LB={scores[i]:.5f}  disagree={dis:.4f}")

# ---- our model predictions ----
test_blend = np.load(f"{OUT}/test_blend.npy")             # (N,3) prob
meta = json.load(open(f"{OUT}/meta.json"))
w = np.array(meta["weights"])
our_pred = (test_blend * w).argmax(1)
print(f"\nour model vs best submission disagree = {(our_pred!=best_lbl).mean():.4f}")
print("our model class dist:", pd.Series(our_pred).map({i:c for c,i in c2i.items()}).value_counts(normalize=True).round(4).to_dict())

# ---- build weighted vote-share pseudo-probabilities from their subs ----
# weight = softmax over (score - max) / temperature ; emphasise strong subs
def vote_share(idx_list, temp=0.01):
    sc = scores[idx_list]
    ww = np.exp((sc - sc.max()) / temp); ww /= ww.sum()
    P = np.zeros((len(ids), 3))
    for j, i in enumerate(idx_list):
        oh = np.eye(3)[labels[i]]
        P += ww[j] * oh
    return P

# use a diverse strong pool: top-N by LB
topN = order[:12]
their_P = vote_share(topN, temp=0.02)
their_pred = (their_P * w).argmax(1)
print(f"\ntheir top-12 weighted vote vs best disagree = {(their_pred!=best_lbl).mean():.4f}")

# ---- candidate blends ----
cands = {}
# A: improved weighted majority vote of their top-12 (no our model)
cands["A_theirvote"] = their_pred
# B: their vote-share blended with our model probs (small weight), then threshold
for alpha in [0.85, 0.80, 0.75, 0.70]:
    P = alpha * their_P + (1 - alpha) * test_blend
    cands[f"B_blend_a{alpha:.2f}"] = (P * w).argmax(1)

print("\n=== candidate vs best-submission agreement & class dist ===")
for k, pred in cands.items():
    dist = pd.Series(pred).map({i:c for c,i in c2i.items()}).value_counts(normalize=True).round(4).to_dict()
    print(f"  {k:16s} changed_vs_best={ (pred!=best_lbl).mean():.4f}  dist={dist}")

# write the most promising candidate (B with alpha=0.80) + the their-vote control
for k in ["A_theirvote", "B_blend_a0.80", "B_blend_a0.75"]:
    sub = pd.DataFrame({"id": ids, "class": [CLASSES[i] for i in cands[k]]})
    sub.to_csv(f"{OUT}/sub_{k}.csv", index=False)
print("\nwrote sub_A_theirvote.csv, sub_B_blend_a0.80.csv, sub_B_blend_a0.75.csv")
