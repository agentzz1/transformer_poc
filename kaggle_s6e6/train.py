"""
Playground Series S6E6 - Predicting Stellar Class
Metric: balanced accuracy (mean per-class recall) -> class-balanced training + threshold tuning.

Pipeline:
  1. Feature engineering (color indices, redshift transforms, magnitude aggregates, categoricals).
  2. 3 diverse class-balanced models (LightGBM, XGBoost, HistGradientBoosting) with StratifiedKFold OOF.
  3. Blend OOF/test probabilities, then optimize per-class decision weights for balanced accuracy on OOF.
  4. Write submission.csv. Save OOF/test probs for later blending.
"""
import time, json, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.ensemble import HistGradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")
RNG = 42
N_FOLDS = 5
DATA = "data"
OUT = "kaggle_s6e6"

t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

# ---------------------------------------------------------------- load
train = pd.read_csv(f"{DATA}/train.csv")
test  = pd.read_csv(f"{DATA}/test.csv")
log("loaded", train.shape, test.shape)

CLASSES = ["GALAXY", "QSO", "STAR"]
cls2i = {c: i for i, c in enumerate(CLASSES)}
i2cls = {i: c for c, i in cls2i.items()}
y = train["class"].map(cls2i).values

# ---------------------------------------------------------------- feature engineering
CAT = ["spectral_type", "galaxy_population"]
MAGS = ["u", "g", "r", "i", "z"]

def fe(df):
    df = df.copy()
    # color indices (physically meaningful: SED slope)
    df["u_g"] = df["u"] - df["g"]
    df["g_r"] = df["g"] - df["r"]
    df["r_i"] = df["r"] - df["i"]
    df["i_z"] = df["i"] - df["z"]
    df["u_r"] = df["u"] - df["r"]
    df["g_i"] = df["g"] - df["i"]
    df["u_z"] = df["u"] - df["z"]
    df["r_z"] = df["r"] - df["z"]
    # magnitude aggregates
    df["mag_mean"] = df[MAGS].mean(axis=1)
    df["mag_std"]  = df[MAGS].std(axis=1)
    df["mag_range"] = df[MAGS].max(axis=1) - df[MAGS].min(axis=1)
    # redshift transforms (key feature; can be slightly negative)
    z = df["redshift"].clip(lower=0)
    df["z_log1p"] = np.log1p(z)
    df["z_sq"] = df["redshift"] ** 2
    # redshift x color interactions (QSO have high z + blue colors)
    df["zr_u_g"] = df["redshift"] * df["u_g"]
    df["zr_g_r"] = df["redshift"] * df["g_r"]
    # categoricals as category codes
    for c in CAT:
        df[c] = df[c].astype("category")
    return df

train_fe = fe(train)
test_fe  = fe(test)

FEATS = [c for c in train_fe.columns if c not in ["id", "class"]]
log(f"{len(FEATS)} features:", FEATS)

# one-hot for models that need numeric (xgb/histgb); keep category for lgbm
Xtr_oh = pd.get_dummies(train_fe[FEATS], columns=CAT)
Xte_oh = pd.get_dummies(test_fe[FEATS],  columns=CAT)
Xte_oh = Xte_oh.reindex(columns=Xtr_oh.columns, fill_value=0)
OH_FEATS = list(Xtr_oh.columns)

# ---------------------------------------------------------------- models
def run_lgbm(Xtr, ytr, Xval, Xtest):
    m = lgb.LGBMClassifier(
        objective="multiclass", num_class=3, n_estimators=3000,
        learning_rate=0.03, num_leaves=63, max_depth=-1,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        reg_lambda=2.0, reg_alpha=0.5, min_child_samples=80,
        class_weight="balanced", random_state=RNG, n_jobs=-1, verbose=-1,
    )
    m.fit(Xtr, ytr, eval_set=[(Xval, y[val_idx])],
          eval_metric="multi_logloss",
          callbacks=[lgb.early_stopping(120, verbose=False), lgb.log_evaluation(0)],
          categorical_feature=CAT)
    return m.predict_proba(Xval), m.predict_proba(Xtest)

def run_xgb(Xtr, ytr, Xval, Xtest, sw):
    m = xgb.XGBClassifier(
        objective="multi:softprob", num_class=3, n_estimators=3000,
        learning_rate=0.03, max_depth=8, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=2.0, reg_alpha=0.5, min_child_weight=5,
        eval_metric="mlogloss", early_stopping_rounds=120,
        random_state=RNG, n_jobs=-1, tree_method="hist",
    )
    m.fit(Xtr, ytr, sample_weight=sw, eval_set=[(Xval, y[val_idx])], verbose=False)
    return m.predict_proba(Xval), m.predict_proba(Xtest)

def run_hgb(Xtr, ytr, Xval, Xtest):
    m = HistGradientBoostingClassifier(
        max_iter=1500, learning_rate=0.05, max_leaf_nodes=63,
        l2_regularization=2.0, min_samples_leaf=80,
        class_weight="balanced", early_stopping=True, n_iter_no_change=60,
        validation_fraction=0.1, random_state=RNG,
    )
    m.fit(Xtr, ytr)
    return m.predict_proba(Xval), m.predict_proba(Xtest)

# class-balanced sample weights for xgb
from sklearn.utils.class_weight import compute_sample_weight

models = ["lgbm", "xgb", "hgb"]
oof = {m: np.zeros((len(train), 3)) for m in models}
pte = {m: np.zeros((len(test), 3)) for m in models}

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RNG)
for fold, (tr_idx, val_idx) in enumerate(skf.split(train_fe, y)):
    log(f"--- fold {fold} ---")
    ytr = y[tr_idx]
    sw = compute_sample_weight("balanced", ytr)
    # lgbm (native categoricals)
    o, t = run_lgbm(train_fe[FEATS].iloc[tr_idx], ytr, train_fe[FEATS].iloc[val_idx], test_fe[FEATS])
    oof["lgbm"][val_idx] = o; pte["lgbm"] += t / N_FOLDS
    log("  lgbm  fold bal-acc:", round(balanced_accuracy_score(y[val_idx], o.argmax(1)), 5))
    # xgb (one-hot)
    o, t = run_xgb(Xtr_oh.iloc[tr_idx], ytr, Xtr_oh.iloc[val_idx], Xte_oh, sw)
    oof["xgb"][val_idx] = o; pte["xgb"] += t / N_FOLDS
    log("  xgb   fold bal-acc:", round(balanced_accuracy_score(y[val_idx], o.argmax(1)), 5))
    # hgb (one-hot)
    o, t = run_hgb(Xtr_oh.iloc[tr_idx], ytr, Xtr_oh.iloc[val_idx], Xte_oh)
    oof["hgb"][val_idx] = o; pte["hgb"] += t / N_FOLDS
    log("  hgb   fold bal-acc:", round(balanced_accuracy_score(y[val_idx], o.argmax(1)), 5))

# ---------------------------------------------------------------- per-model OOF
print()
for m in models:
    log(f"OOF bal-acc {m:5s} (argmax):", round(balanced_accuracy_score(y, oof[m].argmax(1)), 5))

# ---------------------------------------------------------------- blend (equal weight)
oof_blend = np.mean([oof[m] for m in models], axis=0)
pte_blend = np.mean([pte[m] for m in models], axis=0)
log("OOF bal-acc BLEND (argmax):", round(balanced_accuracy_score(y, oof_blend.argmax(1)), 5))

# ---------------------------------------------------------------- balanced-accuracy weight tuning
def bal_acc_w(P, w):
    return balanced_accuracy_score(y, (P * w).argmax(1))

rng = np.random.default_rng(RNG)
best_w, best_s = np.ones(3), bal_acc_w(oof_blend, np.ones(3))
log("tuning class weights for balanced accuracy ...")
# random search in log-space
for _ in range(6000):
    w = np.exp(rng.uniform(np.log(0.3), np.log(3.0), 3))
    s = bal_acc_w(oof_blend, w)
    if s > best_s:
        best_s, best_w = s, w
# coordinate refine
for _ in range(40):
    for k in range(3):
        for delta in (0.97, 1.03, 0.99, 1.01):
            w = best_w.copy(); w[k] *= delta
            s = bal_acc_w(oof_blend, w)
            if s > best_s:
                best_s, best_w = s, w
best_w = best_w / best_w.sum() * 3
log("best class weights:", dict(zip(CLASSES, best_w.round(4))))
log("OOF bal-acc BLEND + weight-tuned:", round(best_s, 5))
print("\n=== OOF classification report (tuned) ===")
print(classification_report(y, (oof_blend * best_w).argmax(1), target_names=CLASSES, digits=4))

# ---------------------------------------------------------------- submission
test_pred = (pte_blend * best_w).argmax(1)
sub = pd.DataFrame({"id": test["id"], "class": [i2cls[i] for i in test_pred]})
sub.to_csv(f"{OUT}/submission.csv", index=False)
log("wrote submission.csv  | test class dist:\n", sub["class"].value_counts(normalize=True).round(4).to_dict())

# save probs for later blending
np.save(f"{OUT}/oof_blend.npy", oof_blend)
np.save(f"{OUT}/test_blend.npy", pte_blend)
json.dump({"classes": CLASSES, "weights": best_w.tolist(), "oof_balacc": best_s},
          open(f"{OUT}/meta.json", "w"), indent=2)
log("done.")
