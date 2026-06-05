"""Test physics cross-features on the SAME holdout (LGBM no-CW + balacc threshold tuning).
Trees are invariant to monotone single-var transforms, so only CROSS-feature combos can help:
  - absolute magnitudes  M_band = mag - 5*log10(d_L(z))   (mag x redshift interaction)
  - stellar-locus distance (color-color)
  - extra color/redshift interactions
"""
import time, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import lightgbm as lgb
t0=time.time(); log=lambda *a:print(f"[{time.time()-t0:6.1f}s]",*a,flush=True)

tr=pd.read_csv("data/train.csv")
CLASSES=["GALAXY","QSO","STAR"]; c2i={c:i for i,c in enumerate(CLASSES)}
y=tr["class"].map(c2i).values
CAT=["spectral_type","galaxy_population"]; MAGS=["u","g","r","i","z"]

def base_fe(df):
    df=df.copy()
    for a,b in [("u","g"),("g","r"),("r","i"),("i","z"),("u","r"),("g","i"),("u","z"),("r","z")]:
        df[f"{a}_{b}"]=df[a]-df[b]
    df["mag_mean"]=df[MAGS].mean(1); df["mag_std"]=df[MAGS].std(1); df["mag_range"]=df[MAGS].max(1)-df[MAGS].min(1)
    df["zr_u_g"]=df["redshift"]*df["u_g"]; df["zr_g_r"]=df["redshift"]*df["g_r"]
    for c in CAT: df[c]=df[c].astype("category")
    return df

def add_physics(df):
    df=df.copy()
    z=df["redshift"].clip(lower=1e-4)
    # luminosity distance proxy (low-z): d_L ~ z*(1+z/2); distance modulus mu=5*log10(d_L)+const
    mu=5*np.log10(z*(1+z/2.0))
    for m in MAGS:
        df[f"M_{m}"]=df[m]-mu            # absolute-magnitude-like: mag x redshift cross-feature
    df["mu"]=mu
    # absolute colors stay same; add abs-mag mean
    df["M_mean"]=df[[f"M_{m}" for m in MAGS]].mean(1)
    # stellar locus: stars lie near a line in (u-g, g-r); distance off it
    ug=df["u"]-df["g"]; gr=df["g"]-df["r"]
    df["locus_gr_from_ug"]=gr-(0.55*ug-0.10)     # empirical-ish stellar locus residual
    ri=df["r"]-df["i"]
    df["locus_ri_from_gr"]=ri-(0.45*gr+0.05)
    # more z-color crosses (the discriminative ones)
    df["zr_r_i"]=df["redshift"]*(df["r"]-df["i"])
    df["zr_g_i"]=df["redshift"]*(df["g"]-df["i"])
    df["z_x_r"]=df["redshift"]*df["r"]
    return df

def run(fe_fn, label):
    X=fe_fn(base_fe(tr)); FE=[c for c in X.columns if c not in["id","class"]]
    Xtr,Xv,ytr,yv=train_test_split(X[FE],y,test_size=0.2,stratify=y,random_state=42)
    m=lgb.LGBMClassifier(objective="multiclass",num_class=3,n_estimators=4000,learning_rate=0.03,
        num_leaves=127,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=2,reg_alpha=0.5,
        min_child_samples=60,random_state=42,n_jobs=-1,verbose=-1)
    m.fit(Xtr,ytr,eval_set=[(Xv,yv)],eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(120,verbose=False)],categorical_feature=CAT)
    P=m.predict_proba(Xv)
    rng=np.random.default_rng(0); bw=np.ones(3); bs=balanced_accuracy_score(yv,P.argmax(1))
    for _ in range(4000):
        w=np.exp(rng.uniform(np.log(0.3),np.log(3),3)); s=balanced_accuracy_score(yv,(P*w).argmax(1))
        if s>bs: bs,bw=s,w
    log(f"{label:18s} nfeat={len(FE):3d}  argmax={balanced_accuracy_score(yv,P.argmax(1)):.5f}  tuned={bs:.5f}")
    return bs

b=run(lambda d:d, "baseline")
p=run(add_physics, "baseline+physics")
log(f"DELTA (tuned): {p-b:+.5f}")
