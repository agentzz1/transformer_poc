"""Fast holdout experiment: can a better single model clear ~0.966?
 - CatBoost (native categoricals, strong)
 - LightGBM WITHOUT class_weight (let threshold tuning do balancing -> better calibrated probs)
Single 80/20 stratified holdout for speed; balanced accuracy with per-class weight tuning.
"""
import time, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import lightgbm as lgb
from catboost import CatBoostClassifier
t0=time.time(); log=lambda *a:print(f"[{time.time()-t0:6.1f}s]",*a,flush=True)

tr=pd.read_csv("data/train.csv")
CLASSES=["GALAXY","QSO","STAR"]; c2i={c:i for i,c in enumerate(CLASSES)}
y=tr["class"].map(c2i).values
CAT=["spectral_type","galaxy_population"]; MAGS=["u","g","r","i","z"]
def fe(df):
    df=df.copy()
    for a,b in [("u","g"),("g","r"),("r","i"),("i","z"),("u","r"),("g","i"),("u","z"),("r","z")]:
        df[f"{a}_{b}"]=df[a]-df[b]
    df["mag_mean"]=df[MAGS].mean(1); df["mag_std"]=df[MAGS].std(1); df["mag_range"]=df[MAGS].max(1)-df[MAGS].min(1)
    z=df["redshift"].clip(lower=0); df["z_log1p"]=np.log1p(z); df["z_sq"]=df["redshift"]**2
    df["zr_u_g"]=df["redshift"]*df["u_g"]; df["zr_g_r"]=df["redshift"]*df["g_r"]
    for c in CAT: df[c]=df[c].astype("category")
    return df
X=fe(tr); FEATS=[c for c in X.columns if c not in["id","class"]]
Xtr,Xval,ytr,yval=train_test_split(X[FEATS],y,test_size=0.2,stratify=y,random_state=42)
log("data ready",Xtr.shape)

def tune(P):
    rng=np.random.default_rng(0); bw=np.ones(3); bs=balanced_accuracy_score(yval,P.argmax(1))
    for _ in range(4000):
        w=np.exp(rng.uniform(np.log(0.3),np.log(3),3)); s=balanced_accuracy_score(yval,(P*w).argmax(1))
        if s>bs: bs,bw=s,w
    return bs

# CatBoost
cb=CatBoostClassifier(loss_function="MultiClass",iterations=4000,learning_rate=0.04,depth=8,
    l2_leaf_reg=4,random_seed=42,auto_class_weights="Balanced",thread_count=-1,verbose=0,
    eval_metric="MultiClass",early_stopping_rounds=120)
cat_idx=[FEATS.index(c) for c in CAT]
cb.fit(Xtr,ytr,cat_features=cat_idx,eval_set=(Xval,yval),verbose=0)
Pcb=cb.predict_proba(Xval)
log("CatBoost  argmax",round(balanced_accuracy_score(yval,Pcb.argmax(1)),5),"| tuned",round(tune(Pcb),5))

# LGBM no class weight
lg=lgb.LGBMClassifier(objective="multiclass",num_class=3,n_estimators=4000,learning_rate=0.03,
    num_leaves=127,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=2,reg_alpha=0.5,
    min_child_samples=60,random_state=42,n_jobs=-1,verbose=-1)
lg.fit(Xtr,ytr,eval_set=[(Xval,yval)],eval_metric="multi_logloss",
    callbacks=[lgb.early_stopping(120,verbose=False)],categorical_feature=CAT)
Plg=lg.predict_proba(Xval)
log("LGBM noCW argmax",round(balanced_accuracy_score(yval,Plg.argmax(1)),5),"| tuned",round(tune(Plg),5))

# blend cb+lgb
Pbl=(Pcb+Plg)/2
log("CB+LGB    argmax",round(balanced_accuracy_score(yval,Pbl.argmax(1)),5),"| tuned",round(tune(Pbl),5))
