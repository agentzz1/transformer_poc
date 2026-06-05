"""Trick-hunt step 1: leakage / duplicate analysis + error analysis.
 - exact & rounded duplicate rows train<->test (free labels?)
 - duplicate rows within train: are labels consistent? (label noise / determinism)
 - confusion structure of our OOF (where does the ensemble struggle?)
"""
import numpy as np, pandas as pd
tr=pd.read_csv("data/train.csv"); te=pd.read_csv("data/test.csv")
FEAT=["alpha","delta","u","g","r","i","z","redshift","spectral_type","galaxy_population"]
NUM=["alpha","delta","u","g","r","i","z","redshift"]

print("=== exact duplicate feature-rows WITHIN train ===")
d=tr.duplicated(subset=FEAT,keep=False)
print("rows that are exact dups:",d.sum())
if d.sum():
    g=tr[d].groupby(FEAT)["class"].nunique()
    print("dup-groups:",len(g),"| groups with >1 class label:",(g>1).sum())

print("\n=== exact train<->test matches on ALL features ===")
key=FEAT
trk=tr[key].astype(str).agg("|".join,axis=1)
tek=te[key].astype(str).agg("|".join,axis=1)
lut=tr.assign(k=trk).groupby("k")["class"].agg(lambda s:s.value_counts().index[0])
hit=tek.map(lut)
print("test rows with exact train match:",hit.notna().sum(),f"({hit.notna().mean()*100:.2f}%)")

print("\n=== rounded match (photometry to 2 dp, redshift 3 dp) ===")
def rkey(df):
    a=df[["u","g","r","i","z"]].round(2).astype(str).agg("|".join,axis=1)
    b=df["redshift"].round(3).astype(str)
    c=df[["spectral_type","galaxy_population"]].astype(str).agg("|".join,axis=1)
    return a+"|"+b+"|"+c
trr=rkey(tr); ter=rkey(te)
lut2=tr.assign(k=trr).groupby("k")["class"].agg(lambda s:s.value_counts().index[0])
hit2=ter.map(lut2)
print("test rows with rounded train match:",hit2.notna().sum(),f"({hit2.notna().mean()*100:.2f}%)")
# purity of rounded groups in train
cons=tr.assign(k=trr).groupby("k")["class"].nunique()
print("rounded train groups:",len(cons),"| multi-class groups:",(cons>1).sum(),f"({(cons>1).mean()*100:.2f}%)")

print("\n=== OOF confusion (our blend, tuned) ===")
import json
CLASSES=["GALAXY","QSO","STAR"]; c2i={c:i for i,c in enumerate(CLASSES)}
y=tr["class"].map(c2i).values
oof=np.load("kaggle_s6e6/oof_blend.npy"); w=np.array(json.load(open("kaggle_s6e6/meta.json"))["weights"])
pred=(oof*w).argmax(1)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y,pred)
print("rows=true, cols=pred  [GALAXY,QSO,STAR]")
print(pd.DataFrame(cm,index=CLASSES,columns=CLASSES))
# where are errors concentrated by redshift?
err=pred!=y
print("\nerror rate by true class:",{CLASSES[i]:round(err[y==i].mean(),4) for i in range(3)})
print("median redshift of ERRORS vs correct:",round(tr["redshift"][err].median(),3),"vs",round(tr["redshift"][~err].median(),3))
