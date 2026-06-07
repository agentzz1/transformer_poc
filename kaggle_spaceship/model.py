"""spaceship-titanic: strong FE + LGBM/CatBoost/HGB ensemble, StratifiedKFold CV, submit.
Metric: accuracy. Goal: beat existing 0.803 CatBoost baseline.
"""
import numpy as np, pandas as pd, time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
t0=time.time(); log=lambda *a: print(f"[{time.time()-t0:6.1f}s]",*a,flush=True)
D='data/spaceship'
tr=pd.read_csv(f'{D}/train.csv'); te=pd.read_csv(f'{D}/test.csv')
SPEND=['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']

def fe(df):
    df=df.copy()
    df['Group']=df['PassengerId'].str.split('_').str[0]
    df['Member']=df['PassengerId'].str.split('_').str[1].astype(int)
    gs=df.groupby('Group')['PassengerId'].transform('count'); df['GroupSize']=gs; df['Alone']=(gs==1).astype(int)
    cab=df['Cabin'].str.split('/',expand=True); df['Deck']=cab[0]; df['CabinNum']=pd.to_numeric(cab[1],errors='coerce'); df['Side']=cab[2]
    df['CabinRegion']=(df['CabinNum']//300).astype('float')
    for c in SPEND: df[c]=df[c].fillna(0.0)
    df['TotalSpend']=df[SPEND].sum(axis=1); df['NoSpend']=(df['TotalSpend']==0).astype(int)
    df['nSpendCats']=(df[SPEND]>0).sum(axis=1)
    df['SpendLux']=df[['Spa','VRDeck','RoomService']].sum(axis=1); df['SpendBasic']=df[['FoodCourt','ShoppingMall']].sum(axis=1)
    # CryoSleep logic: spenders are awake
    df['CryoSleep']=df['CryoSleep'].astype('object')
    df.loc[df['CryoSleep'].isna() & (df['TotalSpend']>0),'CryoSleep']=False
    df['CryoSleep']=df['CryoSleep'].fillna(True).astype(int)
    df['VIP']=df['VIP'].fillna(False).astype(int)
    df['Age']=df['Age'].fillna(df['Age'].median()); df['IsChild']=(df['Age']<13).astype(int)
    df['Surname']=df['Name'].str.split().str[-1]
    fam=df.groupby('Surname')['PassengerId'].transform('count'); df['FamilySize']=fam.fillna(1)
    df['logTotalSpend']=np.log1p(df['TotalSpend'])
    for c in ['HomePlanet','Destination','Deck','Side']:
        df[c]=df[c].fillna('NA').astype('category')
    return df

both=fe(pd.concat([tr.assign(_s=0),te.assign(_s=1)],ignore_index=True))
tr2=both[both['_s']==0].copy(); te2=both[both['_s']==1].copy()
y=tr['Transported'].astype(int).values
FEAT=['HomePlanet','CryoSleep','Destination','Age','VIP','Member','GroupSize','Alone',
      'Deck','CabinNum','Side','CabinRegion','IsChild','FamilySize',
      *SPEND,'TotalSpend','NoSpend','nSpendCats','SpendLux','SpendBasic','logTotalSpend']
CAT=['HomePlanet','Destination','Deck','Side']
Xtr=tr2[FEAT]; Xte=te2[FEAT]
oh_tr=pd.get_dummies(Xtr,columns=CAT); oh_te=pd.get_dummies(Xte,columns=CAT).reindex(columns=oh_tr.columns,fill_value=0)

skf=StratifiedKFold(5,shuffle=True,random_state=42)
oof=np.zeros((len(tr2),3)); pte=np.zeros((len(te2),3))
for f,(tri,vai) in enumerate(skf.split(Xtr,y)):
    # lgbm
    m=lgb.LGBMClassifier(n_estimators=2000,learning_rate=0.02,num_leaves=31,subsample=0.8,subsample_freq=1,
        colsample_bytree=0.7,reg_lambda=3,min_child_samples=40,random_state=42,n_jobs=-1,verbose=-1)
    m.fit(Xtr.iloc[tri],y[tri],eval_set=[(Xtr.iloc[vai],y[vai])],eval_metric='binary_error',
          callbacks=[lgb.early_stopping(100,verbose=False)],categorical_feature=CAT)
    oof[vai,0]=m.predict_proba(Xtr.iloc[vai])[:,1]; pte[:,0]+=m.predict_proba(Xte)[:,1]/5
    # catboost
    cb=CatBoostClassifier(iterations=2000,learning_rate=0.03,depth=6,l2_leaf_reg=5,random_seed=42,
        verbose=0,early_stopping_rounds=100,cat_features=[FEAT.index(c) for c in CAT])
    cb.fit(Xtr.iloc[tri],y[tri],eval_set=(Xtr.iloc[vai],y[vai]))
    oof[vai,1]=cb.predict_proba(Xtr.iloc[vai])[:,1]; pte[:,1]+=cb.predict_proba(Xte)[:,1]/5
    # hgb (one-hot)
    h=HistGradientBoostingClassifier(max_iter=800,learning_rate=0.05,max_leaf_nodes=31,l2_regularization=2,
        min_samples_leaf=40,early_stopping=True,n_iter_no_change=60,random_state=42)
    h.fit(oh_tr.iloc[tri],y[tri]); oof[vai,2]=h.predict_proba(oh_tr.iloc[vai])[:,1]; pte[:,2]+=h.predict_proba(oh_te)[:,1]/5
    log(f"fold {f} acc: lgbm={accuracy_score(y[vai],oof[vai,0]>.5):.4f} cat={accuracy_score(y[vai],oof[vai,1]>.5):.4f} hgb={accuracy_score(y[vai],oof[vai,2]>.5):.4f}")

for j,nm in enumerate(['lgbm','catboost','hgb']):
    log(f"OOF acc {nm}: {accuracy_score(y,oof[:,j]>.5):.4f}")
blend=oof.mean(1); log(f"OOF acc BLEND: {accuracy_score(y,blend>.5):.4f}")
# tune threshold
ths=np.linspace(0.4,0.6,41); bt=max(ths,key=lambda t:accuracy_score(y,blend>t))
log(f"best threshold {bt:.3f} acc {accuracy_score(y,blend>bt):.4f}")
pred=(pte.mean(1)>bt)
sub=pd.read_csv(f'{D}/sample_submission.csv'); sub['Transported']=pred
sub.to_csv('kaggle_spaceship/submission.csv',index=False)
log("wrote submission.csv", sub['Transported'].mean().round(3))
