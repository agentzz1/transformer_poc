"""spaceship-titanic v2: smart imputation (Group/Deck->HomePlanet, Group->Cabin),
group aggregates, 4 base models + LogReg stacker. Submit.
"""
import numpy as np, pandas as pd, time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
t0=time.time(); log=lambda *a: print(f"[{time.time()-t0:6.1f}s]",*a,flush=True)
D='data/spaceship'
tr=pd.read_csv(f'{D}/train.csv'); te=pd.read_csv(f'{D}/test.csv')
SPEND=['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
DECK_HP={'A':'Europa','B':'Europa','C':'Europa','T':'Europa','G':'Earth'}

def fe(df):
    df=df.copy()
    df['Group']=df['PassengerId'].str.split('_').str[0]
    df['Member']=df['PassengerId'].str.split('_').str[1].astype(int)
    cab=df['Cabin'].str.split('/',expand=True); df['Deck']=cab[0]; df['CabinNum']=pd.to_numeric(cab[1],errors='coerce'); df['Side']=cab[2]
    # impute Cabin parts within group
    for c in ['Deck','Side','CabinNum']:
        df[c]=df.groupby('Group')[c].transform(lambda s: s.ffill().bfill())
    # impute HomePlanet: group -> deck-map -> global
    df['HomePlanet']=df.groupby('Group')['HomePlanet'].transform(lambda s: s.ffill().bfill())
    df['HomePlanet']=df['HomePlanet'].fillna(df['Deck'].map(DECK_HP))
    df['HomePlanet']=df['HomePlanet'].fillna('Earth')
    df['Destination']=df.groupby('Group')['Destination'].transform(lambda s: s.ffill().bfill()).fillna('TRAPPIST-1e')
    gs=df.groupby('Group')['PassengerId'].transform('count'); df['GroupSize']=gs; df['Alone']=(gs==1).astype(int)
    df['CabinRegion']=(df['CabinNum']//300)
    for c in SPEND: df[c]=df[c].fillna(0.0)
    df['TotalSpend']=df[SPEND].sum(axis=1); df['NoSpend']=(df['TotalSpend']==0).astype(int)
    df['nSpendCats']=(df[SPEND]>0).sum(axis=1)
    df['SpendLux']=df[['Spa','VRDeck','RoomService']].sum(axis=1); df['SpendBasic']=df[['FoodCourt','ShoppingMall']].sum(axis=1)
    df['logTotalSpend']=np.log1p(df['TotalSpend'])
    # group spend aggregates
    df['GroupSpend']=df.groupby('Group')['TotalSpend'].transform('sum')
    df['GroupNoSpend']=df.groupby('Group')['NoSpend'].transform('mean')
    df['CryoSleep']=df['CryoSleep'].astype('object')
    df.loc[df['CryoSleep'].isna() & (df['TotalSpend']>0),'CryoSleep']=False
    df['CryoSleep']=df['CryoSleep'].fillna(True).astype(int)
    df['VIP']=df['VIP'].fillna(False).astype(int)
    df['Age']=df.groupby('HomePlanet')['Age'].transform(lambda s: s.fillna(s.median())); df['Age']=df['Age'].fillna(27)
    df['IsChild']=(df['Age']<13).astype(int)
    df['Surname']=df['Name'].str.split().str[-1]
    df['FamilySize']=df.groupby('Surname')['PassengerId'].transform('count').fillna(1)
    df['CabinNum']=df['CabinNum'].fillna(-1); df['CabinRegion']=df['CabinRegion'].fillna(-1)
    df['Side']=df['Side'].fillna('NA'); df['Deck']=df['Deck'].fillna('NA')
    for c in ['HomePlanet','Destination','Deck','Side']: df[c]=df[c].astype('category')
    return df

both=fe(pd.concat([tr.assign(_s=0),te.assign(_s=1)],ignore_index=True))
tr2=both[both['_s']==0].copy(); te2=both[both['_s']==1].copy()
y=tr['Transported'].astype(int).values
FEAT=['HomePlanet','CryoSleep','Destination','Age','VIP','Member','GroupSize','Alone',
      'Deck','CabinNum','Side','CabinRegion','IsChild','FamilySize',
      *SPEND,'TotalSpend','NoSpend','nSpendCats','SpendLux','SpendBasic','logTotalSpend','GroupSpend','GroupNoSpend']
CAT=['HomePlanet','Destination','Deck','Side']
Xtr=tr2[FEAT]; Xte=te2[FEAT]
oh_tr=pd.get_dummies(Xtr,columns=CAT); oh_te=pd.get_dummies(Xte,columns=CAT).reindex(columns=oh_tr.columns,fill_value=0)

skf=StratifiedKFold(5,shuffle=True,random_state=42)
M=4; oof=np.zeros((len(tr2),M)); pte=np.zeros((len(te2),M))
for f,(tri,vai) in enumerate(skf.split(Xtr,y)):
    m=lgb.LGBMClassifier(n_estimators=2500,learning_rate=0.02,num_leaves=31,subsample=0.8,subsample_freq=1,
        colsample_bytree=0.7,reg_lambda=3,min_child_samples=40,random_state=42,n_jobs=-1,verbose=-1)
    m.fit(Xtr.iloc[tri],y[tri],eval_set=[(Xtr.iloc[vai],y[vai])],eval_metric='binary_error',
          callbacks=[lgb.early_stopping(100,verbose=False)],categorical_feature=CAT)
    oof[vai,0]=m.predict_proba(Xtr.iloc[vai])[:,1]; pte[:,0]+=m.predict_proba(Xte)[:,1]/5
    cb=CatBoostClassifier(iterations=2500,learning_rate=0.03,depth=6,l2_leaf_reg=5,random_seed=42,
        verbose=0,early_stopping_rounds=100,cat_features=[FEAT.index(c) for c in CAT])
    cb.fit(Xtr.iloc[tri],y[tri],eval_set=(Xtr.iloc[vai],y[vai]))
    oof[vai,1]=cb.predict_proba(Xtr.iloc[vai])[:,1]; pte[:,1]+=cb.predict_proba(Xte)[:,1]/5
    h=HistGradientBoostingClassifier(max_iter=900,learning_rate=0.05,max_leaf_nodes=31,l2_regularization=2,
        min_samples_leaf=40,early_stopping=True,n_iter_no_change=60,random_state=42)
    h.fit(oh_tr.iloc[tri],y[tri]); oof[vai,2]=h.predict_proba(oh_tr.iloc[vai])[:,1]; pte[:,2]+=h.predict_proba(oh_te)[:,1]/5
    xg=xgb.XGBClassifier(n_estimators=2500,learning_rate=0.02,max_depth=5,subsample=0.8,colsample_bytree=0.7,
        reg_lambda=3,min_child_weight=5,eval_metric='error',early_stopping_rounds=100,random_state=42,n_jobs=-1,tree_method='hist')
    xg.fit(oh_tr.iloc[tri],y[tri],eval_set=[(oh_tr.iloc[vai],y[vai])],verbose=False)
    oof[vai,3]=xg.predict_proba(oh_tr.iloc[vai])[:,1]; pte[:,3]+=xg.predict_proba(oh_te)[:,1]/5
    log(f"fold {f} done")

for j,nm in enumerate(['lgbm','cat','hgb','xgb']): log(f"OOF acc {nm}: {accuracy_score(y,oof[:,j]>.5):.4f}")
log(f"OOF acc mean-blend: {accuracy_score(y,oof.mean(1)>.5):.4f}")
# LogReg stacker
st=LogisticRegression(max_iter=1000,C=1.0)
from sklearn.model_selection import cross_val_predict
stack_oof=cross_val_predict(st,oof,y,cv=skf,method='predict_proba')[:,1]
log(f"OOF acc STACK(logreg): {accuracy_score(y,stack_oof>.5):.4f}")
st.fit(oof,y); stack_te=st.predict_proba(pte)[:,1]
ths=np.linspace(0.42,0.58,33); bt=max(ths,key=lambda t:accuracy_score(y,stack_oof>t))
log(f"stack best th {bt:.3f} acc {accuracy_score(y,stack_oof>bt):.4f}")
sub=pd.read_csv(f'{D}/sample_submission.csv'); sub['Transported']=(stack_te>bt)
sub.to_csv('kaggle_spaceship/submission_v2.csv',index=False); log("wrote v2", round(sub['Transported'].mean(),3))
