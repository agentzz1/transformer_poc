"""spaceship-titanic v3: v2 pipeline + pseudo-labeling of confident test rows + extra feats.
Honest CV: pseudo test rows added to each fold's TRAIN only; OOF measured on real labels.
"""
import numpy as np, pandas as pd, time
from sklearn.model_selection import StratifiedKFold, cross_val_predict
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
    for c in ['Deck','Side','CabinNum']: df[c]=df.groupby('Group')[c].transform(lambda s: s.ffill().bfill())
    df['HomePlanet']=df.groupby('Group')['HomePlanet'].transform(lambda s: s.ffill().bfill())
    df['HomePlanet']=df['HomePlanet'].fillna(df['Deck'].map(DECK_HP)).fillna('Earth')
    df['Destination']=df.groupby('Group')['Destination'].transform(lambda s: s.ffill().bfill()).fillna('TRAPPIST-1e')
    gs=df.groupby('Group')['PassengerId'].transform('count'); df['GroupSize']=gs; df['Alone']=(gs==1).astype(int)
    df['CabinRegion']=(df['CabinNum']//300)
    for c in SPEND: df[c]=df[c].fillna(0.0)
    df['TotalSpend']=df[SPEND].sum(axis=1); df['NoSpend']=(df['TotalSpend']==0).astype(int)
    df['nSpendCats']=(df[SPEND]>0).sum(axis=1)
    df['SpendLux']=df[['Spa','VRDeck','RoomService']].sum(axis=1); df['SpendBasic']=df[['FoodCourt','ShoppingMall']].sum(axis=1)
    df['logTotalSpend']=np.log1p(df['TotalSpend'])
    df['GroupSpend']=df.groupby('Group')['TotalSpend'].transform('sum')
    df['GroupNoSpend']=df.groupby('Group')['NoSpend'].transform('mean')
    df['SpendPerPerson']=df['GroupSpend']/df['GroupSize']
    df['CryoSleep']=df['CryoSleep'].astype('object')
    df.loc[df['CryoSleep'].isna() & (df['TotalSpend']>0),'CryoSleep']=False
    df['CryoSleep']=df['CryoSleep'].fillna(True).astype(int)
    df['VIP']=df['VIP'].fillna(False).astype(int)
    df['Age']=df.groupby('HomePlanet')['Age'].transform(lambda s: s.fillna(s.median())).fillna(27)
    df['IsChild']=(df['Age']<13).astype(int); df['AgeBin']=pd.cut(df['Age'],[-1,12,18,25,35,50,200],labels=False)
    df['CryoNoSpend']=((df['CryoSleep']==1)&(df['NoSpend']==1)).astype(int)
    df['Surname']=df['Name'].str.split().str[-1]
    df['FamilySize']=df.groupby('Surname')['PassengerId'].transform('count').fillna(1)
    df['CabinNum']=df['CabinNum'].fillna(-1); df['CabinRegion']=df['CabinRegion'].fillna(-1)
    df['Side']=df['Side'].fillna('NA'); df['Deck']=df['Deck'].fillna('NA')
    for c in ['HomePlanet','Destination','Deck','Side']: df[c]=df[c].astype('category')
    return df

both=fe(pd.concat([tr.assign(_s=0),te.assign(_s=1)],ignore_index=True))
tr2=both[both['_s']==0].reset_index(drop=True); te2=both[both['_s']==1].reset_index(drop=True)
y=tr['Transported'].astype(int).values
FEAT=['HomePlanet','CryoSleep','Destination','Age','AgeBin','VIP','Member','GroupSize','Alone',
      'Deck','CabinNum','Side','CabinRegion','IsChild','FamilySize','CryoNoSpend',
      *SPEND,'TotalSpend','NoSpend','nSpendCats','SpendLux','SpendBasic','logTotalSpend','GroupSpend','GroupNoSpend','SpendPerPerson']
CAT=['HomePlanet','Destination','Deck','Side']
catidx=[FEAT.index(c) for c in CAT]

def base_models():
    return [
        ('lgbm', lgb.LGBMClassifier(n_estimators=1500,learning_rate=0.02,num_leaves=31,subsample=0.8,subsample_freq=1,
            colsample_bytree=0.7,reg_lambda=3,min_child_samples=40,random_state=1,n_jobs=-1,verbose=-1),'cat'),
        ('cat', CatBoostClassifier(iterations=1500,learning_rate=0.03,depth=6,l2_leaf_reg=5,random_seed=1,verbose=0),'cat'),
        ('hgb', HistGradientBoostingClassifier(max_iter=700,learning_rate=0.05,max_leaf_nodes=31,l2_regularization=2,
            min_samples_leaf=40,random_state=1),'oh'),
        ('xgb', xgb.XGBClassifier(n_estimators=1500,learning_rate=0.02,max_depth=5,subsample=0.8,colsample_bytree=0.7,
            reg_lambda=3,min_child_weight=5,random_state=1,n_jobs=-1,tree_method='hist',eval_metric='logloss'),'oh'),
    ]

Xtr=tr2[FEAT]; Xte=te2[FEAT]
oh_tr=pd.get_dummies(Xtr,columns=CAT); oh_te=pd.get_dummies(Xte,columns=CAT).reindex(columns=oh_tr.columns,fill_value=0)

def run_cv(pseudo_X=None, pseudo_y=None, pseudo_oh=None):
    skf=StratifiedKFold(5,shuffle=True,random_state=42); M=4
    oof=np.zeros((len(tr2),M)); pte=np.zeros((len(te2),M))
    for tri,vai in skf.split(Xtr,y):
        for k,(nm,mdl,kind) in enumerate(base_models()):
            if kind=='cat':
                Xt=Xtr.iloc[tri]; yt=y[tri]
                if pseudo_X is not None: Xt=pd.concat([Xt,pseudo_X],ignore_index=True); yt=np.concatenate([yt,pseudo_y])
                if nm=='cat': mdl.fit(Xt,yt,cat_features=catidx)
                else: mdl.fit(Xt,yt,categorical_feature=CAT)
                oof[vai,k]=mdl.predict_proba(Xtr.iloc[vai])[:,1]; pte[:,k]+=mdl.predict_proba(Xte)[:,1]/5
            else:
                Xt=oh_tr.iloc[tri]; yt=y[tri]
                if pseudo_oh is not None: Xt=pd.concat([Xt,pseudo_oh],ignore_index=True); yt=np.concatenate([yt,pseudo_y])
                mdl.fit(Xt,yt); oof[vai,k]=mdl.predict_proba(oh_tr.iloc[vai])[:,1]; pte[:,k]+=mdl.predict_proba(oh_te)[:,1]/5
    return oof,pte

# round 1 (no pseudo)
oof,pte=run_cv();
skf=StratifiedKFold(5,shuffle=True,random_state=42)
st=LogisticRegression(max_iter=1000)
soof=cross_val_predict(st,oof,y,cv=skf,method='predict_proba')[:,1]
log(f"v3 round1 OOF stack acc: {accuracy_score(y,soof>.5):.4f}")
st.fit(oof,y); ste=st.predict_proba(pte)[:,1]
# pseudo-label confident test
conf=(ste>0.95)|(ste<0.05); log(f"confident test rows: {conf.sum()} / {len(te2)}")
pX=Xte[conf].reset_index(drop=True); pOH=oh_te[conf].reset_index(drop=True); py=(ste[conf]>0.5).astype(int)
# round 2 (with pseudo)
oof2,pte2=run_cv(pX,py,pOH)
soof2=cross_val_predict(st,oof2,y,cv=skf,method='predict_proba')[:,1]
acc2=accuracy_score(y,soof2>.5); log(f"v3 round2 (pseudo) OOF stack acc: {acc2:.4f}")
st.fit(oof2,y); ste2=st.predict_proba(pte2)[:,1]
bt=max(np.linspace(0.42,0.58,33),key=lambda t:accuracy_score(y,soof2>t))
log(f"best th {bt:.3f} acc {accuracy_score(y,soof2>bt):.4f}")
sub=pd.read_csv(f'{D}/sample_submission.csv'); sub['Transported']=(ste2>bt)
sub.to_csv('kaggle_spaceship/submission_v3.csv',index=False); log("wrote v3",round(sub['Transported'].mean(),3))
