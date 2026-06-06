"""Generate the Kaggle GPU kernel notebook + metadata for a strong AutoGluon model."""
import json, nbformat as nbf

cells = []
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))

md("# PS-S6E6 — strong model (AutoGluon, GPU)\nFE + AutoGluon best_quality (eval=balanced_accuracy) + balanced-accuracy threshold tuning, blended with the 0.97079 vote.")

code(r"""
import subprocess, sys
subprocess.run([sys.executable,"-m","pip","install","-q","autogluon.tabular[lightgbm,catboost,xgboost,fastai]"], check=False)
""")

code(r"""
import glob, os, numpy as np, pandas as pd
print('inputs:', os.listdir('/kaggle/input'))
# locate competition data robustly (mount path can vary)
tr_paths = glob.glob('/kaggle/input/**/train.csv', recursive=True)
te_paths = glob.glob('/kaggle/input/**/test.csv', recursive=True)
print('train.csv candidates:', tr_paths)
COMP = os.path.dirname(tr_paths[0])
tr = pd.read_csv(f'{COMP}/train.csv'); te = pd.read_csv(f'{COMP}/test.csv')
print('train', tr.shape, 'test', te.shape, '| COMP=', COMP)
CLASSES=['GALAXY','QSO','STAR']; CAT=['spectral_type','galaxy_population']; MAGS=['u','g','r','i','z']
def fe(df):
    df=df.copy()
    for a,b in [('u','g'),('g','r'),('r','i'),('i','z'),('u','r'),('g','i'),('u','z'),('r','z')]:
        df[f'{a}_{b}']=df[a]-df[b]
    df['mag_mean']=df[MAGS].mean(axis=1); df['mag_std']=df[MAGS].std(axis=1)
    df['mag_range']=df[MAGS].max(axis=1)-df[MAGS].min(axis=1)
    df['zr_u_g']=df['redshift']*df['u_g']; df['zr_g_r']=df['redshift']*df['g_r']
    df['zr_r_i']=df['redshift']*df['r_i']; df['z_x_r']=df['redshift']*df['r']
    return df
trf=fe(tr); tef=fe(te)
FEATS=[c for c in trf.columns if c not in ['id','class']]
train_data=trf[FEATS+['class']]; test_data=tef[FEATS]
""")

code(r"""
from autogluon.tabular import TabularPredictor
predictor = TabularPredictor(label='class', eval_metric='balanced_accuracy', path='ag_s6e6')
predictor.fit(train_data, presets='best_quality', time_limit=5400,
              num_bag_folds=5, num_stack_levels=1, ag_args_fit={'num_gpus':1})
print(predictor.leaderboard(silent=True).head(15))
""")

code(r"""
from sklearn.metrics import balanced_accuracy_score, classification_report
y = train_data['class'].map({c:i for i,c in enumerate(CLASSES)}).values
oof_df = predictor.predict_proba_oof()[CLASSES]      # align columns
test_df = predictor.predict_proba(test_data)[CLASSES]
oof = oof_df.values; ptest = test_df.values
print('AG OOF bal-acc (argmax):', round(balanced_accuracy_score(y, oof.argmax(1)),5))

# balanced-accuracy class-weight tuning on OOF
rng=np.random.default_rng(42); bw=np.ones(3); bs=balanced_accuracy_score(y,oof.argmax(1))
for _ in range(8000):
    w=np.exp(rng.uniform(np.log(0.3),np.log(3),3)); s=balanced_accuracy_score(y,(oof*w).argmax(1))
    if s>bs: bs,bw=s,w
for _ in range(50):
    for k in range(3):
        for d in (0.97,1.03,0.99,1.01):
            w=bw.copy(); w[k]*=d; s=balanced_accuracy_score(y,(oof*w).argmax(1))
            if s>bs: bs,bw=s,w
print('AG OOF bal-acc (tuned):', round(bs,5), '| weights', dict(zip(CLASSES,bw.round(4))))
print(classification_report(y,(oof*bw).argmax(1),target_names=CLASSES,digits=4))
""")

code(r"""
# outputs: AG-only (tuned) + blends with the existing 0.97079 vote
sub = pd.read_csv(f'{COMP}/sample_submission.csv')
def write(pred, name):
    s=sub.copy(); s['class']=[CLASSES[i] for i in pred]; s.to_csv(name,index=False)
    print(name, pd.Series(s['class']).value_counts(normalize=True).round(4).to_dict())
write((ptest*bw).argmax(1), 'submission_ag.csv')
np.save('oof_ag.npy', oof); np.save('test_ag.npy', ptest)

# blend with 0.97079 vote if available
import glob
cand = glob.glob('/kaggle/input/ps-s6e6/0.97079.csv') + glob.glob('/kaggle/input/**/0.97079.csv', recursive=True)
if cand:
    v = pd.read_csv(cand[0]).set_index('id').reindex(sub['id'])['class'].map({c:i for i,c in enumerate(CLASSES)}).values
    onehot = np.eye(3)[v]
    for a in [0.5,0.6,0.7]:
        P = a*onehot + (1-a)*ptest
        write((P*bw).argmax(1), f'submission_blend_a{a:.1f}.csv')
else:
    print('0.97079.csv not found in inputs; skipped blend')
print('done')
""")

nb = nbf.v4.new_notebook(); nb['cells']=cells
nb['metadata'] = {
  "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
  "language_info": {"name": "python", "version": "3.11"},
}
nbf.write(nb, 'kaggle_s6e6/push/ps-s6e6-ag-gpu.ipynb')

meta = {
  "id": "agentzz/ps-s6e6-ag-gpu",
  "title": "ps-s6e6 ag gpu",
  "code_file": "ps-s6e6-ag-gpu.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": True,
  "enable_gpu": True,
  "enable_tpu": False,
  "enable_internet": True,
  "dataset_sources": ["nina2025/ps-s6e6"],
  "competition_sources": ["playground-series-s6e6"],
  "kernel_sources": [],
}
json.dump(meta, open('kaggle_s6e6/push/kernel-metadata.json','w'), indent=2)
print("wrote notebook + metadata")
