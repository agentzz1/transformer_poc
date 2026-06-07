"""rogii: predict beyond-PS TVT drift (TVT - anchor) with LightGBM over all 773 wells.
Well-level GroupKFold CV; report pooled RMSE (competition-style) vs hold-last baseline.
"""
import glob, os, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
D = 'data/rogii_full'; t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

def robust_z(x):
    x = np.asarray(x, float); m = np.nanmedian(x); s = np.nanmedian(np.abs(x-m))*1.4826+1e-9
    return (x-m)/s

def feats_for_well(hpath, need_target=True):
    df = pd.read_csv(hpath)
    if need_target and 'TVT' not in df: return None
    if not df['TVT_input'].isna().any(): return None
    ps = int(df['TVT_input'].isna().values.argmax())
    if ps == 0 or ps >= len(df): return None
    wid = os.path.basename(hpath).split('__')[0]
    md = df['MD'].values.astype(float); X = df['X'].values.astype(float); Y = df['Y'].values.astype(float)
    Z = df['Z'].values.astype(float); g = df['GR'].values.astype(float)
    nan = np.isnan(g)
    if nan.all(): return None
    g[nan] = np.interp(md[nan], md[~nan], g[~nan]) if (~nan).any() else 0.0
    gz = robust_z(g)
    gsm = pd.Series(g).rolling(15, center=True, min_periods=1).mean().values
    ggrad = np.gradient(gsm, md)
    anchorZ = Z[ps-1]; anchorMD = md[ps-1]; anchorX = X[ps-1]; anchorY = Y[ps-1]
    # azimuth of drilling (from pre-PS direction)
    k = max(1, min(50, ps-1)); az = np.arctan2(Y[ps-1]-Y[ps-1-k], X[ps-1]-X[ps-1-k])
    sl = slice(ps, len(df))
    n = len(df) - ps
    horiz = np.sqrt((X[sl]-anchorX)**2 + (Y[sl]-anchorY)**2)
    f = pd.DataFrame({
        'md_since_ps': md[sl]-anchorMD,
        'dz_since_ps': Z[sl]-anchorZ,
        'horiz_dist': horiz,
        'z_abs': Z[sl],
        'gr': g[sl], 'gr_z': gz[sl], 'gr_smooth': gsm[sl], 'gr_grad': ggrad[sl],
        'gr_minus_anchor': g[sl]-g[ps-1],
        'az_sin': np.sin(az), 'az_cos': np.cos(az),
        'frac_along': np.arange(n)/n,
    })
    f['well'] = wid
    f['anchor'] = df['TVT_input'].iloc[ps-1]
    if need_target:
        f['target'] = df['TVT'].values[sl] - df['TVT_input'].iloc[ps-1]  # drift
        f['tvt_true'] = df['TVT'].values[sl]
    return f

if __name__ == '__main__':
    files = sorted(glob.glob(f'{D}/train/*__horizontal_well.csv'))
    parts = []
    for i, hp in enumerate(files):
        fr = feats_for_well(hp)
        if fr is not None: parts.append(fr)
        if i % 150 == 0: log(f"loaded {i}/{len(files)} wells")
    data = pd.concat(parts, ignore_index=True)
    log(f"rows={len(data)} wells={data['well'].nunique()}")
    FEAT = ['md_since_ps','dz_since_ps','horiz_dist','z_abs','gr','gr_z','gr_smooth','gr_grad',
            'gr_minus_anchor','az_sin','az_cos','frac_along']
    gkf = GroupKFold(n_splits=5); oof = np.zeros(len(data))
    for tr, va in gkf.split(data, groups=data['well']):
        m = lgb.LGBMRegressor(n_estimators=1200, learning_rate=0.03, num_leaves=127,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=3.0,
            min_child_samples=100, random_state=42, n_jobs=-1, verbose=-1)
        m.fit(data[FEAT].iloc[tr], data['target'].iloc[tr])
        oof[va] = m.predict(data[FEAT].iloc[va])
    pred_tvt = data['anchor'].values + oof
    pooled = np.sqrt(np.mean((pred_tvt - data['tvt_true'].values)**2))
    hold = np.sqrt(np.mean((data['anchor'].values - data['tvt_true'].values)**2))
    log(f"POOLED RMSE  GBM-drift = {pooled:.3f}   |  hold-last = {hold:.3f}")
    # per-well
    dfm = pd.DataFrame({'well':data['well'],'e_gbm':(pred_tvt-data['tvt_true'])**2,'e_hold':(data['anchor']-data['tvt_true'])**2})
    pw = dfm.groupby('well').mean().apply(np.sqrt)
    log(f"per-well mean RMSE  GBM={pw['e_gbm'].mean():.3f}  hold={pw['e_hold'].mean():.3f}  win={ (pw['e_gbm']<pw['e_hold']).mean():.2f}")
    import joblib; data.head(0).to_csv('kaggle_rogii/_featcols.csv',index=False)
