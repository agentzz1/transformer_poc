"""rogii per-well geosteering v2: calibrate horizontal GR to the typewell using the KNOWN
pre-PS (TVT,GR) pairs, then Viterbi-track TVT beyond PS with strong smoothness + anchor.
Validate pooled RMSE vs hold-last on train wells; grid-search lambda.
"""
import glob, os, time, numpy as np, pandas as pd
D='data/rogii_full'; t0=time.time()
log=lambda *a: print(f"[{time.time()-t0:6.1f}s]",*a,flush=True)

def load(hp):
    df=pd.read_csv(hp); wid=os.path.basename(hp).split('__')[0]
    tw=glob.glob(os.path.join(os.path.dirname(hp),f'{wid}__typewell*.csv'))
    return df,(pd.read_csv(tw[0]) if tw else None)

def predict(df,tw,grid_half=70.0,dz=0.5,band_ft=6.0,lam=0.2,anchor_mu=0.002,smooth=9):
    n=len(df)
    if not df['TVT_input'].isna().any(): return None,n
    ps=int(df['TVT_input'].isna().values.argmax())
    if ps<20 or ps>=n or tw is None:
        a=df['TVT_input'].dropna().iloc[-1] if df['TVT_input'].notna().any() else 0.0
        return np.full(n-ps,a),ps
    md=df['MD'].values.astype(float); g=df['GR'].values.astype(float)
    nan=np.isnan(g);
    if nan.all(): return np.full(n-ps,df['TVT_input'].iloc[ps-1]),ps
    g[nan]=np.interp(md[nan],md[~nan],g[~nan]) if (~nan).any() else 0.0
    g=pd.Series(g).rolling(smooth,center=True,min_periods=1).mean().values
    anchor=float(df['TVT_input'].iloc[ps-1])
    # typewell profile
    t=tw.dropna(subset=['TVT','GR']).sort_values('TVT')
    tvt_tw=t['TVT'].values.astype(float); gr_tw=t['GR'].values.astype(float)
    gr_tw=pd.Series(gr_tw).rolling(smooth,center=True,min_periods=1).mean().values
    # calibrate horiz GR -> typewell GR space using pre-PS known TVT
    tvt_pre=df['TVT_input'].values[:ps].astype(float); g_pre=g[:ps]
    gr_tw_at_pre=np.interp(tvt_pre,tvt_tw,gr_tw)
    A=np.vstack([gr_tw_at_pre,np.ones(ps)]).T
    try: a_,b_=np.linalg.lstsq(A,g_pre,rcond=None)[0]
    except Exception: a_,b_=1.0,0.0
    if abs(a_)<1e-6: a_=1.0
    g_cal=(g-b_)/a_   # horiz GR in typewell-GR units
    lo=max(anchor-grid_half,tvt_tw.min()); hi=min(anchor+grid_half,tvt_tw.max())
    if hi-lo<dz: return np.full(n-ps,anchor),ps
    grid=np.arange(lo,hi+dz,dz); G=len(grid)
    gr_grid=np.interp(grid,tvt_tw,gr_tw)
    gz=g_cal[ps:]; L=len(gz)
    band=int(round(band_ft/dz)); offs=np.arange(-band,band+1); tp=lam*(offs*dz)**2
    apen=anchor_mu*(grid-anchor)**2
    cost=((grid-anchor)/2.0)**2+(gz[0]-gr_grid)**2+apen
    bptr=np.empty((L,G),np.int32)
    for i in range(1,L):
        best=np.full(G,1e18); arg=np.zeros(G,np.int32)
        for o,p in zip(offs,tp):
            src=np.full(G,1e18)
            if o>0: src[o:]=cost[:G-o]+p
            elif o<0: src[:G+o]=cost[-o:]+p
            else: src=cost+p
            u=src<best; best[u]=src[u]; arg[u]=(np.arange(G)-o)[u]
        cost=best+(gz[i]-gr_grid)**2+apen; bptr[i]=arg
    s=int(np.argmin(cost)); path=np.empty(L,np.int32); path[-1]=s
    for i in range(L-1,0,-1): s=bptr[i,s]; path[i-1]=s
    return grid[path],ps

if __name__=='__main__':
    import sys
    lam=float(sys.argv[1]) if len(sys.argv)>1 else 0.2
    files=sorted(glob.glob(f'{D}/train/*__horizontal_well.csv'))
    rng=np.random.default_rng(1); samp=rng.choice(len(files),80,replace=False)
    se_v=[]; se_h=[]
    for j in samp:
        df,tw=load(files[j])
        if 'TVT' not in df or not df['TVT_input'].isna().any(): continue
        pred,ps=predict(df,tw,lam=lam)
        y=df['TVT'].values[ps:]
        if len(y)==0 or pred is None: continue
        se_v.append((y-pred)**2); se_h.append((y-df['TVT'].iloc[ps-1])**2)
    se_v=np.concatenate(se_v); se_h=np.concatenate(se_h)
    log(f"lam={lam}  POOLED RMSE  viterbi-v2={np.sqrt(se_v.mean()):.3f}   hold-last={np.sqrt(se_h.mean()):.3f}")
