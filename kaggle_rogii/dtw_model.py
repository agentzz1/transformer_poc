"""rogii Geosteering: predict TVT beyond PS by Viterbi-tracking the horizontal GR through
the typewell GR(TVT) profile, with a smoothness (dip) prior and anchor at the PS point.

Validated on train wells (true TVT known). Metric: RMSE(dTVT) over predicted (beyond-PS) points.
"""
import glob, os, numpy as np, pandas as pd
D = 'data/rogii_full'

def robust_z(x):
    x = np.asarray(x, float)
    m = np.nanmedian(x); s = np.nanmedian(np.abs(x - m)) * 1.4826 + 1e-9
    return (x - m) / s

def load_well(hpath):
    df = pd.read_csv(hpath)
    wid = os.path.basename(hpath).split('__')[0]
    tw = glob.glob(os.path.join(os.path.dirname(hpath), f'{wid}__typewell*.csv'))
    tw = pd.read_csv(tw[0]) if tw else None
    return df, tw

def predict_well(df, tw, grid_half=160.0, dz=1.0, band_ft=12.0, lam=0.015, smooth_tw=5):
    """Return predicted TVT array for beyond-PS rows (and the ps index)."""
    ps = int(df['TVT_input'].isna().values.argmax()) if df['TVT_input'].isna().any() else len(df)
    n = len(df)
    if ps == 0 or ps >= n or tw is None:
        # fallback: hold last known / TVT_input
        anchor = df['TVT_input'].dropna().iloc[-1] if df['TVT_input'].notna().any() else 0.0
        return np.full(n - ps, anchor), ps
    anchor = float(df['TVT_input'].iloc[ps - 1])
    # horizontal GR beyond PS (interp NaNs along MD)
    g = df['GR'].values.astype(float)
    md = df['MD'].values.astype(float)
    nanmask = np.isnan(g)
    if nanmask.any():
        g[nanmask] = np.interp(md[nanmask], md[~nanmask], g[~nanmask]) if (~nanmask).any() else 0.0
    gz_all = robust_z(g)
    gz = gz_all[ps:]
    L = len(gz)
    # typewell GR(TVT): sort, smooth, interp onto grid; z-normalise
    t = tw.dropna(subset=['TVT', 'GR']).sort_values('TVT')
    tvt_tw = t['TVT'].values.astype(float); gr_tw = t['GR'].values.astype(float)
    if smooth_tw > 1:
        k = np.ones(smooth_tw) / smooth_tw
        gr_tw = np.convolve(gr_tw, k, mode='same')
    lo = max(anchor - grid_half, tvt_tw.min()); hi = min(anchor + grid_half, tvt_tw.max())
    if hi - lo < dz:  # typewell doesn't cover; hold anchor
        return np.full(L, anchor), ps
    grid = np.arange(lo, hi + dz, dz)
    gr_grid = robust_z(np.interp(grid, tvt_tw, gr_tw))
    G = len(grid)
    band = int(round(band_ft / dz))
    offs = np.arange(-band, band + 1)
    trans_pen = lam * (offs * dz) ** 2  # cost of moving 'off' states
    INF = 1e18
    cost = np.full(G, INF); bptr = np.zeros((L, G), np.int32)
    # init: anchor strongly at nearest grid to anchor TVT
    a_idx = int(round((anchor - lo) / dz))
    cost[:] = ((grid - anchor) / 3.0) ** 2 + (gz[0] - gr_grid) ** 2
    for i in range(1, L):
        best = np.full(G, INF); arg = np.zeros(G, np.int32)
        for o, tp in zip(offs, trans_pen):
            # state s comes from prev state s-o
            src = np.full(G, INF)
            if o >= 0:
                src[o:] = cost[:G - o] + tp if o > 0 else cost + tp
            else:
                src[:G + o] = cost[-o:] + tp
            upd = src < best
            best[upd] = src[upd]; arg[upd] = (np.arange(G) - o)[upd]
        cost = best + (gz[i] - gr_grid) ** 2
        bptr[i] = arg
    # backtrack from min final
    s = int(np.argmin(cost)); path = np.zeros(L, np.int32); path[-1] = s
    for i in range(L - 1, 0, -1):
        s = bptr[i, s]; path[i - 1] = s
    return grid[path], ps

if __name__ == '__main__':
    import sys, time
    t0 = time.time()
    files = sorted(glob.glob(f'{D}/train/*__horizontal_well.csv'))
    rng = np.random.default_rng(0); sample = rng.choice(len(files), size=60, replace=False)
    rmse_dtw, rmse_hold = [], []
    for j in sample:
        df, tw = load_well(files[j])
        if 'TVT' not in df or not df['TVT_input'].isna().any():
            continue
        pred, ps = predict_well(df, tw)
        y = df['TVT'].values[ps:]
        if len(y) == 0:
            continue
        rmse_dtw.append(np.sqrt(np.mean((y - pred) ** 2)))
        rmse_hold.append(np.sqrt(np.mean((y - df['TVT'].iloc[ps - 1]) ** 2)))
    print(f"wells={len(rmse_dtw)}  time={time.time()-t0:.0f}s")
    print(f"Viterbi-GR  mean RMSE {np.mean(rmse_dtw):7.3f}  median {np.median(rmse_dtw):7.3f}")
    print(f"hold-last   mean RMSE {np.mean(rmse_hold):7.3f}  median {np.median(rmse_hold):7.3f}")
    print(f"win rate vs hold-last: {np.mean(np.array(rmse_dtw)<np.array(rmse_hold)):.2f}")
