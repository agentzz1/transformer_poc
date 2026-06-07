"""rogii wellbore geology: PS detection + baseline RMSE analysis.
Task: predict TVT beyond Prediction Start (PS) from horizontal GR + typewell GR(TVT). Metric: RMSE(dTVT).
PS = first row where TVT_input is NaN. Train wells carry true TVT + 6 formation tops.
Findings (150 train wells): hold-last-TVT RMSE~12.5 (median 10.4); geometric -dZ ~93 (TVT does NOT track vertical);
ridge public LB 8.723. Lever: GR-to-typewell sequence correlation (DTW) with continuity at PS.
"""
import glob, numpy as np, pandas as pd
D='data/rogii_full'
def ps_index(df): return int(df['TVT_input'].isna().values.argmax()) if df['TVT_input'].isna().any() else len(df)
# (see git history / chat for the baseline computation)
