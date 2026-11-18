import numpy as np, pandas as pd
d = pd.read_csv('results/trials.csv'); cal = pd.read_csv('results/calibration.csv')
v = d[d.has_sync & d.gt_on.notna()].copy()
print('validation: files', v.groupby(['session','stim']).ngroups, 'sessions', v.session.nunique(), 'trials', len(v))
for c in ('new', 'new_env', 'legacy', 'legacy_env'):
    v['e_' + c] = (v[c] - v.gt_on) * 1e6
def summ(g, c):
    e = g['e_' + c]; a = e.abs()
    return pd.Series(dict(n=len(e), med=e.median(), mad=(e - e.median()).abs().median(), p95=a.quantile(.95), p99=a.quantile(.99), max=a.max(), gt100us=(a > 100).mean(), gt1ms=(a > 1000).mean()))
for c in ('legacy', 'legacy_env', 'new_env', 'new'):
    print('\n==', c); print(v.groupby(['stim', 'name']).apply(lambda g: summ(g, c), include_groups=False).round(3).to_string())
print(cal[cal.has_sync][['session','stim','recon','ppm','cal_r']].to_string())
