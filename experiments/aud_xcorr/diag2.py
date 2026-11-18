import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
d = pd.read_csv('results/trials.csv')
v = d[d.has_sync & d.gt_on.notna()].copy()
v['e'] = (v.new - v.gt_on) * 1e6
bad = v.e.abs() > 1000
print('r_env good pct', np.percentile(v[~bad].r_env, [0, .1, 1, 5, 50]).round(3), ' bad pct', np.percentile(v[bad].r_env, [0, 50, 99, 100]).round(3))
print('r_car good pct', np.percentile(v[~bad].r_car, [0, .1, 1, 5, 50]).round(3), ' bad pct', np.percentile(v[bad].r_car, [0, 50, 99, 100]).round(3))
o = v[bad & ~v.session.str.contains('686176_2023-12-06')]
print(o[['session','stim','trial','name','freq','e','r_env','r_car']].to_string())
print('\nall trials (incl old) r_env < 0.5 per file:'); print(d[d.r_env < 0.5].groupby(['session','stim']).size().to_string())
# per-session bias
v2 = v[~bad]
ps = v2.groupby(['session', 'stim', 'name']).e.median().unstack(['stim','name']).round(1)
print(ps.to_string())
