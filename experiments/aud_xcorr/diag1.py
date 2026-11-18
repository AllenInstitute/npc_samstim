import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
d = pd.read_csv('results/trials.csv')
v = d[d.has_sync & d.gt_on.notna()].copy()
v['e'] = (v.new - v.gt_on) * 1e6; v['el'] = (v.legacy - v.gt_on) * 1e6
v['gt_lat'] = (v.gt_on - v.trigger) * 1e3; v['new_lat'] = (v.new - v.trigger) * 1e3
v['gt_dur_err'] = (v.gt_off - v.gt_on - (v.n_wf - 1) / 47990.3) * 1e3
g = v[v.e.abs() > 1000]
print('gross errors per file:'); print(g.groupby(['session', 'stim']).size().to_string())
print(g[['session','stim','trial','name','gt_lat','new_lat','gt_dur_err','r_env','r_car']].head(30).to_string())
print('\nGT latency distribution (ms) for non-gross:', np.percentile(v[v.e.abs() < 1000].gt_lat, [0, 1, 50, 99, 100]).round(2))
print('gt dur err (ms) distribution non-gross', np.percentile(v[v.e.abs() < 1000].gt_dur_err.dropna(), [0, 1, 50, 99, 100]).round(3))
print('gt dur err (ms) gross', np.percentile(g.gt_dur_err.dropna(), [0, 1, 50, 99, 100]).round(3))
