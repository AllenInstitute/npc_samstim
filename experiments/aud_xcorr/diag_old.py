import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings('ignore')
d = pd.read_csv('results/trials.csv')
o = d[~d.has_sync].copy()
o['lat'] = (o.new - o.trigger) * 1e3; o['lat_leg'] = (o.legacy - o.trigger) * 1e3
print('old trials', len(o), 'files', o.groupby(['session','stim']).ngroups)
print('r_env pct [0,0.1,1,5,50]:', np.percentile(o.r_env, [0, .1, 1, 5, 50]).round(3))
print('n r_env<0.5:', (o.r_env < 0.5).sum(), o[o.r_env < 0.5].groupby(['session','stim']).size().to_dict())
g = o[o.r_env >= 0.5]
print('new latency (ms) pct [0, .1, 1, 50, 99, 99.9, 100]:', np.percentile(g.lat, [0, .1, 1, 50, 99, 99.9, 100]).round(2))
print('per mode latency (ms):'); print(g.groupby('mode').lat.describe(percentiles=[.01, .5, .99]).round(1).to_string())
print('legacy - new (us) by mode/name: |diff| p50, p95, frac>1ms')
o['dleg'] = (o.legacy - o.new) * 1e6
print(o[o.r_env >= .5].groupby(['mode','name']).dleg.apply(lambda x: pd.Series(dict(p50=x.abs().median(), p95=x.abs().quantile(.95), gt1ms=(x.abs() > 1000).mean()))).unstack().round(3).to_string())
S = json.load(open('sessions.json'))
from collections import Counter
print(Counter((s['session'].split('_')[2][:7] >= '2023-08', d.split('/')[-1]) for s in S for d in s['rec_dirs']))
