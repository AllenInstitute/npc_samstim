import numpy as np, pandas as pd, warnings, clockmap as C; warnings.filterwarnings('ignore')
d = pd.read_csv('results/trials.csv')
v = d[d.has_sync & d.gt_on.notna() & (d.r_env > 0.5)].copy()
v = v[((v.new - v.gt_on).abs() < 1e-3)]
methods = [('npc_ephys (2-pt linear)', None, 0), ('LS linear fit', 'linear_fit', 0), ('interp', 'interp', 0)] + [(f'movmean w={w}', 'movmean', w) for w in (3, 5, 9, 15)] + [(f'locallin w={w}', 'locallin', w) for w in (5, 9, 15)]
out = {m[0]: [] for m in methods}
spans = []
for s, g in v.groupby('session'):
    k, t, lsr, lst = C.load_pairs(s)
    spans.append((s, len(k), np.diff(k).mean() / lsr if len(k) > 1 else np.nan))
    ks = (g.new.values - lst) * lsr  # NI-DAQ sample index of estimated onset
    for name, meth, w in methods:
        est = g.new.values if meth is None else C.make_map(k, t, meth, w)(ks)
        out[name].append(pd.DataFrame(dict(session=s, stim=g.stim.values, name=g.name.values, e=(est - g.gt_on.values) * 1e6)))
print('barcode pairs per session (n, mean interval s):', [(a[8:23], b, round(c, 1)) for a, b, c in spans][:5], '...')
rows = []
for name, parts in out.items():
    e = pd.concat(parts)
    for nm, g in e.groupby('name'):
        a = g.e.abs(); sm = g.groupby('session').e.median()
        rows.append(dict(clock=name, sound=nm, n=len(g), median=g.e.median(), p50abs=a.median(), p95abs=a.quantile(.95), p99abs=a.quantile(.99), max=a.max(), session_median_range=f'{sm.min():.0f}..{sm.max():.0f}'))
print(pd.DataFrame(rows).round(1).to_string())
