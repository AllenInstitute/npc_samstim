import numpy as np, pandas as pd, warnings, clockmap as C; warnings.filterwarnings('ignore')
d = pd.read_csv('results/trials.csv')
v = d[d.has_sync & d.gt_on.notna() & (d.r_env > 0.5) & (d.stim == 'RFMapping')].copy()
parts = []
for s, g in v.groupby('session'):
    k, t, lsr, lst = C.load_pairs(s); f = C.make_map(k, t, 'movmean', 9)
    g = g.copy()
    for c in ('new', 'new_env', 'legacy_env'):
        g['e_' + c] = (f((g[c] - lst) * lsr) - g.gt_on) * 1e6 - 19.4
    parts.append(g)
v = pd.concat(parts); v['f'] = (v.freq / 1000).round()
t = v[v.name == 'tone']
print(t.groupby('f').apply(lambda g: pd.Series({**{f'{c}_p50': g['e_' + c].abs().median() for c in ('new', 'new_env')}, **{f'{c}_p95': g['e_' + c].abs().quantile(.95) for c in ('new', 'new_env', 'legacy_env')}, 'n': len(g), 'r_car': g.r_car.median()})).round(1).to_string())
# error in units of carrier period
t = t.assign(cyc=t.e_new * 1e-6 * t.freq)
print('\nfraction of tone errors within +-0.5 cycle:', (t.cyc.abs() < .5).mean().round(3), ' within 1.5 cycles:', (t.cyc.abs() < 1.5).mean().round(3))
print('per-session tone p95 (us):', t.groupby('session').e_new.apply(lambda x: x.abs().quantile(.95)).round(0).values)
print('per-session RF calib r:'); cal = pd.read_csv('results/calibration.csv'); print(cal[(cal.stim=='RFMapping') & cal.has_sync][['session','recon','cal_r']].assign(p95=t.groupby('session').e_new.apply(lambda x: x.abs().quantile(.95)).reindex(cal[(cal.stim=='RFMapping') & cal.has_sync].session).values).round(3).to_string())
print('\nsigned median error (us) per freq, and in cycles')
print(t.groupby('f').apply(lambda g: pd.Series(dict(new=g.e_new.median(), env=g.e_new_env.median(), new_cyc=(g.e_new*1e-6*g.freq).median(), new_iqr=g.e_new.quantile(.75)-g.e_new.quantile(.25), env_iqr=g.e_new_env.quantile(.75)-g.e_new_env.quantile(.25)))).round(2).to_string())
