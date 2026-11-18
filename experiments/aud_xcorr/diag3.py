import numpy as np, pandas as pd, warnings, matplotlib; warnings.filterwarnings('ignore'); matplotlib.use('Agg'); import matplotlib.pyplot as plt
d = pd.read_csv('results/trials.csv')
v = d[d.has_sync & d.gt_on.notna() & (d.r_env > 0.5)].copy()
v['e'] = (v.new - v.gt_on) * 1e6
v = v[v.e.abs() < 1000]
sess = sorted(v.session.unique())
fig, axes = plt.subplots(6, 6, figsize=(20, 16), sharey=True); axes = axes.ravel()
rows = []
for ax, s in zip(axes, sess):
    g = v[v.session == s]
    for (stim, name), gg in g.groupby(['stim', 'name']):
        ax.plot(gg.trigger / 60, gg.e, '.', ms=1.5, label=f'{stim[:2]} {name}')
    x = g[g.name == 'AM_noise']
    p = np.polyfit(x.trigger, x.e, 1) if len(x) > 10 else [np.nan, np.nan]
    rows.append(dict(session=s, slope_us_per_hr=p[0] * 3600, resid_mad=np.median(np.abs(x.e - np.polyval(p, x.trigger)))))
    ax.set_title(s[8:30], fontsize=8); ax.set_ylim(-200, 250); ax.axhline(0, color='k', lw=.5)
axes[0].legend(fontsize=6, markerscale=5)
for ax in axes[30:]: ax.set_xlabel('session time (min)')
fig.tight_layout(); fig.savefig('results/diag_error_vs_time.png', dpi=70)
print(pd.DataFrame(rows).round(1).to_string())
