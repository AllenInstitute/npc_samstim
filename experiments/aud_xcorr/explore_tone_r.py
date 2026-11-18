import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import methods as M
meta, trials = M.load(M.CACHE / 'ecephys_668755_2023-08-31_12-33-31__RFMapping.npz')
tones = [t for t in trials if t.name == 'tone']
def freq(t):
    f = np.fft.rfft(t.wf); return np.fft.rfftfreq(len(t.wf), 1/t.wsr)[np.argmax(np.abs(f))]
fs_eff = 80e6/1667
sel = {}
for t in tones:
    sel.setdefault(round(freq(t), -2), t)
fig, axes = plt.subplots(len(sel)//2+1, 2, figsize=(13, 16)); axes = axes.ravel()
for ax, (f, t) in zip(axes, sorted(sel.items())):
    seg = t.seg.astype(float); st = M._seg_times(t)
    j0 = int(np.searchsorted(st, t.gt_on))
    taus = np.arange(-1e-3, 1e-3, 2e-6)
    rs = []
    for tau in taus:
        on = t.gt_on + tau
        j = int(np.floor((on - t.seg_t0) * t.sr)); frac = (on - st[j]) * t.sr
        w = M.dac_template(t.wf, fs_eff, t.sr, frac, 'zoh'); v = seg[j:j+len(w)]
        w = w - w.mean(); v = v - v.mean()
        rs.append(v @ w / np.linalg.norm(v) / np.linalg.norm(w))
    rs = np.array(rs)
    ax.plot(taus*1e6, rs, lw=.6); ax.set_title(f'{f:.0f} Hz: argmax {taus[np.argmax(rs)]*1e6:.0f} us, r={rs.max():.5f}', fontsize=8)
    ax.axvline(0, color='r', lw=.5)
fig.tight_layout(); fig.savefig('explore_tone_r.png', dpi=90)
