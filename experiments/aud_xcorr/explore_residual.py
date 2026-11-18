import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import methods as M
meta, trials = M.load(M.CACHE / 'ecephys_668755_2023-08-31_12-33-31__RFMapping.npz')
def freq(t):
    F = np.fft.rfft(t.wf); return np.fft.rfftfreq(len(t.wf), 1/t.wsr)[np.argmax(np.abs(F))]
fig, axes = plt.subplots(3, 2, figsize=(14, 9))
for row, t in zip(axes, [next(t for t in trials if t.name=='tone' and abs(freq(t)-6000)<50), next(t for t in trials if t.name=='tone' and abs(freq(t)-12000)<50), next(t for t in trials if t.name=='AM_noise')]):
    st = M._seg_times(t); j = np.searchsorted(st, t.gt_on - 0.002); n = int((t.dur+0.004)*t.sr)
    f = (st[j] - t.gt_on) * t.sr
    w = M.dac_template(t.wf, M.DAQ_RATE(t.wsr), t.sr, -f, 'zoh', n=n)
    v = t.seg[j:j+n].astype(float)
    A = np.c_[w, np.ones(n)]; coef = np.linalg.lstsq(A, v, rcond=None)[0]; res = v - A @ coef
    tt = (st[j:j+n] - t.gt_on) * 1e3
    row[0].plot(tt, v, lw=.3, label='rec'); row[0].plot(tt, res, lw=.3, label='residual'); row[0].set_title(f'{t.name} {freq(t):.0f}Hz gain={coef[0]:.1f} dc={coef[1]:.1f} resid rms/sig rms={res.std()/v.std():.4f}', fontsize=8); row[0].legend(fontsize=6)
    row[1].scatter(A @ coef, res, s=.2); row[1].set_xlabel('model'); row[1].set_ylabel('residual')
fig.tight_layout(); fig.savefig('explore_residual.png', dpi=90)
