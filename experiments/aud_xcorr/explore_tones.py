import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import methods as M, collections
meta, trials = M.load(M.CACHE / 'ecephys_668755_2023-08-31_12-33-31__RFMapping.npz')
tones = [t for t in trials if t.name == 'tone']
def freq(t):
    f = np.fft.rfft(t.wf); return np.fft.rfftfreq(len(t.wf), 1/t.wsr)[np.argmax(np.abs(f))]
fr = np.array([freq(t) for t in tones])
err = np.array([M.baseline(t)[0] - t.gt_on for t in tones])*1e3
for f in np.unique(np.round(fr, -1)):
    e = err[np.round(fr,-1) == f]; print(f'{f:8.0f} Hz n={len(e)} med err {np.median(e):8.3f} ms  frac>1ms {np.mean(np.abs(e)>1):.2f}')
fig, axes = plt.subplots(4, 1, figsize=(12, 11))
for ax, f in zip(axes, np.unique(np.round(fr,-1))[[0, 3, 6, -1]]):
    t = tones[int(np.flatnonzero(np.round(fr,-1) == f)[0])]
    st = M._seg_times(t) - t.gt_on
    ax.plot(st*1e3, t.seg, lw=.4, label='nidaq ch1')
    ax.plot((np.arange(len(t.wf))/t.wsr)*1e3, t.wf/np.abs(t.wf).max()*np.abs(t.seg).max()*0.8, lw=.4, alpha=.6, label='ideal (aligned at GT)')
    ax.set_xlim(-10, 270); ax.set_title(f'{f:.0f} Hz  baseline err {(M.baseline(t)[0]-t.gt_on)*1e3:.2f} ms')
axes[0].legend(); fig.tight_layout(); fig.savefig('explore_tones.png', dpi=100)
