import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import methods as M
meta, trials = M.load(M.CACHE / 'ecephys_668755_2023-08-31_12-33-31__RFMapping.npz')
def freq(t):
    F = np.fft.rfft(t.wf); return np.fft.rfftfreq(len(t.wf), 1/t.wsr)[np.argmax(np.abs(F))]
fig, axes = plt.subplots(3, 1, figsize=(12, 9))
for ax, f0 in zip(axes, (6000, 9000, 16000)):
    t = next(t for t in trials if t.name == 'tone' and abs(freq(t) - f0) < 50)
    st = M._seg_times(t); j = np.searchsorted(st, t.gt_on + 0.01); n = int(0.22 * t.sr)
    v = t.seg[j:j+n].astype(float); v -= v.mean()
    F = np.abs(np.fft.rfft(v * np.hanning(n))); fr = np.fft.rfftfreq(n, 1/t.sr)
    tau = st[j] - t.gt_on
    for model, c in (('zoh', 'C1'), ('linear', 'C2')):
        w = M.dac_template(t.wf, M.DAQ_RATE(t.wsr), t.sr, -tau * t.sr, model, n=n + 10)[:n]
        w = w - w.mean(); w *= np.std(v)/np.std(w)
        Fw = np.abs(np.fft.rfft(w * np.hanning(n)))
        ax.semilogy(fr, Fw, c, lw=.6, alpha=.8, label=f'model {model}')
    ax.semilogy(fr, F, 'k', lw=.6, label='recorded ch1'); ax.set_title(f'{f0} Hz tone'); ax.set_ylim(F.max()*1e-6, F.max()*2)
    # print main peaks
    pk = np.argsort(F)[::-1]; seen=[]
    for p in pk:
        if all(abs(fr[p]-s) > 200 for s in seen): seen.append(fr[p])
        if len(seen) == 4: break
    print(f0, 'recorded top peaks (Hz, rel dB):', [(round(s), round(20*np.log10(F[np.argmin(abs(fr-s))]/F.max()),1)) for s in seen])
axes[0].legend(); fig.tight_layout(); fig.savefig('explore_spectrum.png', dpi=90)
