"""Old sessions: latency range (robust, from new estimator w/ nominal & daq rate), r quality,
and effective DAC rate estimate by scanning a stretch factor."""
import numpy as np, json, glob, os, methods as M, collections
files = sorted(glob.glob(str(M.CACHE / '*.npz')))
seen = set()
for f in files:
    meta, T = M.load(f)
    key = (meta['sound_mode'], meta['stim'].split('_')[0])
    if meta['has_sync_line'] or sum(1 for k in seen if k == key) >= 1 and np.random.rand() > .25:
        continue
    seen.add(key)
    sel = T[::max(1, len(T)//25)]
    out = []
    for fs in (48000.0, M.DAQ_RATE(48000), 48000*(1-1e-4), 48000*(1+1e-4), 48000 * (1 - 4e-4)):
        res = [M.v4(t, fs_eff=fs, n_frac=4, refine='parabolic', search=(-0.1, 0.3)) for t in sel]
        out.append((fs, np.median([r for _, r in res]), np.array([on - t.trigger for (on, _), t in zip(res, sel)])))
    best = max(out, key=lambda x: x[1])
    lat = best[2] * 1e3
    print(os.path.basename(f)[8:40], meta['sound_mode'], collections.Counter(t.name for t in T).most_common(2),
          ' r by fs:', ' '.join(f'{fs:.1f}:{r:.4f}' for fs, r, _ in out),
          f' lat ms [min,med,max]=[{lat.min():.1f},{np.median(lat):.1f},{lat.max():.1f}]', flush=True)
