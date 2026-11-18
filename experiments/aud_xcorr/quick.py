import numpy as np, sys, time, collections
import methods as M
f = sys.argv[1]; meths = sys.argv[2].split(',')
meta, trials = M.load(M.CACHE / f)
print(meta['session'], meta['stim'], meta['sound_mode'], len(trials), collections.Counter(t.name for t in trials))
lat = np.array([t.gt_on - t.trigger for t in trials]); print('gt latency ms pct', np.round(np.nanpercentile(lat*1e3,[0,5,50,95,100]),3))
gtd = np.array([t.gt_off - t.gt_on - t.dur for t in trials]); print('gt dur - nominal (ms) pct', np.round(np.nanpercentile(gtd*1e3,[0,50,100]),3))
for m in meths:
    t0=time.time(); est = np.array([getattr(M, m)(t) for t in trials]); dt=time.time()-t0
    err = (est[:,0] - np.array([t.gt_on for t in trials]))*1e6
    for nm in sorted(set(t.name for t in trials)):
        e = err[[t.name == nm for t in trials]]
        print(f'{m:14s} {nm:10s} n={len(e):4d} med={np.median(e):8.1f}us MAD={np.median(np.abs(e-np.median(e))):7.1f} p95|e|={np.percentile(np.abs(e),95):8.1f} max|e|={np.abs(e).max():9.1f}  >1ms={np.mean(np.abs(e)>1000):.3f}  {dt:.1f}s')
