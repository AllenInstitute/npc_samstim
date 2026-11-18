"""Which analog model best explains NI-DAQ ch1? Evaluate NCC at the GT alignment."""
import numpy as np, scipy.signal as ss, methods as M, sys
meta, trials = M.load(M.CACHE / sys.argv[1])
UP = 20
def analog_model(wf, fs_eff, fc, order, kind):
    x = np.repeat(wf.astype(np.float64), UP)  # ZOH at fs_eff*UP
    x = np.r_[np.zeros(UP*10), x, np.zeros(UP*200)]
    fs_hi = fs_eff * UP
    if fc:
        if kind == 'butter':
            sos = ss.butter(order, fc, fs=fs_hi, output='sos')
        else:
            sos = ss.bessel(order, fc, fs=fs_hi, output='sos', norm='delay')
        x = ss.sosfilt(sos, x)
    return x, fs_hi, UP*10
def r_at(t, tau, model):
    x, fs_hi, off = model
    st = M._seg_times(t); j = np.searchsorted(st, t.gt_on + tau - 0.002); n = int((t.dur + 0.004) * t.sr)
    ts = st[j:j+n] - (t.gt_on + tau)
    w = np.interp(ts * fs_hi + off, np.arange(len(x)), x, left=0, right=0)
    v = t.seg[j:j+n].astype(float); v -= v.mean(); w = w - w.mean()
    return v @ w / np.linalg.norm(v) / np.linalg.norm(w)
sel = trials[::4]
fs_eff = M.DAQ_RATE(sel[0].wsr)
for kind, order, fc in [('none',0,0), ('butter',1,20e3), ('butter',1,40e3), ('butter',1,80e3), ('butter',2,20e3), ('butter',2,30e3), ('butter',2,40e3), ('butter',2,60e3),('bessel',2,30e3),('bessel',4,30e3)]:
    res = {}
    for t in sel:
        m = analog_model(t.wf, fs_eff, fc, order, kind)
        # best tau near GT (allow filter group delay): scan -60..+60 us
        taus = np.arange(-60e-6, 61e-6, 2e-6)
        rs = [r_at(t, tau, m) for tau in taus]
        k = int(np.argmax(rs))
        res.setdefault(t.name, []).append((rs[k], taus[k]))
    print(f'{kind:6s} o{order} fc={fc/1e3:4.0f}k ', '  '.join(f"{nm}: 1-r med={np.median([1-a for a,_ in v]):.5f} tau*={np.median([b for _,b in v])*1e6:5.1f}us" for nm, v in sorted(res.items())), flush=True)
