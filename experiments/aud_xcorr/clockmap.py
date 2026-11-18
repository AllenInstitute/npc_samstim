"""NI-DAQ sample -> sync time mappings from matched barcodes."""
import numpy as np, pathlib
CACHE = pathlib.Path(__file__).parent / 'cache'

def load_pairs(session):
    z = np.load(CACHE / f'{session}__barcodes.npz')
    sid, ss = z['sync_id'], z['sync_t']; pid, ps = z['nidaq_id'], z['nidaq_t_nominal'] * float(z['nominal_sr'])
    # unique ids only
    su = {i: t for i, t in zip(sid, ss) if (sid == i).sum() == 1}
    pu = {i: k for i, k in zip(pid, ps) if (pid == i).sum() == 1}
    common = sorted(set(su) & set(pu), key=lambda i: pu[i])
    k = np.array([pu[i] for i in common]); t = np.array([su[i] for i in common])
    # drop outlier pairs (inconsistent with local rate)
    if len(k) > 3:
        p = np.polyfit(k, t, 1); res = t - np.polyval(p, k)
        ok = np.abs(res - np.median(res)) < 5e-3
        k, t = k[ok], t[ok]
    return k, t, float(z['lin_sr']), float(z['lin_start'])

def make_map(k, t, method='linear', w=5):
    p = np.polyfit(k, t, 1)
    if method == 'linear_fit':
        return lambda x: np.polyval(p, x)
    res = t - np.polyval(p, k)
    if method == 'interp':
        r = res
    elif method == 'movmean':
        r = np.array([res[max(0, i - w // 2): i + w // 2 + 1].mean() for i in range(len(res))])
    elif method == 'locallin':
        # local linear regression over w neighbours (handles curvature better than mean at ends)
        r = np.empty(len(res))
        for i in range(len(res)):
            a, b = max(0, i - w // 2), min(len(res), i + w // 2 + 1)
            if b - a >= 3:
                q = np.polyfit(k[a:b] - k[i], res[a:b], 1); r[i] = q[1]
            else:
                r[i] = res[i]
    else:
        raise ValueError(method)
    return lambda x: np.polyval(p, x) + np.interp(x, k, r)
