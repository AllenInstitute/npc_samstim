import numpy as np, pandas as pd, glob, sys, time, concurrent.futures as cf, warnings, clockmap as C, methods as M
from npc_samstim import sound_alignment as SA
warnings.filterwarnings('ignore')
cal = pd.read_csv('results/calibration.csv').set_index(['session', 'stim'])
files = sorted(glob.glob(str(M.CACHE / '*RFMapping.npz')))
files = [f for f in files if M.load.__module__ and '2023-09' <= f.split('ecephys_')[1][7:14] <= '2023-12'][::4][:8]
def run(f, nfrac):
    meta, T = M.load(__import__('pathlib').Path(f))
    c = cal.loc[(meta['session'], 'RFMapping')]
    model = SA.PlaybackModel(c.rate, c.recon, meta['wave_sr'])
    k, t, lsr, lst = C.load_pairs(meta['session']); fmap = C.make_map(k, t, 'movmean', 9)
    out = []
    t0 = time.time()
    for tr in T:
        if tr.name != 'tone' or not np.isfinite(tr.gt_on):
            continue
        a = SA.align_sound_waveform(tr.seg, tr.seg_t0, tr.sr, tr.wf, model, tr.trigger - 0.02, tr.trigger + 0.25, n_envelope_frac_fine=nfrac)
        e = (fmap((a.onset_time - lst) * lsr) - tr.gt_on) * 1e6 - 19.4
        out.append((round(np.argmax(np.abs(np.fft.rfft(tr.wf))) * tr.wsr / len(tr.wf) / 1000), e))
    return out, time.time() - t0
if __name__ == '__main__':
  for nfrac in (128,):
    with cf.ProcessPoolExecutor(8) as ex:
        res = list(ex.map(run, files, [nfrac] * len(files)))
    df = pd.DataFrame([r for o, _ in res for r in o], columns=['f', 'e'])
    s = df.groupby('f').e.apply(lambda x: x.abs().quantile(.95)).round(0)
    print(f'nfrac={nfrac:4d} sec/trial={np.mean([t for _, t in res]) / (len(df) / len(files)):.3f}  p95 all={df.e.abs().quantile(.95):.0f} p50 all={df.e.abs().median():.0f}  p95 by kHz:', dict(s), flush=True)
