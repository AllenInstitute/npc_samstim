import numpy as np, glob, os, methods as M, sys
files = [f for f in sorted(glob.glob(str(M.CACHE / '*.npz')))]
picks = ['620263_2022-07-26', '644547_2022-12-05_13-07-55__RF', '649944_2023-02-28', '662983_2023-05-16_12-33-22__RF', '660023_2023-08-09_08-09-09__RF', '670248_2023-08-03_12-04-15__Dyna', '668755_2023-08-31_12-33-31__RF', '676909_2023-12-13_13-43-40__Dyna']
for p in picks:
    f = next(f for f in files if p in f)
    meta, T = M.load(f)
    sel = T[::max(1, len(T)//12)][:12]
    guesses = [M.v4(t, fs_eff=48000 if meta['sound_mode'] != 'daq' else M.DAQ_RATE(48000), n_frac=4, stage_b=False, search=(-0.1, 0.3))[0] for t in sel]
    row = []
    for model in ('zoh', 'linear', 'bandlimited'):
        for ppm in (-200.0, -20, 0, 20):  # (originally run with -208.3: an arithmetic slip; -200 ppm is 80 MHz/1667)
            fs = 48000 * (1 + ppm * 1e-6)
            rs = [M.local_carrier_r(t, g, fs, model, hw_sec=0.4e-3)[0] for t, g in zip(sel, guesses)]
            row.append(f'{model[:3]}{ppm:+.0f}:{np.median(rs):.3f}')
    print(os.path.basename(f)[8:42], meta['sound_mode'], ' '.join(row), flush=True)
