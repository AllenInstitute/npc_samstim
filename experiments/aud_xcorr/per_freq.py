import numpy as np, sys, methods as M
f = sys.argv[1]; meths = sys.argv[2].split(',')
meta, trials = M.load(M.CACHE / f)
tones = [t for t in trials if t.name == 'tone']
def freq(t):
    F = np.fft.rfft(t.wf); return np.fft.rfftfreq(len(t.wf), 1/t.wsr)[np.argmax(np.abs(F))]
fr = np.round([freq(t) for t in tones], -2)
for m in meths:
    err = np.array([getattr(M, m)(t)[0] - t.gt_on for t in tones]) * 1e6
    print(m, ' | '.join(f'{u/1e3:.0f}k:{np.mean(np.abs(err[fr==u])>1000):.2f}/{np.median(np.abs(err[fr==u])):.0f}' for u in np.unique(fr)))
