"""End-to-end: public API on S3 (new default vs legacy) against sync-line ground truth."""
import _creds  # noqa
import json, sys, time, logging, numpy as np, pandas as pd
import npc_samstim.waveforms as W
logging.basicConfig(level=logging.WARNING)
S = {s['session']: s for s in json.load(open('sessions.json'))}
rows = []
for sess in sys.argv[1:]:
    s = next(v for k, v in S.items() if sess in k)
    root = f"s3://aind-ephys-data/{s['session']}"
    sync = f"{root}/behavior/{s['sync']}"
    rec = tuple(f"s3://{d}" for d in s['rec_dirs'])[:1]
    for stim in s['stims']:
        stim_path = f"{root}/behavior/{stim}"
        gt = W.get_stim_latencies_from_sync(stim_path, sync, 'sound')
        out = {}
        for label, kw in (('new', {}), ('legacy', dict(legacy=True))):
            t0 = time.time()
            r = W.get_stim_latencies_from_nidaq_recording(stim_path, sync, rec, 'sound', correlation_method_kwargs=kw)
            out[label] = (r, time.time() - t0)
        for i, g in enumerate(gt):
            if g is None:
                continue
            nr = out['new'][0][i]
            rows.append(dict(session=s['session'], stim=stim.split('_')[0], trial=i, name=g.name, gt=g.onset_time_on_sync,
                             new=np.nan if np.isnan(nr.latency) else nr.onset_time_on_sync,
                             legacy=out['legacy'][0][i].onset_time_on_sync))
        print(s['session'], stim.split('_')[0], 'sec new %.0f legacy %.0f' % (out['new'][1], out['legacy'][1]), flush=True)
d = pd.DataFrame(rows)
d.to_csv('results/e2e_trials_final.csv', index=False)
for c in ('new', 'legacy'):
    d['e_' + c] = (d[c] - d['gt']) * 1e6
print(d.groupby(['session', 'stim', 'name'])[['e_new', 'e_legacy']].agg([lambda x: np.nanmedian(x), lambda x: np.nanpercentile(np.abs(x), 95), lambda x: np.nanmax(np.abs(x))]).round(1).to_string())
