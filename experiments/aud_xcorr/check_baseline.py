"""Check: offline replica `methods.baseline` == library `npc_samstim.waveforms.xcorr`."""
import _creds  # noqa
import numpy as np, npc_ephys, npc_sync, upath, json
import npc_samstim.waveforms as W
import methods as M

sess = 'ecephys_668755_2023-08-31_12-33-31'
S = {s['session']: s for s in json.load(open('sessions.json'))}[sess]
for stimkind in ('DynamicRouting1', 'RFMapping'):
    meta, trials = M.load(M.CACHE / f'{sess}__{stimkind}.npz')
    sync = npc_sync.get_sync_data(upath.UPath(f"s3://aind-ephys-data/{sess}/behavior/{S['sync']}"))
    rec = tuple(f"s3://{d}" for d in S['rec_dirs'])
    dev = npc_ephys.get_pxi_nidaq_info(rec)
    timing = next(npc_ephys.get_ephys_timing_on_sync(sync=sync, devices=(dev,)))
    data = npc_ephys.get_pxi_nidaq_data(*rec, device_name=dev.device.name)
    sel = trials[::25][:12]
    pres = [W.StimPresentation(trial_idx=t.trial_idx, waveform=W.SimpleWaveform('x', 'sound', t.wsr, t.wf.astype(np.float64)), trigger_time_on_sync=t.trigger) for t in sel]
    for env in (False, True):
        lib = W.xcorr(data, timing, 1, pres, use_envelope=env)
        mine = [M.baseline(t, use_envelope=env)[0] for t in sel]
        d = np.array([r.onset_time_on_sync - m for r, m in zip(lib, mine)])
        print(stimkind, 'env' if env else 'raw', 'max |lib - replica| =', np.abs(d).max(), 's  timing match:', timing.sampling_rate == meta['nidaq_sr'], timing.start_time == meta['nidaq_start'])
