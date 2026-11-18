import _creds  # noqa
import numpy as np, npc_sync, npc_stim, npc_ephys, h5py, upath, time, json, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import npc_samstim.waveforms as W

root = 's3://aind-ephys-data/ecephys_668755_2023-08-31_12-33-31'
stim_path = upath.UPath(root + '/behavior/DynamicRouting1_668755_20230831_131418.hdf5')
sync = npc_sync.get_sync_data(root + '/behavior/20230831T123331.h5')
recording_dirs = (root + '/ecephys_clipped/Record Node 102/experiment2/recording1',
                  root + '/ecephys_clipped/Record Node 103/experiment2/recording1')
t0 = time.time()
dev = npc_ephys.get_pxi_nidaq_info(recording_dirs)
timing = next(npc_ephys.get_ephys_timing_on_sync(sync=sync, recording_dirs=recording_dirs, devices=(dev,)))
data = npc_ephys.get_pxi_nidaq_data(*recording_dirs, device_name=dev.device.name)
print('nidaq', dev.device.name, timing.sampling_rate, timing.start_time, data.shape, type(data), time.time()-t0)
print('sync freq', sync.sample_freq, 'line labels', getattr(sync, 'line_labels', None))
stim = npc_stim.get_h5_stim_data(stim_path)
print('rig', stim['rigName'][()], 'soundMode?', [k for k in stim.keys() if 'sound' in k.lower()])
print('soundSampleRate', stim['soundSampleRate'][()])
on = sync.get_rising_edges(1, units='seconds'); off = sync.get_falling_edges(1, units='seconds')
print('n sound edges', len(on), len(off), 'dur stats', np.percentile(off[:len(on)]-on[:len(off)], [0, 50, 100]))
wfs = W.get_waveforms_from_stim_file(stim, 'sound')
vs = npc_stim.get_stim_frame_times(stim_path, sync=sync, frame_time_type='vsync')[stim_path]
trig = npc_stim.get_stim_trigger_frames(stim)
idxs = [i for i, w in enumerate(wfs) if w is not None][:6]
fig, axes = plt.subplots(len(idxs), 1, figsize=(12, 3*len(idxs)))
for ax, i in zip(axes, idxs):
    tt = vs[trig[i]]
    edge = on[np.searchsorted(on, tt)]
    s0 = int((tt - 0.05 - timing.start_time) * timing.sampling_rate)
    n = int(0.7 * timing.sampling_rate)
    seg = np.asarray(data[s0:s0+n, :])
    t = (np.arange(n) / timing.sampling_rate) + (tt - 0.05) - edge
    for ch in range(seg.shape[1]):
        y = seg[:, ch].astype(float); y = (y - y.mean()) / (np.ptp(y) + 1e-9)
        ax.plot(t*1e3, y + ch, lw=0.5, label=f'ch{ch}')
    w = wfs[i]
    ax.plot(w.timestamps*1e3, w.samples / np.abs(w.samples).max() * 0.5 + 8.5, 'k', lw=0.5)
    ax.axvline(0, color='r'); ax.set_title(f'trial {i} {w.name} lat(sync)={(edge-tt)*1e3:.2f} ms dur={w.duration}')
    ax.set_xlim(-60, 650)
axes[0].legend(ncol=9, fontsize=6)
fig.tight_layout(); fig.savefig('explore_668755.png', dpi=110)
