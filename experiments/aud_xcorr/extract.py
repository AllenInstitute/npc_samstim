"""Cache everything needed to evaluate audio-onset alignment offline.

For each session x stim file (DynamicRouting1 / RFMapping) with sound trials:
  - trigger time on sync (vsync of trialStimStartFrame)
  - ground-truth onset/offset from sync line 1 ("sound running" marker, >= 2023-08-31)
  - NI-DAQ ch1 (speaker copy) segment around each trial, int16
  - the ideal waveform (float32) as used by npc_samstim
  - NI-DAQ timing on sync (barcode-aligned sampling rate & start time)

Usage: python extract.py [session_substring ...]
"""
from __future__ import annotations

import _creds  # noqa: F401

import concurrent.futures as cf
import datetime
import json
import pathlib
import sys
import time
import traceback

import numpy as np
import npc_ephys
import npc_stim
import npc_sync
import upath

import npc_samstim.waveforms as W

CACHE = pathlib.Path(__file__).parent / "cache"
CACHE.mkdir(exist_ok=True)
PRE = 0.2  # s before trigger
POST = 0.35  # s after trigger + duration
NIDAQ_CH = 1
FIRST_SOUND_ON_SYNC = datetime.date(2023, 8, 31)


def extract_session(s: dict) -> list[str]:
    session = s["session"]
    root = f"s3://aind-ephys-data/{session}"
    sync = npc_sync.get_sync_data(upath.UPath(f"{root}/behavior/{s['sync']}"))
    # if NI-DAQ was recorded in several experiment/recording dirs, use the one
    # spanning the most sync time (each tried alone)
    candidates = []
    for d in s["rec_dirs"]:
        rd = (f"s3://{d}",)
        try:
            dev = npc_ephys.get_pxi_nidaq_info(rd)
            timing = next(npc_ephys.get_ephys_timing_on_sync(sync=sync, devices=(dev,)))
            data = npc_ephys.get_pxi_nidaq_data(*rd, device_name=dev.device.name)
        except Exception as exc:  # noqa: BLE001
            print(session, d, "unusable:", repr(exc)[:200])
            continue
        candidates.append((data.shape[0] / timing.sampling_rate, dev, timing, data))
    if not candidates:
        raise FileNotFoundError(f"no usable NI-DAQ recording for {session}")
    _, dev, timing, data = max(candidates, key=lambda c: c[0])
    has_sync_line = sync.start_time.date() >= FIRST_SOUND_ON_SYNC
    if has_sync_line:
        on = sync.get_rising_edges(1, units="seconds")
        off = sync.get_falling_edges(1, units="seconds")
    else:
        on = off = np.array([])
    done = []
    for stim_name in s["stims"]:
        out = CACHE / f"{session}__{stim_name.split('_')[0]}.npz"
        if out.exists():
            done.append(out.name)
            continue
        stim_path = upath.UPath(f"{root}/behavior/{stim_name}")
        stim = npc_stim.get_h5_stim_data(stim_path)
        try:
            wfs = W.get_waveforms_from_stim_file(stim, "sound")
        except Exception:
            print(session, stim_name, "no waveforms", traceback.format_exc()[-300:])
            continue
        if not any(w is not None for w in wfs):
            continue
        vs = npc_stim.get_stim_frame_times(stim_path, sync=sync, frame_time_type="vsync")[stim_path]
        vs = npc_stim.assert_stim_times(vs)
        trig_frames = npc_stim.get_stim_trigger_frames(stim)
        idxs = [i for i, w in enumerate(wfs) if w is not None]
        trig = np.array([vs[trig_frames[i]] for i in idxs], dtype=float)
        durs = np.array([wfs[i].duration for i in idxs])
        sr_w = float(wfs[idxs[0]].sampling_rate)
        names = np.array([wfs[i].name for i in idxs])
        wf_samples = [np.asarray(wfs[i].samples, dtype=np.float32) for i in idxs]

        gt_on = np.full(len(idxs), np.nan)
        gt_off = np.full(len(idxs), np.nan)
        if has_sync_line and len(on):
            k = np.searchsorted(on, trig, side="right")
            ok = k < len(on)
            gt_on[ok] = on[k[ok]]
            k_off = np.searchsorted(off, gt_on[ok], side="right")
            ok2 = k_off < len(off)
            tmp = np.full(ok.sum(), np.nan)
            tmp[ok2] = off[k_off[ok2]]
            gt_off[ok] = tmp

        seg_t0 = trig - PRE  # sync time of nominal first sample requested
        s0 = np.round((seg_t0 - timing.start_time) * timing.sampling_rate).astype(int)
        n = np.round((PRE + durs + POST) * timing.sampling_rate).astype(int)
        seg_t0 = s0 / timing.sampling_rate + timing.start_time  # exact time of first sample

        def read(j):
            a, b = s0[j], s0[j] + n[j]
            if a < 0 or b > data.shape[0]:
                return np.zeros(n[j], dtype=np.int16)
            return np.asarray(data[a:b, NIDAQ_CH], dtype=np.int16)

        with cf.ThreadPoolExecutor(16) as ex:
            segs = list(ex.map(read, range(len(idxs))))

        meta = dict(
            session=session, stim=stim_name,
            rig=stim["rigName"][()].decode() if "rigName" in stim else "",
            sound_mode=stim["soundMode"][()].decode() if "soundMode" in stim else "",
            date=str(sync.start_time.date()),
            nidaq_sr=float(timing.sampling_rate), nidaq_start=float(timing.start_time),
            nidaq_nominal_sr=float(dev.sampling_rate), wave_sr=sr_w,
            has_sync_line=bool(has_sync_line), n_sync_edges=int(len(on)),
        )
        np.savez(
            out,
            meta=json.dumps(meta),
            trial_idx=np.array(idxs), trigger=trig, dur=durs, names=names,
            gt_on=gt_on, gt_off=gt_off, seg_t0=seg_t0,
            seg_len=np.array([len(x) for x in segs]), segs=np.concatenate(segs),
            wf_len=np.array([len(x) for x in wf_samples]), wfs=np.concatenate(wf_samples),
        )
        done.append(out.name)
    return done


def main(filters: list[str]) -> None:
    S = json.load(open(pathlib.Path(__file__).parent / "sessions.json"))
    if filters:
        S = [s for s in S if any(f in s["session"] for f in filters)]
    log = open(CACHE / "extract_log.txt", "a")
    for s in S:
        t = time.time()
        try:
            res = extract_session(s)
            msg = f"OK {s['session']} {res} {time.time()-t:.0f}s"
        except Exception as e:
            msg = f"FAIL {s['session']} {e!r} {traceback.format_exc()[-500:]}"
        print(msg, flush=True)
        log.write(msg + "\n"); log.flush()


if __name__ == "__main__":
    main(sys.argv[1:])
