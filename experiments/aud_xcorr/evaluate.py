"""Run legacy xcorr (raw + envelope) and the new library aligner on every cached
stim file; write per-trial results to results/trials.csv and per-file playback
calibration to results/calibration.csv.

Usage: python evaluate.py [n_workers]
"""
from __future__ import annotations

import concurrent.futures as cf
import glob
import json
import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd

import methods as M
from npc_samstim import sound_alignment as SA

OUT = pathlib.Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
SEARCH = (-0.02, 0.25)  # onset search window relative to trigger (s)
N_CAL = 12


def dominant_freq(wf, sr):
    F = np.abs(np.fft.rfft(wf))
    return float(np.fft.rfftfreq(len(wf), 1 / sr)[np.argmax(F)])


def run_file(path: str) -> tuple[list[dict], dict]:
    meta, trials = M.load(pathlib.Path(path))
    trials = [t for t in trials if t.seg.any()]
    t0 = time.time()
    cal_idx = np.linspace(0, len(trials) - 1, min(N_CAL, len(trials))).astype(int)
    cal = [trials[i] for i in cal_idx]
    model = SA.calibrate_playback_model(
        recordings=[t.seg for t in cal],
        recording_t0s=[t.seg_t0 for t in cal],
        recording_rate=meta["nidaq_sr"],
        waveforms=[t.wf for t in cal],
        nominal_rate=meta["wave_sr"],
        search_windows=[(t.trigger + SEARCH[0], t.trigger + SEARCH[1]) for t in cal],
    )
    t_cal = time.time() - t0
    rows = []
    t_new = t_old = t_env = 0.0
    for t in trials:
        a = time.time()
        res = SA.align_sound_waveform(
            t.seg, t.seg_t0, t.sr, t.wf, model, t.trigger + SEARCH[0], t.trigger + SEARCH[1]
        )
        b = time.time()
        old, old_q = M.baseline(t)
        c = time.time()
        env, _ = M.baseline_env(t)
        d = time.time()
        t_new += b - a
        t_old += c - b
        t_env += d - c
        rows.append(
            dict(
                session=meta["session"], stim=meta["stim"].split("_")[0], mode=meta["sound_mode"],
                date=meta["date"], has_sync=meta["has_sync_line"], trial=t.trial_idx, name=t.name,
                freq=dominant_freq(t.wf, t.wsr), dur=t.dur, trigger=t.trigger, gt_on=t.gt_on,
                gt_off=t.gt_off, n_wf=len(t.wf),
                new=res.onset_time, new_env=res.envelope_onset_time,
                r_car=res.carrier_correlation, r_env=res.envelope_correlation,
                legacy=old, legacy_env=env,
            )
        )
    calrow = dict(
        session=meta["session"], stim=meta["stim"].split("_")[0], mode=meta["sound_mode"],
        date=meta["date"], has_sync=meta["has_sync_line"], recon=model.reconstruction,
        rate=model.sampling_rate, ppm=model.ppm, cal_r=model.calibration_correlation,
        n_trials=len(trials), sec_calibration=t_cal, sec_new=t_new, sec_legacy=t_old,
        sec_legacy_env=t_env, nidaq_sr=meta["nidaq_sr"],
    )
    print(f"{os.path.basename(path)} {model.reconstruction} {model.ppm:+.1f}ppm r={model.calibration_correlation:.3f} "
          f"cal {t_cal:.1f}s new {t_new:.1f}s legacy {t_old:.1f}s env {t_env:.1f}s", flush=True)
    return rows, calrow


def main(n_workers: int) -> None:
    files = sorted(glob.glob(str(M.CACHE / "*.npz")))
    rows, cals = [], []
    with cf.ProcessPoolExecutor(n_workers) as ex:
        futs = {ex.submit(run_file, f): f for f in files}
        for fut in cf.as_completed(futs):
            try:
                r, c = fut.result()
            except Exception as e:  # noqa: BLE001
                print("FAIL", futs[fut], repr(e), flush=True)
                continue
            rows += r
            cals.append(c)
    pd.DataFrame(rows).to_csv(OUT / "trials.csv", index=False)
    pd.DataFrame(cals).sort_values(["session", "stim"]).to_csv(OUT / "calibration.csv", index=False)
    print("wrote", len(rows), "trials,", len(cals), "files")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
