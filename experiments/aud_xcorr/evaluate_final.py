"""Final evaluation of the shipped library pipeline on all cached stim files.

Mirrors `npc_samstim.waveforms.align_sound` (minus the constant onset
correction, which is estimated from these results with leave-one-session-out
in `analyze.py`):
  - NI-DAQ clock: `SampleClock.from_barcodes` (all matched barcodes, smoothed)
    vs. linear (npc_ephys) for comparison
  - per-file playback-model calibration on 12 evenly spaced trials
  - `align_sound_waveform` per trial

Writes results/final_trials.csv and results/final_calibration.csv.
Legacy-method columns are merged from results/trials.csv (identical inputs).
"""
from __future__ import annotations

import concurrent.futures as cf
import glob
import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd

import clockmap as C
import methods as M
from npc_samstim import sound_alignment as SA

OUT = pathlib.Path(__file__).parent / "results"
SEARCH = (-0.02, 0.25)
N_CAL = 12


def dominant_freq(wf, sr):
    F = np.abs(np.fft.rfft(wf))
    return float(np.fft.rfftfreq(len(wf), 1 / sr)[np.argmax(F)])


PARTS = OUT / "final_parts"
PARTS.mkdir(exist_ok=True)


def run_file(path: str):
    part = PARTS / (pathlib.Path(path).stem + ".pkl")
    if part.exists():
        return pd.read_pickle(part)
    result = _run_file(path)
    pd.to_pickle(result, part)
    return result


def _run_file(path: str):
    meta, trials = M.load(pathlib.Path(path))
    trials = [t for t in trials if t.seg.any()]
    k, tt, lin_sr, lin_start = C.load_pairs(meta["session"])
    assert np.isclose(lin_sr, meta["nidaq_sr"]) and np.isclose(lin_start, meta["nidaq_start"])
    clock = SA.SampleClock.from_barcodes(k, tt)
    linear = SA.SampleClock.linear(meta["nidaq_start"], meta["nidaq_sr"])

    def s0(t):
        return int(round((t.seg_t0 - meta["nidaq_start"]) * meta["nidaq_sr"]))

    local = {id(t): clock.local_linear(s0(t), len(t.seg)) for t in trials}
    t0 = time.time()
    cal_idx = np.linspace(0, len(trials) - 1, min(N_CAL, len(trials))).astype(int)
    cal = [trials[i] for i in cal_idx]
    model = SA.calibrate_playback_model(
        recordings=[t.seg for t in cal],
        recording_t0s=[local[id(t)][0] for t in cal],
        recording_rate=float(np.median([local[id(t)][1] for t in cal])),
        waveforms=[t.wf for t in cal],
        nominal_rate=meta["wave_sr"],
        search_windows=[(t.trigger + SEARCH[0], t.trigger + SEARCH[1]) for t in cal],
    )
    t_cal = time.time() - t0
    rows = []
    for t in trials:
        a = time.time()
        lt0, lsr = local[id(t)]
        res = SA.align_sound_waveform(
            t.seg, lt0, lsr, t.wf, model, t.trigger + SEARCH[0], t.trigger + SEARCH[1]
        )
        dt = time.time() - a
        # same estimate expressed on the linear (npc_ephys) clock, for comparison
        k_est = clock.time_to_sample(res.onset_time)
        rows.append(
            dict(
                session=meta["session"], stim=meta["stim"].split("_")[0], mode=meta["sound_mode"],
                date=meta["date"], has_sync=meta["has_sync_line"], trial=t.trial_idx, name=t.name,
                freq=dominant_freq(t.wf, t.wsr), dur=t.dur, trigger=t.trigger, gt_on=t.gt_on,
                final=res.onset_time, final_linear_clock=float(linear.sample_to_time(k_est)),
                final_env=res.envelope_onset_time, r_car=res.carrier_correlation,
                r_env=res.envelope_correlation, sec=dt,
                legacy_barcode_clock=np.nan,  # filled below
            )
        )
    calrow = dict(
        session=meta["session"], stim=meta["stim"].split("_")[0], mode=meta["sound_mode"],
        date=meta["date"], has_sync=meta["has_sync_line"], recon=model.reconstruction,
        rate=model.sampling_rate, ppm=model.ppm, cal_r=model.calibration_correlation,
        n_trials=len(trials), sec_calibration=t_cal, sec_align=sum(r["sec"] for r in rows),
    )
    print(f"{os.path.basename(path)} {model.reconstruction} {model.ppm:+.1f}ppm "
          f"r={model.calibration_correlation} cal {t_cal:.1f}s align {calrow['sec_align']:.1f}s", flush=True)
    return rows, calrow


def main(n_workers: int) -> None:
    files = sorted(glob.glob(str(M.CACHE / "*[!s].npz")))  # exclude *__barcodes.npz
    files = [f for f in files if not f.endswith("__barcodes.npz")]
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
    d = pd.DataFrame(rows)
    old = pd.read_csv(OUT / "trials.csv")[["session", "stim", "trial", "legacy", "legacy_env", "new"]]
    old = old.rename(columns={"new": "prelim"})
    d = d.merge(old, on=["session", "stim", "trial"], how="left")
    # legacy estimates re-mapped to the barcode clock (isolates estimator vs clock effects)
    parts = []
    for s, g in d.groupby("session"):
        k, tt, lsr, lst = C.load_pairs(s)
        clock = SA.SampleClock.from_barcodes(k, tt)
        g = g.copy()
        g["legacy_barcode_clock"] = clock.sample_to_time((g.legacy.values - lst) * lsr)
        parts.append(g)
    d = pd.concat(parts)
    d.to_csv(OUT / "final_trials.csv", index=False)
    pd.DataFrame(cals).sort_values(["session", "stim"]).to_csv(OUT / "final_calibration.csv", index=False)
    print("wrote", len(d), "trials,", len(cals), "files")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
