"""Cache barcode times on sync and on the NI-DAQ (sample numbers) for each session,
plus the linear timing npc_ephys uses, so clock-mapping variants can be evaluated."""
from __future__ import annotations

import _creds  # noqa: F401

import concurrent.futures as cf
import json
import pathlib
import sys
import traceback

import numpy as np
import npc_ephys
import npc_sync
import upath

CACHE = pathlib.Path(__file__).parent / "cache"


def fetch(s: dict) -> str:
    session = s["session"]
    out = CACHE / f"{session}__barcodes.npz"
    if out.exists():
        return f"skip {session}"
    sync = npc_sync.get_sync_data(upath.UPath(f"s3://aind-ephys-data/{session}/behavior/{s['sync']}"))
    candidates = []
    for d in s["rec_dirs"]:
        rd = (f"s3://{d}",)
        try:
            dev = npc_ephys.get_pxi_nidaq_info(rd)
            timing = next(npc_ephys.get_ephys_timing_on_sync(sync=sync, devices=(dev,)))
            n = npc_ephys.get_pxi_nidaq_data(*rd, device_name=dev.device.name).shape[0]
        except Exception:  # noqa: BLE001
            continue
        candidates.append((n / timing.sampling_rate, dev, timing))
    _, dev, timing = max(candidates, key=lambda c: c[0])
    s_t, s_id = npc_ephys.barcodes.extract_barcodes_from_times(
        on_times=sync.get_rising_edges("barcode_ephys", units="seconds"),
        off_times=sync.get_falling_edges("barcode_ephys", units="seconds"),
        total_time_on_line=sync.total_seconds,
    )
    dv = dev.device
    p_t, p_id = npc_ephys.barcodes.extract_barcodes_from_times(
        on_times=dv.ttl_sample_numbers[dv.ttl_states > 0] / dev.sampling_rate,
        off_times=dv.ttl_sample_numbers[dv.ttl_states < 0] / dev.sampling_rate,
        total_time_on_line=dv.ttl_sample_numbers[-1] / dev.sampling_rate,
    )
    np.savez(
        out, sync_t=s_t, sync_id=s_id, nidaq_t_nominal=p_t, nidaq_id=p_id,
        nominal_sr=dev.sampling_rate, lin_sr=timing.sampling_rate, lin_start=timing.start_time,
    )
    return f"OK {session} sync {len(s_t)} nidaq {len(p_t)}"


if __name__ == "__main__":
    S = json.load(open(pathlib.Path(__file__).parent / "sessions.json"))
    if sys.argv[1:]:
        S = [s for s in S if any(f in s["session"] for f in sys.argv[1:])]
    with cf.ThreadPoolExecutor(6) as ex:
        def safe(s):
            try:
                return fetch(s)
            except Exception as e:  # noqa: BLE001
                return f"FAIL {s['session']} {e!r}"[:300]

        for r in ex.map(safe, S):
            print(r, flush=True)
