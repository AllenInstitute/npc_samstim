"""Candidate audio-onset estimators operating on cached NI-DAQ segments.

Every method takes a `Trial` and returns (onset_time_on_sync, quality).
"""
from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np
import scipy.signal

CACHE = pathlib.Path(__file__).parent / "cache"


@dataclasses.dataclass
class Trial:
    seg: np.ndarray  # int16 NI-DAQ ch1 samples
    seg_t0: float  # sync time of seg[0]
    sr: float  # NI-DAQ sampling rate (barcode-corrected, on sync clock)
    nidaq_start: float  # sync time of NI-DAQ sample 0
    wf: np.ndarray  # ideal waveform (float32)
    wsr: float  # waveform sampling rate
    trigger: float  # sync time of trigger vsync
    dur: float
    name: str
    gt_on: float  # ground truth (nan if unavailable)
    gt_off: float
    trial_idx: int


def load(npz_path: pathlib.Path) -> tuple[dict, list[Trial]]:
    z = np.load(npz_path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    segs = np.split(z["segs"], np.cumsum(z["seg_len"])[:-1])
    wfs = np.split(z["wfs"], np.cumsum(z["wf_len"])[:-1])
    trials = [
        Trial(
            seg=segs[i], seg_t0=float(z["seg_t0"][i]), sr=meta["nidaq_sr"],
            nidaq_start=meta["nidaq_start"], wf=wfs[i], wsr=meta["wave_sr"],
            trigger=float(z["trigger"][i]), dur=float(z["dur"][i]), name=str(z["names"][i]),
            gt_on=float(z["gt_on"][i]), gt_off=float(z["gt_off"][i]),
            trial_idx=int(z["trial_idx"][i]),
        )
        for i in range(len(segs))
    ]
    return meta, trials


# --------------------------------------------------------------------------- #
# Baseline: exact replica of npc_samstim.waveforms.xcorr (v0.1.15)
# --------------------------------------------------------------------------- #
def _baseline_window(t: Trial, padding_sec: float = 0.15):
    """Return the NI-DAQ samples and `nidaq_times` exactly as xcorr() builds them."""
    sr = t.sr
    padding_samples = int(padding_sec * sr)
    trig_nidaq = t.trigger - t.nidaq_start
    on_s = round(trig_nidaq * sr)
    off_s = round((trig_nidaq + t.dur) * sr)
    my_s0 = round((t.seg_t0 - t.nidaq_start) * sr)
    a = on_s - padding_samples - my_s0
    b = off_s + padding_samples - my_s0
    assert a >= 0 and b <= len(t.seg), (a, b, len(t.seg))
    v = t.seg[a:b]
    nidaq_times = np.arange((off_s + padding_samples) - (on_s - padding_samples)) / sr - padding_sec
    return v, nidaq_times


def _interp_waveform(t: Trial) -> tuple[np.ndarray, np.ndarray]:
    wt = np.arange(0, t.dur, 1 / t.sr)
    w_ts = np.linspace(0.0, len(t.wf) / t.wsr, len(t.wf), endpoint=False)
    return wt, np.interp(wt, w_ts, t.wf.astype(np.float64))


def find_envelope(s, t, dmin=1, dmax=1):
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1
    lmax = lmax[[i + np.argmax(s[lmax[i : i + dmax]]) for i in range(0, len(lmax), dmax)]]
    return None, np.interp(t, t[lmax], s[lmax])


def baseline(t: Trial, use_envelope: bool = False, padding_sec: float = 0.15):
    v, nidaq_times = _baseline_window(t, padding_sec)
    wt, w = _interp_waveform(t)
    if use_envelope:
        _, v = find_envelope(v.astype(np.float64), nidaq_times)
        _, w = find_envelope(w, wt)
    c = scipy.signal.correlate(v.astype(np.float64), w, mode="valid", method="fft")
    k = int(np.argmax(c))
    return t.trigger + nidaq_times[k], float(c[k])


def baseline_env(t: Trial):
    return baseline(t, use_envelope=True)


# --------------------------------------------------------------------------- #
# Improved building blocks
# --------------------------------------------------------------------------- #
def _seg_times(t: Trial) -> np.ndarray:
    """Exact sync times of each cached NI-DAQ sample."""
    return t.seg_t0 + np.arange(len(t.seg)) / t.sr


def zoh_template(t: Trial, tau: float, n: int) -> np.ndarray:
    """Value of a zero-order-hold DAC output of `wf` (starting at sync time tau)
    sampled at the NI-DAQ sample times seg_t0 + k/sr, for k in 0..n-1."""
    ts = t.seg_t0 + np.arange(n) / t.sr - tau
    idx = np.floor(ts * t.wsr).astype(np.int64)
    out = np.zeros(n)
    ok = (idx >= 0) & (idx < len(t.wf))
    out[ok] = t.wf[idx[ok]]
    return out


def _ncc_valid(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Normalised cross-correlation (Pearson r) of template w at every valid lag in v."""
    v = v.astype(np.float64)
    w = w - w.mean()
    m = len(w)
    num = scipy.signal.correlate(v, w, mode="valid", method="fft")
    cs = np.cumsum(np.r_[0.0, v])
    cs2 = np.cumsum(np.r_[0.0, v * v])
    s = cs[m:] - cs[:-m]
    s2 = cs2[m:] - cs2[:-m]
    var = np.maximum(s2 - s * s / m, 1e-12)
    return num / (np.sqrt(var) * np.linalg.norm(w))


def coarse_ncc(t: Trial, search=(-0.1, 0.15), resample_template=True):
    """Integer-sample NCC over the lag search window (relative to trigger).

    Template = ideal waveform resampled to NI-DAQ rate. Returns
    (onset estimate, r at peak, r array, lag-times array).
    """
    seg_times = _seg_times(t)
    if resample_template:
        # band-limited resampling (polyphase) instead of linear interp
        w = scipy.signal.resample_poly(t.wf.astype(np.float64), 1, 1, padtype="line")
        wt, w = _interp_waveform(t)
    else:
        wt, w = _interp_waveform(t)
    r = _ncc_valid(t.seg, w)
    lag_times = seg_times[: len(r)]  # onset time if template starts at sample k
    lo = np.searchsorted(lag_times, t.trigger + search[0])
    hi = np.searchsorted(lag_times, t.trigger + search[1])
    rr = r[lo:hi]
    k = int(np.argmax(rr)) + lo
    return lag_times[k], float(r[k]), r, lag_times, k


def parabolic(r: np.ndarray, k: int) -> float:
    if k <= 0 or k >= len(r) - 1:
        return 0.0
    y0, y1, y2 = r[k - 1], r[k], r[k + 1]
    d = y0 - 2 * y1 + y2
    return 0.0 if d == 0 else 0.5 * (y0 - y2) / d


def ncc_parabolic(t: Trial):
    on, q, r, lt, k = coarse_ncc(t)
    return on + parabolic(r, k) / t.sr, q


def ncc_int(t: Trial):
    on, q, *_ = coarse_ncc(t)
    return on, q


def zoh_refine(t: Trial, step=2e-6, half_width=None, base=None):
    """Coarse NCC, then a fine grid search over tau in +/- 1.5 NI-DAQ samples,
    with a ZOH model of the DAC output point-sampled at the NI-DAQ times
    (reproduces aliasing of content above the NI-DAQ Nyquist)."""
    if base is None:
        on, q, r, lt, k = coarse_ncc(t)
    else:
        on, q = base
    hw = half_width if half_width is not None else 1.5 / t.sr
    # restrict to segment covering the stimulus (plus a margin)
    seg_times = _seg_times(t)
    a = max(int(np.searchsorted(seg_times, on - 0.005)), 0)
    b = min(int(np.searchsorted(seg_times, on + t.dur + 0.005)), len(t.seg))
    v = t.seg[a:b].astype(np.float64)
    v = v - v.mean()
    sub = dataclasses.replace(t, seg_t0=float(seg_times[a]))
    taus = np.arange(on - hw, on + hw + step / 2, step)
    best, best_r = on, -np.inf
    rs = np.empty(len(taus))
    for i, tau in enumerate(taus):
        w = zoh_template(sub, tau, len(v))
        w = w - w.mean()
        rs[i] = float(v @ w / (np.linalg.norm(v) * np.linalg.norm(w) + 1e-12))
    i = int(np.argmax(rs))
    best, best_r = taus[i], rs[i]
    # parabolic on the fine grid too
    best += parabolic(rs, i) * step
    return best, float(best_r)


# --------------------------------------------------------------------------- #
# v2: explicit DAC model (rate + ZOH/linear), coarse NCC, multi-candidate refine
# --------------------------------------------------------------------------- #
def dac_template(wf, fs_eff, sr, tau_frac=0.0, model="zoh", n=None):
    """Analog model of `wf` played at `fs_eff`, point-sampled at NI-DAQ rate `sr`.
    Sample k is at time (k + tau_frac)/sr after waveform onset... i.e. template
    value at t_k = k/sr - tau_frac/sr relative to onset."""
    dur = len(wf) / fs_eff
    if n is None:
        n = int(np.ceil(dur * sr)) + 1
    ts = np.arange(n) / sr - tau_frac / sr
    if model == "zoh":
        idx = np.floor(ts * fs_eff).astype(np.int64)
        out = np.zeros(n)
        ok = (idx >= 0) & (idx < len(wf))
        out[ok] = wf[idx[ok]]
        return out
    elif model == "linear":
        return np.interp(ts, np.arange(len(wf)) / fs_eff, wf.astype(np.float64), left=0, right=0)
    raise ValueError(model)


def _local_maxima(r, k_top, min_sep):
    order = np.argsort(r)[::-1]
    picked = []
    for i in order:
        if all(abs(i - p) >= min_sep for p in picked):
            picked.append(int(i))
        if len(picked) >= k_top:
            break
    return picked


def v2(t: Trial, fs_eff=None, model="zoh", search=(-0.1, 0.15), k_top=5, step_frac=0.02,
       onset_window=None, return_r=False):
    """NCC with a DAC-model template; refine top-k coarse peaks at sub-sample tau.

    onset_window: if given (seconds), only the first `onset_window` s of the
    waveform are used for refinement (reduces sensitivity to rate error).
    """
    fs_eff = fs_eff or t.wsr
    sr = t.sr
    seg = t.seg.astype(np.float64)
    seg_times = _seg_times(t)
    w0 = dac_template(t.wf, fs_eff, sr, 0.0, model)
    r = _ncc_valid(seg, w0)
    lag_times = seg_times[: len(r)]
    lo = max(int(np.searchsorted(lag_times, t.trigger + search[0])), 1)
    hi = min(int(np.searchsorted(lag_times, t.trigger + search[1])), len(r) - 1)
    cands = [lo + c for c in _local_maxima(r[lo:hi], k_top, min_sep=2)]
    fracs = np.arange(-1.0, 1.0 + 1e-9, step_frac)
    best = (-np.inf, None)
    n = len(w0) if onset_window is None else int(onset_window * sr)
    for c in cands:
        for f in fracs:
            # template whose onset is at seg sample c + f
            w = dac_template(t.wf, fs_eff, sr, f, model, n=len(w0) + 2)
            a = c - 1
            v = seg[a : a + len(w)]
            w = w[: len(v)][:n + 2]
            v = v[: len(w)]
            wm = w - w.mean()
            vm = v - v.mean()
            rr = float(vm @ wm / (np.linalg.norm(vm) * np.linalg.norm(wm) + 1e-12))
            if rr > best[0]:
                best = (rr, seg_times[a] + (1 + f) / sr)
    if return_r:
        return best[1], best[0], r, lag_times
    return best[1], best[0]


def v2_nominal(t):
    return v2(t, fs_eff=t.wsr, model="zoh")


def v2_daqrate(t):
    return v2(t, fs_eff=80e6 / round(80e6 / t.wsr), model="zoh")


def v2_daqrate_linear(t):
    return v2(t, fs_eff=80e6 / round(80e6 / t.wsr), model="linear")


def v2_nominal_onset20(t):
    return v2(t, fs_eff=t.wsr, model="zoh", onset_window=0.02)


# --------------------------------------------------------------------------- #
# v3: envelope stage (robust) + carrier stage (precise)
# --------------------------------------------------------------------------- #
DAQ_RATE = lambda fs: 80e6 / round(80e6 / fs)  # noqa: E731  NI AO sample clock from 80 MHz timebase


def moving_rms(x, n):
    x = x - np.mean(x)
    k = np.ones(n) / n
    return np.sqrt(np.maximum(np.convolve(x * x, k, mode="same"), 0))


def envelope_stage(t: Trial, fs_eff, search=(-0.1, 0.15), win_sec=0.001, n_frac=1, model="zoh"):
    sr = t.sr
    n = max(int(round(win_sec * sr)), 1)
    seg = t.seg.astype(np.float64)
    ev = moving_rms(seg, n)
    seg_times = _seg_times(t)
    best = (-np.inf, None)
    for f in np.arange(n_frac) / n_frac:
        w = dac_template(t.wf, fs_eff, sr, f, model)
        # pad template with silence either side so the ramps are matched
        pad = 2 * n
        w = np.r_[np.zeros(pad), w, np.zeros(pad)]
        ew = moving_rms(w, n)
        r = _ncc_valid(ev, ew)
        lag_times = seg_times[: len(r)] + (pad - f) / sr  # onset time for lag k
        lo = max(int(np.searchsorted(lag_times, t.trigger + search[0])), 1)
        hi = min(int(np.searchsorted(lag_times, t.trigger + search[1])), len(r) - 1)
        k = lo + int(np.argmax(r[lo:hi]))
        on = lag_times[k] + parabolic(r, k) / sr
        if r[k] > best[0]:
            best = (float(r[k]), on)
    return best[1], best[0]


def env_daq(t):
    return envelope_stage(t, DAQ_RATE(t.wsr))


def env_daq_2ms(t):
    return envelope_stage(t, DAQ_RATE(t.wsr), win_sec=0.002)


def env_daq_frac4(t):
    return envelope_stage(t, DAQ_RATE(t.wsr), n_frac=4)


def env_nominal(t):
    return envelope_stage(t, t.wsr)


# --------------------------------------------------------------------------- #
# v4: clean two-stage estimator (time conventions fixed)
#   template sample k  <->  time (k - frac)/sr after waveform onset
# --------------------------------------------------------------------------- #
def zoh_on_grid(wf, fs_eff, sr, frac=0.0, n=None):
    """ZOH DAC output sampled at times (k - frac)/sr after onset, k=0..n-1."""
    if n is None:
        n = int(np.ceil(len(wf) / fs_eff * sr)) + 2
    idx = np.floor((np.arange(n) - frac) / sr * fs_eff).astype(np.int64)
    out = np.zeros(n)
    ok = (idx >= 0) & (idx < len(wf))
    out[ok] = wf[idx[ok]]
    return out


def ambiguity_halfwidth(w, sr, cap=0.5e-3, floor_=1.5):
    """Half the lag of the first strong autocorrelation side-peak of template w
    (i.e. half a period for tones). Returns samples (float)."""
    w = w - w.mean()
    n = len(w)
    maxlag = int(cap * 2 * sr) + 2
    ac = scipy.signal.correlate(w, w, mode="full", method="fft")[n - 1 : n - 1 + maxlag]
    ac = ac / ac[0]
    pk = scipy.signal.find_peaks(ac, height=0.5)[0]
    if len(pk):
        return max(pk[0] / 2, floor_)
    return cap * sr


def envelope_ncc(v, w, n_rms):
    ev = moving_rms(v, n_rms)
    pad = 2 * n_rms
    ew = moving_rms(np.r_[np.zeros(pad), w, np.zeros(pad)], n_rms)
    return _ncc_valid(ev, ew), pad


def v4(t: Trial, fs_eff=None, search=(-0.1, 0.15), env_win=0.001, n_frac=1, refine="grid",
       stage_b=True, return_debug=False):
    sr = t.sr
    fs_eff = fs_eff or DAQ_RATE(t.wsr)
    v = t.seg.astype(np.float64)
    st0 = t.seg_t0
    n_rms = max(int(round(env_win * sr)), 1)
    # ---- stage A: envelope NCC (optionally over several sub-sample fractions)
    bestA = (-np.inf, None)
    for frac in np.arange(n_frac) / n_frac:
        w = zoh_on_grid(t.wf, fs_eff, sr, frac)
        r, pad = envelope_ncc(v, w, n_rms)
        # lag k: padded template[0] at seg sample k -> template sample 0 at k+pad
        # onset time = st0 + (k + pad + frac)/sr
        on_k = st0 + (np.arange(len(r)) + pad + frac) / sr
        lo = max(int(np.searchsorted(on_k, t.trigger + search[0])), 1)
        hi = min(int(np.searchsorted(on_k, t.trigger + search[1])), len(r) - 1)
        k = lo + int(np.argmax(r[lo:hi]))
        on = on_k[k] + parabolic(r, k) / sr
        if r[k] > bestA[0]:
            bestA = (float(r[k]), on)
    rA, onA = bestA
    if not stage_b:
        return onA, rA
    # ---- stage B: carrier NCC within +/- ambiguity half-width of stage A
    w0 = zoh_on_grid(t.wf, fs_eff, sr, 0.0)
    hw = ambiguity_halfwidth(w0, sr)
    r = _ncc_valid(v, w0)
    on_k = st0 + np.arange(len(r)) / sr
    kA = (onA - st0) * sr
    lo = max(int(np.ceil(kA - hw)), 1)
    hi = min(int(np.floor(kA + hw)) + 1, len(r) - 1)
    k = lo + int(np.argmax(r[lo:hi]))
    if refine == "parabolic":
        return on_k[k] + parabolic(r, k) / sr, float(r[k])
    # fine grid over fractional offset in [-1, 1] samples around k
    a = k - 2
    m = len(w0) + 4
    seg = v[a : a + m]
    fracs = np.arange(-1.0, 1.0 + 1e-9, 0.02)
    rs = np.empty(len(fracs))
    vm = seg - seg.mean()
    vn = np.linalg.norm(vm)
    for i, f in enumerate(fracs):
        w = zoh_on_grid(t.wf, fs_eff, sr, 2 + f, n=len(seg))  # onset at seg index 2+f
        wm = w - w.mean()
        rs[i] = vm @ wm / (vn * np.linalg.norm(wm) + 1e-12)
    i = int(np.argmax(rs))
    f = fracs[i] + parabolic(rs, i) * 0.02
    return st0 + (a + 2 + f) / sr, float(rs[i])


def v4_envA(t):
    return v4(t, stage_b=False)


def v4_envA_frac8(t):
    return v4(t, stage_b=False, n_frac=8)


def v4_envA_3ms(t):
    return v4(t, stage_b=False, env_win=0.003)


def v4_par(t):
    return v4(t, refine="parabolic")


def v4_grid(t):
    return v4(t, refine="grid")


def v4_grid_nominal(t):
    return v4(t, fs_eff=t.wsr, refine="grid")


def v4_envA_frac16(t):
    return v4(t, stage_b=False, n_frac=16)


def v4_envA_frac32(t):
    return v4(t, stage_b=False, n_frac=32)


def v4_envA_frac16_05ms(t):
    return v4(t, stage_b=False, n_frac=16, env_win=0.0005)


def v4_f16_par(t):
    return v4(t, n_frac=16, refine="parabolic")


def v4_f32_par(t):
    return v4(t, n_frac=32, refine="parabolic")


def v4_f32_grid(t):
    return v4(t, n_frac=32, refine="grid")


# --------------------------------------------------------------------------- #
# reconstruction models for template generation
# --------------------------------------------------------------------------- #
def template_on_grid(wf, fs_eff, sr, frac=0.0, n=None, model="zoh", _cache={}):
    if model == "zoh":
        return zoh_on_grid(wf, fs_eff, sr, frac, n)
    if n is None:
        n = int(np.ceil(len(wf) / fs_eff * sr)) + 2
    ts = (np.arange(n) - frac) / sr
    if model == "linear":
        return np.interp(ts * fs_eff, np.arange(len(wf)), wf.astype(np.float64), left=0, right=0)
    if model == "bandlimited":
        key = (id(wf), len(wf))
        up = _cache.get(key)
        if up is None:
            up = scipy.signal.resample_poly(np.r_[wf.astype(np.float64), np.zeros(8)], 16, 1)
            _cache.clear(); _cache[key] = up
        return np.interp(ts * fs_eff * 16, np.arange(len(up)), up, left=0, right=0)
    raise ValueError(model)


def local_carrier_r(t, onset_guess, fs_eff, model, hw_sec=1e-3, step_frac=0.05):
    """Max carrier NCC over continuous onset in onset_guess +/- hw_sec (fine grid)."""
    sr = t.sr
    v = t.seg.astype(np.float64)
    k0 = int(np.floor((onset_guess - hw_sec - t.seg_t0) * sr))
    n = int(np.ceil(len(t.wf) / fs_eff * sr)) + 2
    nl = int(np.ceil(2 * hw_sec * sr)) + 2
    best = (-np.inf, None)
    if k0 < 0 or k0 + nl + n > len(v):
        return best
    from numpy.lib.stride_tricks import sliding_window_view
    V = sliding_window_view(v[k0 : k0 + nl + n], n)[:nl]  # (nl, n)
    Vm = V - V.mean(1, keepdims=True)
    Vn = np.linalg.norm(Vm, axis=1)
    for f in np.arange(0, 1, step_frac):
        w = template_on_grid(t.wf, fs_eff, sr, f, n, model)
        wm = w - w.mean()
        r = Vm @ wm / (Vn * np.linalg.norm(wm) + 1e-12)
        i = int(np.argmax(r))
        if r[i] > best[0]:
            best = (float(r[i]), t.seg_t0 + (k0 + i + f) / sr)
    return best
