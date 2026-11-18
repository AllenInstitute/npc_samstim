"""Align ideal sound waveforms to an analog recording of the speaker signal.

Used for sessions without a "sound running" sync line (before 2023-08-31), where
the onset of each sound must be found in the NI-DAQ recording of the speaker
signal (PXI NI-DAQ channel 1, ~30 kHz).

Method (validated against the sync-line marker in later sessions; see
`experiments/aud_xcorr` and the accompanying report):

1. **Playback model.** The recorded signal is modelled as the ideal waveform
   played at an *effective* sample rate through a reconstruction stage, then
   point-sampled at the NI-DAQ sample times:

   - ``"zoh"``: zero-order hold (NI-DAQ analog output, ``soundMode='daq'``).
     The AO sample clock is derived from an 80 MHz timebase, so a nominal
     48 kHz rate is really 80 MHz / 1667 = 47990.4 Hz (-200 ppm).
   - ``"bandlimited"``: ideal (sinc) reconstruction, approximating a sound
     card's anti-imaging filter (``soundMode='sound card'`` / ``'internal'``).

   The model (reconstruction type and rate) is calibrated once per stimulus
   file by maximizing the median waveform correlation over a subset of trials.

2. **Envelope stage (robust).** Normalized cross-correlation (NCC) of the
   1 ms moving-RMS envelope of the recording with the envelope of the modelled
   template, over the lag search window. The template is evaluated at several
   sub-sample offsets because aliasing of tones near rational fractions of the
   NI-DAQ rate makes the sampled envelope depend on sub-sample phase. This stage
   avoids the cycle-slips and gross errors (up to ~100 ms) the raw-waveform
   correlation suffers for pure tones.

3. **Carrier stage (precise).** NCC of the raw waveform with the modelled
   template, restricted to +/- half the waveform's autocorrelation side-peak lag
   (half a period for tones) around the envelope estimate, at sub-sample
   resolution.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt
import scipy.fft
import scipy.ndimage
import scipy.signal

logger = logging.getLogger(__name__)

Reconstruction = Literal["zoh", "bandlimited"]

_BANDLIMITED_UPSAMPLING = 16


@dataclasses.dataclass(frozen=True)
class PlaybackModel:
    """How the ideal waveform becomes the analog signal that was recorded."""

    sampling_rate: float
    """Effective rate at which waveform samples were played, on the recording's clock."""
    reconstruction: Reconstruction = "zoh"
    nominal_sampling_rate: float | None = None
    calibration_correlation: float | None = None
    """Median carrier correlation of calibration trials with this model."""

    @property
    def ppm(self) -> float | None:
        if not self.nominal_sampling_rate:
            return None
        return (self.sampling_rate / self.nominal_sampling_rate - 1) * 1e6


@dataclasses.dataclass(frozen=True)
class SoundAlignment:
    onset_time: float
    """Time of waveform sample 0 in the same clock as `recording_t0`. NaN if alignment failed."""
    carrier_correlation: float
    """Pearson r of recorded vs modelled waveform at the estimated onset."""
    envelope_correlation: float
    """Pearson r of recorded vs modelled envelope at the envelope-stage onset."""
    envelope_onset_time: float
    periodic: bool = False
    """Waveform is periodic (tone): `onset_time` is the envelope-stage estimate and
    `carrier_correlation` is not a reliable quality measure."""


def ni_ao_sampling_rate(nominal_rate: float, timebase: float = 80e6) -> float:
    """Actual NI-DAQmx AO sample-clock rate for a requested rate (integer divisor of timebase).

    >>> round(ni_ao_sampling_rate(48000), 2)
    47990.4
    """
    return timebase / round(timebase / nominal_rate)


# --------------------------------------------------------------------------- #
# recording clock
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class SampleClock:
    """Maps a recording's sample index to time on the sync clock (and back).

    `npc_ephys` aligns NI-DAQ samples to sync with a straight line through the
    first and last matched barcodes. The two clocks drift non-linearly
    (validated sessions show +/-100 us wander over a session), so
    `from_barcodes` instead uses a least-squares line plus a moving average of
    the residuals at every matched barcode (~30 s apart), interpolated.

    >>> clock = SampleClock.linear(start_time=10.0, sampling_rate=30000.0)
    >>> float(clock.sample_to_time(30000))
    11.0
    >>> float(clock.time_to_sample(11.0))
    30000.0
    """

    slope: float
    """sec per sample"""
    intercept: float
    """sync time of sample 0 (sec)"""
    knot_samples: npt.NDArray[np.float64] = dataclasses.field(
        default_factory=lambda: np.array([0.0])
    )
    knot_residuals: npt.NDArray[np.float64] = dataclasses.field(
        default_factory=lambda: np.array([0.0])
    )

    @classmethod
    def linear(cls, start_time: float, sampling_rate: float) -> SampleClock:
        return cls(slope=1 / sampling_rate, intercept=start_time)

    @classmethod
    def from_barcodes(
        cls,
        sample_numbers: npt.ArrayLike,
        sync_times: npt.ArrayLike,
        smoothing_window: int = 9,
        max_residual_sec: float = 5e-3,
    ) -> SampleClock:
        """`sample_numbers[i]` on the recording occurred at `sync_times[i]`."""
        k = np.asarray(sample_numbers, dtype=np.float64)
        t = np.asarray(sync_times, dtype=np.float64)
        order = np.argsort(k)
        k, t = k[order], t[order]
        if len(k) < 2:
            raise ValueError("At least two matched barcodes are needed")
        slope, intercept = np.polyfit(k, t, 1)
        res = t - (slope * k + intercept)
        ok = np.abs(res - np.median(res)) < max_residual_sec  # drop mismatched barcodes
        k, t = k[ok], t[ok]
        slope, intercept = np.polyfit(k, t, 1)
        res = t - (slope * k + intercept)
        half = smoothing_window // 2
        smoothed = np.array(
            [res[max(0, i - half) : i + half + 1].mean() for i in range(len(res))]
        )
        return cls(float(slope), float(intercept), k, smoothed)

    @property
    def sampling_rate(self) -> float:
        """Mean sampling rate on the sync clock."""
        return 1 / self.slope

    def sample_to_time(self, sample: npt.ArrayLike) -> npt.NDArray[np.float64]:
        k = np.asarray(sample, dtype=np.float64)
        return (
            self.slope * k
            + self.intercept
            + np.interp(k, self.knot_samples, self.knot_residuals)
        )

    def time_to_sample(self, time: npt.ArrayLike) -> npt.NDArray[np.float64]:
        t = np.asarray(time, dtype=np.float64)
        k = (t - self.intercept) / self.slope
        for _ in range(
            3
        ):  # residual correction varies slowly: fixed-point converges fast
            k = (
                t
                - self.intercept
                - np.interp(k, self.knot_samples, self.knot_residuals)
            ) / self.slope
        return k

    def local_linear(self, first_sample: int, num_samples: int) -> tuple[float, float]:
        """(time of `first_sample`, local sampling rate) over a short segment."""
        t0, t1 = self.sample_to_time([first_sample, first_sample + num_samples])
        return float(t0), float(num_samples / (t1 - t0))


# --------------------------------------------------------------------------- #
# template generation
# --------------------------------------------------------------------------- #
class _TemplateFactory:
    """Modelled analog waveform, point-sampled on the recording's sample grid."""

    def __init__(
        self,
        waveform: npt.ArrayLike,
        model: PlaybackModel,
        recording_rate: float,
    ) -> None:
        self.wf = np.asarray(waveform, dtype=np.float64)
        self.model = model
        self.sr = float(recording_rate)
        self.n = int(np.ceil(len(self.wf) / model.sampling_rate * self.sr)) + 2
        self._up: npt.NDArray[np.float64] | None = None

    def __call__(
        self, frac: float = 0.0, n: int | None = None
    ) -> npt.NDArray[np.float64]:
        """Template sample k is the modelled signal at time (k - frac)/sr after onset."""
        return self.batch(np.atleast_1d(frac), n)[0]

    def batch(
        self, fracs: npt.NDArray[np.float64], n: int | None = None
    ) -> npt.NDArray[np.float64]:
        """Templates for several sub-sample phases at once: shape (len(fracs), n)."""
        n = self.n if n is None else n
        x = (np.arange(n)[None, :] - np.asarray(fracs, dtype=np.float64)[:, None]) * (
            self.model.sampling_rate / self.sr
        )  # in waveform samples
        if self.model.reconstruction == "zoh":
            idx = np.floor(x).astype(np.int64)
            out = np.zeros(x.shape)
            ok = (idx >= 0) & (idx < len(self.wf))
            out[ok] = self.wf[idx[ok]]
            return out
        if self.model.reconstruction == "bandlimited":
            if self._up is None:
                self._up = scipy.signal.resample_poly(
                    np.r_[self.wf, np.zeros(8)], _BANDLIMITED_UPSAMPLING, 1
                )
            return np.interp(
                x * _BANDLIMITED_UPSAMPLING,
                np.arange(len(self._up)),
                self._up,
                left=0,
                right=0,
            ).reshape(x.shape)
        raise ValueError(f"Unknown reconstruction {self.model.reconstruction!r}")


# --------------------------------------------------------------------------- #
# correlation helpers
# --------------------------------------------------------------------------- #
def _moving_rms(x: npt.NDArray[np.float64], n: int) -> npt.NDArray[np.float64]:
    """Centered moving RMS (window n samples, zeros beyond the ends) of mean-subtracted x."""
    x = x - x.mean()
    return np.sqrt(
        np.maximum(
            scipy.ndimage.uniform_filter1d(x * x, n, mode="constant", cval=0.0), 0.0
        )
    )


def _moving_rms_rows(x: npt.NDArray[np.float64], n: int) -> npt.NDArray[np.float64]:
    """`_moving_rms` applied to each row of a 2D array."""
    x = x - x.mean(axis=1, keepdims=True)
    return np.sqrt(
        np.maximum(
            scipy.ndimage.uniform_filter1d(x * x, n, axis=1, mode="constant", cval=0.0),
            0.0,
        )
    )


class _SlidingNCC:
    """Normalized cross-correlation of many templates against one fixed signal
    (valid lags only), re-using the signal's FFT and running sums."""

    def __init__(self, v: npt.NDArray[np.float64], max_template_len: int) -> None:
        self.v = v
        self.nfft = scipy.fft.next_fast_len(len(v) + max_template_len)
        self.V = scipy.fft.rfft(v, self.nfft)
        self.cs = np.cumsum(np.r_[0.0, v])
        self.cs2 = np.cumsum(np.r_[0.0, v * v])

    def __call__(self, w: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        m = len(w)
        n_valid = len(self.v) - m + 1
        if n_valid < 1:
            return np.array([])
        wm = w - w.mean()
        wn = np.linalg.norm(wm)
        W = scipy.fft.rfft(wm[::-1], self.nfft)
        num = scipy.fft.irfft(self.V * W, self.nfft)[m - 1 : m - 1 + n_valid]
        s = self.cs[m:] - self.cs[:-m]
        s2 = self.cs2[m:] - self.cs2[:-m]
        var = np.maximum(s2 - s * s / m, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = num / (np.sqrt(var) * wn)
        r[~np.isfinite(r)] = 0.0
        return r


def _parabolic(y: npt.NDArray[np.float64], k: int) -> float:
    if k <= 0 or k >= len(y) - 1:
        return 0.0
    d = y[k - 1] - 2 * y[k] + y[k + 1]
    return 0.0 if d == 0 else float(0.5 * (y[k - 1] - y[k + 1]) / d)


def _ambiguity_halfwidth(
    template: npt.NDArray[np.float64], sr: float, cap_sec: float
) -> float:
    """Half the lag (samples) of the first strong autocorrelation side-peak
    (half a period for a tone); `cap_sec` if there is none (e.g. noise)."""
    w = template - template.mean()
    maxlag = int(2 * cap_sec * sr) + 2
    nfft = scipy.fft.next_fast_len(2 * len(w))
    W = scipy.fft.rfft(w, nfft)
    ac = scipy.fft.irfft(W * np.conj(W), nfft)[:maxlag]
    if ac[0] <= 0:
        return cap_sec * sr
    ac = ac / ac[0]
    peaks = scipy.signal.find_peaks(ac, height=0.5)[0]
    if len(peaks):
        return max(peaks[0] / 2, 1.5)
    return cap_sec * sr


def _local_carrier_search(
    v: npt.NDArray[np.float64],
    templates: _TemplateFactory,
    k_lo: int,
    k_hi: int,
    n_frac: int,
) -> tuple[float, float]:
    """Max carrier NCC for onsets at recording sample k + f, k in [k_lo, k_hi),
    f in [0, 1) on an `n_frac` grid, plus parabolic refinement over f.
    Returns (onset in recording samples, r)."""
    n = templates.n
    k_lo = max(k_lo, 0)
    k_hi = min(k_hi, len(v) - n)
    if k_hi <= k_lo:
        return np.nan, np.nan
    windows = np.lib.stride_tricks.sliding_window_view(v[k_lo : k_hi + n], n)[
        : k_hi - k_lo
    ]
    vm = windows - windows.mean(axis=1, keepdims=True)
    vn = np.linalg.norm(vm, axis=1)
    fracs = np.arange(n_frac) / n_frac
    W = np.stack([templates(f) for f in fracs], axis=1)  # (n, n_frac)
    W = W - W.mean(axis=0, keepdims=True)
    W = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-300)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = (vm @ W) / vn[:, None]  # (lags, fracs)
    R[~np.isfinite(R)] = -np.inf
    # onset position for (lag i, frac j) = k_lo + i + frac_j: flatten in time order
    r = R.ravel()  # index i*n_frac + j  <->  position k_lo + i + j/n_frac
    best = int(np.argmax(r))
    if not np.isfinite(r[best]):
        return np.nan, np.nan
    offset = (
        _parabolic(r, best)
        if 0 < best < len(r) - 1 and np.isfinite(r[[best - 1, best + 1]]).all()
        else 0.0
    )
    return k_lo + (best + offset) / n_frac, float(r[best])


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def _envelope_search(
    ev: npt.NDArray[np.float64],
    templates: _TemplateFactory,
    n_rms: int,
    pos_lo: float,
    pos_hi: float,
    fracs: npt.NDArray[np.float64],
) -> tuple[float, float]:
    """Best envelope-NCC onset position (recording samples) in [pos_lo, pos_hi),
    over sub-sample template phases `fracs`. Returns (position, r)."""
    pad = 2 * n_rms
    # crop the recording envelope to what's needed for this range of onsets
    a = max(int(np.floor(pos_lo)) - pad - 2, 0)
    b = min(int(np.ceil(pos_hi)) + templates.n + pad + 2, len(ev))
    crop = ev[a:b]
    if len(crop) < templates.n + 2 * pad + 3:
        return np.nan, np.nan
    ncc = _SlidingNCC(crop, templates.n + 2 * pad)
    best = (-np.inf, np.nan)
    for f in fracs:
        ew = _moving_rms(np.r_[np.zeros(pad), templates(f), np.zeros(pad)], n_rms)
        r = ncc(ew)
        # onset (recording samples) for crop-lag k and phase f: a + k + pad + f
        lo = max(int(np.ceil(pos_lo - a - pad - f)), 1)
        hi = min(int(np.floor(pos_hi - a - pad - f)) + 1, len(r) - 1)
        if hi <= lo:
            continue
        k = lo + int(np.argmax(r[lo:hi]))
        if r[k] > best[0]:
            best = (float(r[k]), a + k + pad + f + _parabolic(r, k))
    return best[1], best[0]


def _envelope_search_local(
    ev: npt.NDArray[np.float64],
    templates: _TemplateFactory,
    n_rms: int,
    pos_lo: float,
    pos_hi: float,
    fracs: npt.NDArray[np.float64],
    chunk: int = 64,
) -> tuple[float, float]:
    """As `_envelope_search`, but by direct (matrix) evaluation: efficient for a
    narrow range of onsets and many template phases."""
    pad = 2 * n_rms
    m = templates.n + 2 * pad
    k_lo = int(np.floor(pos_lo)) - pad - 1
    k_hi = int(np.ceil(pos_hi)) - pad + 1
    k_lo, k_hi = max(k_lo, 0), min(k_hi, len(ev) - m)
    if k_hi <= k_lo:
        return np.nan, np.nan
    win = np.lib.stride_tricks.sliding_window_view(ev[k_lo : k_hi + m], m)[
        : k_hi - k_lo
    ]
    wm = win - win.mean(axis=1, keepdims=True)
    wm /= np.linalg.norm(wm, axis=1, keepdims=True) + 1e-300
    best = (-np.inf, np.nan)
    for c0 in range(0, len(fracs), chunk):
        fr = fracs[c0 : c0 + chunk]
        t = templates.batch(fr)
        t = np.concatenate(
            [np.zeros((len(fr), pad)), t, np.zeros((len(fr), pad))], axis=1
        )
        e = _moving_rms_rows(t, n_rms)
        e -= e.mean(axis=1, keepdims=True)
        e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-300
        R = wm @ e.T  # (lags, fracs)
        pos = k_lo + np.arange(R.shape[0])[:, None] + pad + fr[None, :]
        R[(pos < pos_lo) | (pos >= pos_hi)] = -np.inf
        i, j = (int(x) for x in np.unravel_index(int(np.argmax(R)), R.shape))
        if R[i, j] > best[0]:
            # no interpolation across lags: for aliased tones the envelope changes
            # sharply with sub-sample phase, so only grid positions are consistent
            best = (float(R[i, j]), float(pos[i, j]))
    return best[1], best[0]


def align_sound_waveform(
    recording: npt.ArrayLike,
    recording_t0: float,
    recording_rate: float,
    waveform: npt.ArrayLike,
    model: PlaybackModel,
    search_start: float,
    search_stop: float,
    envelope_window_sec: float = 0.001,
    n_envelope_frac: int = 16,
    n_envelope_frac_fine: int = 128,
    fine_halfwidth_sec: float = 1.5e-3,
    n_carrier_frac: int = 32,
    max_ambiguity_sec: float = 0.25e-3,
) -> SoundAlignment:
    """Find the onset of `waveform` in `recording`.

    - `recording`: analog samples; sample k is at time `recording_t0 + k / recording_rate`
    - `search_start`, `search_stop`: range of candidate onset times (same clock)
    - returns onset time of waveform sample 0, plus correlation-based quality metrics

    For periodic waveforms (tones) the envelope-stage estimate is returned,
    after refinement on a dense grid of sub-sample template phases (the sampled
    envelope of an aliased tone depends strongly on sub-sample phase): the
    carrier-phase estimate is cycle-ambiguous and small phase errors in the
    playback model make it pick neighbouring cycles. For aperiodic waveforms
    (noise) the carrier-stage estimate is returned.

    >>> rng = np.random.default_rng(0)
    >>> wf = rng.standard_normal(4800) * np.hanning(4800)   # 100 ms at 48 kHz
    >>> model = PlaybackModel(ni_ao_sampling_rate(48000), "zoh")
    >>> sr = 30000.0
    >>> true_onset = 0.0123456
    >>> rec = _TemplateFactory(wf, model, sr)(frac=true_onset * sr, n=int(0.2 * sr))
    >>> a = align_sound_waveform(rec, 0.0, sr, wf, model, 0.0, 0.05)
    >>> abs(a.onset_time - true_onset) < 2e-6
    True
    >>> tone = np.sin(2 * np.pi * 7000 * np.arange(4800) / 48000) * np.hanning(4800)
    >>> rec = _TemplateFactory(tone, model, sr)(frac=true_onset * sr, n=int(0.2 * sr))
    >>> a = align_sound_waveform(rec, 0.0, sr, tone, model, 0.0, 0.05)
    >>> abs(a.onset_time - true_onset) < 50e-6  # tones: limited by cycle ambiguity of the envelope
    True
    """
    v = np.asarray(recording, dtype=np.float64)
    sr = float(recording_rate)
    templates = _TemplateFactory(waveform, model, sr)
    nan = SoundAlignment(np.nan, np.nan, np.nan, np.nan)
    if len(v) < templates.n + 2 or not np.any(v != v[0]):
        return nan

    # ---- stage A: envelope NCC; coarse over the search window, then a dense
    # grid of sub-sample template phases close to the coarse estimate
    n_rms = max(int(round(envelope_window_sec * sr)), 1)
    ev = _moving_rms(v, n_rms)
    pos_lo = (search_start - recording_t0) * sr
    pos_hi = (search_stop - recording_t0) * sr
    pos_a, r_env = _envelope_search(
        ev,
        templates,
        n_rms,
        pos_lo,
        pos_hi,
        np.arange(n_envelope_frac) / n_envelope_frac,
    )
    if not np.isfinite(pos_a):
        return nan
    hw = _ambiguity_halfwidth(templates(0.0), sr, max_ambiguity_sec)
    periodic = hw < max_ambiguity_sec * sr
    if n_envelope_frac_fine and periodic:
        hw_fine = fine_halfwidth_sec * sr
        pos_fine, r_fine = _envelope_search_local(
            ev,
            templates,
            n_rms,
            max(pos_a - hw_fine, pos_lo),
            min(pos_a + hw_fine, pos_hi),
            np.arange(n_envelope_frac_fine) / n_envelope_frac_fine,
        )
        if np.isfinite(pos_fine) and r_fine >= r_env:
            pos_a, r_env = pos_fine, r_fine

    # ---- stage B: carrier NCC within the cycle-ambiguity half-width of stage A
    pos_b, r_car = _local_carrier_search(
        v,
        templates,
        int(np.floor(pos_a - hw)),
        int(np.ceil(pos_a + hw)) + 1,
        n_carrier_frac,
    )
    if periodic or not np.isfinite(pos_b):
        pos_final = pos_a
    else:
        pos_final = pos_b
    return SoundAlignment(
        onset_time=recording_t0 + pos_final / sr,
        carrier_correlation=r_car,
        envelope_correlation=r_env,
        envelope_onset_time=recording_t0 + pos_a / sr,
        periodic=bool(periodic),
    )


def calibrate_playback_model(
    recordings: Sequence[npt.ArrayLike],
    recording_t0s: Sequence[float],
    recording_rate: float,
    waveforms: Sequence[npt.ArrayLike],
    nominal_rate: float,
    search_windows: Sequence[tuple[float, float]],
    reconstructions: Sequence[Reconstruction] = ("zoh", "bandlimited"),
    ppm_range: float = 400.0,
    ppm_step: float = 20.0,
    ppm_fine_step: float = 2.0,
) -> PlaybackModel:
    """Choose reconstruction type and effective playback rate that best explain
    the recordings (max median carrier correlation across trials).

    Candidate rates: a ppm grid around `nominal_rate`, plus the NI-DAQmx AO rate
    derived from an 80 MHz timebase; the best is then refined on a finer grid.
    """
    sr = float(recording_rate)
    recs = [np.asarray(r, dtype=np.float64) for r in recordings]
    # rough onsets, insensitive to small rate errors
    rough_model = PlaybackModel(nominal_rate, "bandlimited", nominal_rate)
    rough = [
        align_sound_waveform(
            r,
            t0,
            sr,
            w,
            rough_model,
            *win,
            n_envelope_frac=8,
            n_envelope_frac_fine=0,
            n_carrier_frac=1,
        ).envelope_onset_time
        for r, t0, w, win in zip(recs, recording_t0s, waveforms, search_windows)
    ]
    trials = [
        (r, t0, w, o)
        for r, t0, w, o in zip(recs, recording_t0s, waveforms, rough)
        if np.isfinite(o)
    ]
    if not trials:
        logger.warning(
            "Playback-model calibration failed: no usable trials; using nominal rate"
        )
        return PlaybackModel(ni_ao_sampling_rate(nominal_rate), "zoh", nominal_rate)

    def score(rate: float, recon: Reconstruction, n_frac: int) -> float:
        model = PlaybackModel(rate, recon, nominal_rate)
        rs = []
        for r, t0, w, o in trials:
            k0 = (o - t0) * sr
            _, rr = _local_carrier_search(
                r, _TemplateFactory(w, model, sr), int(k0) - 15, int(k0) + 16, n_frac
            )
            rs.append(rr if np.isfinite(rr) else 0.0)
        return float(np.median(rs))

    ppms = np.arange(-ppm_range, ppm_range + ppm_step / 2, ppm_step)
    rates = sorted(
        set(nominal_rate * (1 + ppms * 1e-6)) | {ni_ao_sampling_rate(nominal_rate)}
    )
    results = {
        (rate, recon): score(rate, recon, 4)
        for recon in reconstructions
        for rate in rates
    }
    (best_rate, best_recon), _ = max(results.items(), key=lambda kv: kv[1])
    fine_ppm = np.arange(-ppm_step, ppm_step + 1e-9, ppm_fine_step)
    fine_r = np.array(
        [score(best_rate * (1 + d * 1e-6), best_recon, 32) for d in fine_ppm]
    )
    # r is flat-topped in rate: take the vertex of a parabola fitted around the max
    i = int(np.argmax(fine_r))
    sel = slice(max(i - 3, 0), min(i + 4, len(fine_r)))
    a, b, _ = np.polyfit(fine_ppm[sel], fine_r[sel], 2)
    d_best = -b / (2 * a) if a < 0 else fine_ppm[i]
    d_best = float(np.clip(d_best, fine_ppm[0], fine_ppm[-1]))
    best_rate = best_rate * (1 + d_best * 1e-6)
    best_r = score(best_rate, best_recon, 32)
    model = PlaybackModel(best_rate, best_recon, nominal_rate, best_r)
    logger.info(
        f"Calibrated sound playback model: {best_recon} at {best_rate:.2f} Hz "
        f"({model.ppm:+.1f} ppm), median r={best_r:.4f} over {len(trials)} trials"
    )
    return model


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)
