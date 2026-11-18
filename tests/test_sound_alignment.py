from __future__ import annotations

import types

import numpy as np
import pytest

import npc_samstim.waveforms as waveforms
from npc_samstim import sound_alignment as sa

NIDAQ_RATE = 30029.95  # typical PXI NI-DAQ rate on the sync clock
DAQ_MODEL = sa.PlaybackModel(sa.ni_ao_sampling_rate(48000), "zoh", 48000)


def _noise(n: int = 4800, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n) * np.hanning(n)


def _tone(freq: float, n: int = 4800, ramp: int = 240) -> np.ndarray:
    wf = np.sin(2 * np.pi * freq * np.arange(n) / 48000)
    window = np.hanning(2 * ramp + 1)
    wf[:ramp] *= window[:ramp]
    wf[-ramp:] *= window[ramp + 1 :]
    return wf


def _record(
    wf: np.ndarray,
    onset: float,
    model: sa.PlaybackModel = DAQ_MODEL,
    duration: float = 0.25,
    noise_std: float = 0.0,
    seed: int = 1,
) -> np.ndarray:
    rec = sa._TemplateFactory(wf, model, NIDAQ_RATE)(
        frac=onset * NIDAQ_RATE, n=int(duration * NIDAQ_RATE)
    )
    if noise_std:
        rec = rec + np.random.default_rng(seed).normal(0, noise_std, rec.shape)
    return rec


@pytest.mark.parametrize("onset", [0.0123456, 0.0300101, 0.05])
def test_noise_onset_is_recovered_to_microseconds(onset: float) -> None:
    wf = _noise()
    rec = _record(wf, onset, noise_std=0.01)
    result = sa.align_sound_waveform(rec, 0.0, NIDAQ_RATE, wf, DAQ_MODEL, 0.0, 0.08)
    assert result.onset_time == pytest.approx(onset, abs=2e-6)
    assert result.envelope_correlation > 0.9


@pytest.mark.parametrize("freq", [4000, 6000, 9000, 12000, 15000, 16000])
def test_tone_onset_has_no_gross_error(freq: float) -> None:
    """Raw-waveform correlation used to fail by up to ~100 ms for tones (most of
    all at frequencies near rational fractions of the NI-DAQ rate)."""
    wf = _tone(freq)
    onset = 0.0213579
    rec = _record(wf, onset, noise_std=0.01)
    result = sa.align_sound_waveform(rec, 0.0, NIDAQ_RATE, wf, DAQ_MODEL, 0.0, 0.08)
    assert abs(result.onset_time - onset) < 1 / freq  # within one carrier cycle


def test_legacy_xcorr_fails_where_new_method_succeeds() -> None:
    """Documents the original failure mode on an aliased tone."""
    wf = _tone(15000, n=12000)  # 250 ms tone
    onset = 0.3076543
    rec = _record(wf, onset, duration=1.0)
    presentation = waveforms.StimPresentation(
        trial_idx=0,
        waveform=waveforms.SimpleWaveform("tone", "sound", 48000, wf),
        trigger_time_on_sync=0.3,
    )
    timing = types.SimpleNamespace(start_time=0.0, sampling_rate=NIDAQ_RATE)
    data = np.round(rec * 1e4).astype(np.int16)[:, None]
    legacy = waveforms.xcorr(data, timing, 0, (presentation,), legacy=True)[0]
    new = waveforms.xcorr(data, timing, 0, (presentation,), onset_correction_sec=0.0)[0]
    assert legacy is not None and new is not None
    assert abs(new.onset_time_on_sync - onset) < 1 / 15000
    assert abs(legacy.onset_time_on_sync - onset) > abs(new.onset_time_on_sync - onset)


@pytest.mark.parametrize(
    "model",
    [
        DAQ_MODEL,
        sa.PlaybackModel(48000 * (1 + 10e-6), "bandlimited", 48000),
    ],
)
def test_calibration_recovers_playback_model(model: sa.PlaybackModel) -> None:
    onsets = [0.0213579, 0.031, 0.0277]
    waveforms_ = [
        _noise(24000, seed=s) for s in range(3)
    ]  # 0.5 s, as in DynamicRouting1
    recordings = [
        _record(wf, onset, model=model, duration=0.6, noise_std=0.005, seed=i)
        for i, (wf, onset) in enumerate(zip(waveforms_, onsets))
    ]
    calibrated = sa.calibrate_playback_model(
        recordings=recordings,
        recording_t0s=[0.0] * 3,
        recording_rate=NIDAQ_RATE,
        waveforms=waveforms_,
        nominal_rate=48000,
        search_windows=[(0.0, 0.08)] * 3,
    )
    assert calibrated.reconstruction == model.reconstruction
    assert calibrated.sampling_rate == pytest.approx(model.sampling_rate, rel=3e-6)


def test_sample_clock_from_barcodes_follows_nonlinear_drift() -> None:
    samples = np.arange(0, 3600 * 30000, 30 * 30000, dtype=float)  # barcode every 30 s
    drift = 80e-6 * np.sin(samples / samples[-1] * np.pi)  # 80 us bow over an hour
    times = 5.0 + samples / 30000.0 + drift
    rng = np.random.default_rng(0)
    times_quantized = times + rng.uniform(
        -17e-6, 17e-6, times.shape
    )  # 30 kHz stamp jitter
    clock = sa.SampleClock.from_barcodes(samples, times_quantized)
    probe = np.linspace(samples[2], samples[-3], 50)
    truth = 5.0 + probe / 30000.0 + 80e-6 * np.sin(probe / samples[-1] * np.pi)
    assert np.max(np.abs(clock.sample_to_time(probe) - truth)) < 10e-6
    np.testing.assert_allclose(clock.time_to_sample(truth), probe, atol=0.5)
    linear = np.polyval(np.polyfit(samples, times_quantized, 1), probe)
    assert np.max(np.abs(linear - truth)) > 30e-6  # what the smoothed clock corrects


def test_align_sound_marks_missing_or_unrelated_signal_as_nan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(waveforms.tqdm, "tqdm", lambda iterable, **kwargs: iterable)
    wf = _noise(4800)
    presentations = tuple(
        waveforms.StimPresentation(
            trial_idx=i,
            waveform=waveforms.SimpleWaveform("AM_noise", "sound", 48000, wf),
            trigger_time_on_sync=t,
        )
        for i, t in enumerate((1.0, 2.0, 3.0))
    )
    n = int(4 * NIDAQ_RATE)
    rec = np.random.default_rng(5).normal(0, 0.01, n)
    # trial 0: real sound; trial 1: unrelated noise burst; trial 2: nothing
    k0 = int(1.008 * NIDAQ_RATE)
    rec[k0 : k0 + 3100] += _record(
        wf, 1.008 - k0 / NIDAQ_RATE, duration=3100 / NIDAQ_RATE
    )
    k1 = int(2.008 * NIDAQ_RATE)
    rec[k1 : k1 + 3000] += np.random.default_rng(9).normal(0, 0.3, 3000)
    data = np.round(rec * 1e4).astype(np.int16)[:, None]
    timing = types.SimpleNamespace(start_time=0.0, sampling_rate=NIDAQ_RATE)
    recordings = waveforms.xcorr(
        data,
        timing,
        0,
        presentations,
        playback_model=DAQ_MODEL,
        onset_correction_sec=0.0,
    )
    assert recordings[0] is not None and not isinstance(
        recordings[0], waveforms.NullRecording
    )
    assert recordings[0].onset_time_on_sync == pytest.approx(1.008, abs=5e-6)
    assert isinstance(recordings[1], waveforms.NullRecording)
    assert isinstance(recordings[2], waveforms.NullRecording)
    assert np.isnan(recordings[1].latency) and np.isnan(recordings[2].latency)
