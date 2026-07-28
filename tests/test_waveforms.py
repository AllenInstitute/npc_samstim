from __future__ import annotations

import types

import numpy as np
import pytest

import npc_samstim.waveforms as waveforms


class _FakeSync:
    def __init__(
        self: _FakeSync,
        *,
        rising_edges: tuple[float, ...],
        falling_edges: tuple[float, ...],
    ) -> None:
        self._rising_edges = np.asarray(rising_edges)
        self._falling_edges = np.asarray(falling_edges)

    def get_line_for_stim_onset(self: _FakeSync, waveform_type: str) -> str:
        return waveform_type

    def get_rising_edges(
        self: _FakeSync,
        line_index_or_label: int | str,
        units: str | None = None,
    ) -> np.ndarray:
        return self._rising_edges

    def get_falling_edges(
        self: _FakeSync,
        line_index_or_label: int | str,
        units: str | None = None,
    ) -> np.ndarray:
        return self._falling_edges


def _run_opto_sync_alignment(
    monkeypatch: pytest.MonkeyPatch,
    *,
    falling_edges: tuple[float, ...],
    rising_edges: tuple[float, ...] = (10.0005,),
    trigger_times: tuple[float, ...] = (10.0,),
) -> tuple[waveforms.StimRecording, ...]:
    stim_path = object()
    stim_data = {"trialOptoDur": np.full(len(trigger_times), 0.010)}
    sync = _FakeSync(
        rising_edges=rising_edges,
        falling_edges=falling_edges,
    )

    # This reproduces DynamicRoutingTask.getOptoPulseWaveform at 2 kHz for a
    # 10 ms pulse with 1 ms on/off ramps: the full command buffer is 12.5 ms.
    opto_waveform = waveforms.SimpleWaveform(
        name="square",
        modality="opto",
        sampling_rate=2000,
        samples=np.r_[0.0, np.ones(22), 0.0, 0.0],
    )

    monkeypatch.setattr(waveforms.npc_stim, "get_h5_stim_data", lambda _: stim_data)
    monkeypatch.setattr(waveforms.npc_sync, "get_sync_data", lambda _: sync)
    monkeypatch.setattr(
        waveforms.npc_stim,
        "get_stim_frame_times",
        lambda *args, **kwargs: {stim_path: np.asarray(trigger_times)},
    )
    monkeypatch.setattr(waveforms.npc_stim, "assert_stim_times", lambda times: times)
    monkeypatch.setattr(
        waveforms.npc_stim,
        "get_stim_trigger_frames",
        lambda *args, **kwargs: tuple(range(len(trigger_times))),
    )
    monkeypatch.setattr(
        waveforms,
        "get_waveforms_from_stim_file",
        lambda *args, **kwargs: (opto_waveform,) * len(trigger_times),
    )

    recordings = waveforms.get_stim_latencies_from_sync(
        stim_path,
        sync,
        waveform_type="opto",
    )
    assert all(recording is not None for recording in recordings)
    return tuple(recording for recording in recordings if recording is not None)


def test_opto_sync_alignment_uses_recorded_falling_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recording = _run_opto_sync_alignment(
        monkeypatch,
        falling_edges=(10.0115,),
    )[0]

    assert recording.onset_time_on_sync == pytest.approx(10.0005)
    assert recording.offset_time_on_sync == pytest.approx(10.0115)


def test_opto_sync_alignment_without_falling_edge_uses_nominal_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recording = _run_opto_sync_alignment(monkeypatch, falling_edges=())[0]

    assert recording.onset_time_on_sync == pytest.approx(10.0005)
    assert recording.offset_time_on_sync == pytest.approx(10.0105)


def test_opto_sync_alignment_does_not_borrow_next_trial_falling_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recordings = _run_opto_sync_alignment(
        monkeypatch,
        rising_edges=(10.0005, 20.0005),
        falling_edges=(20.0115,),
        trigger_times=(10.0, 20.0),
    )

    assert recordings[0].offset_time_on_sync == pytest.approx(10.0105)
    assert recordings[1].offset_time_on_sync == pytest.approx(20.0115)


def test_opto_sync_alignment_uses_final_fall_after_internal_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recording = _run_opto_sync_alignment(
        monkeypatch,
        falling_edges=(10.0030, 10.0115),
    )[0]

    assert recording.offset_time_on_sync == pytest.approx(10.0115)


def test_opto_nidaq_alignment_reports_active_interval_not_command_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_rate = 2000
    command_samples = np.r_[
        np.zeros(10),  # 5 ms programmed delay
        0.0,
        np.ones(22),
        0.0,
        0.0,
    ]
    opto_waveform = waveforms.SimpleWaveform(
        name="square",
        modality="opto",
        sampling_rate=sample_rate,
        samples=command_samples,
    )
    presentation = waveforms.StimPresentation(
        trial_idx=0,
        waveform=opto_waveform,
        trigger_time_on_sync=0.300,
    )

    # The complete command buffer starts 10 ms after the script trigger.
    nidaq_data = np.zeros((1000, 1), dtype=np.int16)
    command_start_sample = round(0.310 * sample_rate)
    nidaq_data[
        command_start_sample : command_start_sample + len(command_samples), 0
    ] = command_samples
    timing = types.SimpleNamespace(start_time=0.0, sampling_rate=sample_rate)
    monkeypatch.setattr(waveforms.tqdm, "tqdm", lambda iterable, **kwargs: iterable)

    recording = waveforms.xcorr(
        nidaq_data=nidaq_data,
        nidaq_timing=timing,
        nidaq_channel=0,
        presentations=(presentation,),
        padding_sec=0.050,
    )[0]

    assert recording is not None
    assert recording.onset_time_on_sync == pytest.approx(0.3155)
    assert recording.offset_time_on_sync == pytest.approx(0.3265)


def test_generate_opto_waveforms_includes_task_offset_voltage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stim_data = {
        "trialOptoDur": np.array([0.010, 0.010]),
        "trialOptoVoltage": np.array([2.0, 2.0]),
        "trialOptoOnsetFrame": np.array([1, 2]),
        "optoOnRamp": np.array(0.001),
        "optoOffRamp": np.array(0.001),
        "optoOffsetVoltage": np.array(0.1),
        "optoSampleRate": np.array(2000),
    }
    monkeypatch.setattr(waveforms.npc_stim, "get_h5_stim_data", lambda _: stim_data)
    monkeypatch.setattr(waveforms.npc_stim, "get_num_trials", lambda _: 2)

    generated = waveforms.generate_opto_waveforms(stim_data)[0]
    expected = waveforms.get_cached_opto_pulse_waveform(
        sampleRate=2000,
        amp=2.0,
        dur=0.010,
        delay=0.0,
        freq=0.0,
        onRamp=0.001,
        offRamp=0.001,
        offset=0.1,
    )

    assert generated is not None
    np.testing.assert_array_equal(generated.samples, expected)


def test_generate_opto_waveforms_selects_offset_for_each_trial_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stim_data = {
        "trialOptoDur": np.array([0.010, 0.010]),
        "trialOptoVoltage": np.array([2.0, 2.0]),
        "trialOptoOnsetFrame": np.array([1, 2]),
        "trialOptoDevice": np.array(["['laser_488']", "['laser_633']"]),
        "optoOnRamp": np.array(0.001),
        "optoOffRamp": np.array(0.001),
        "optoOffsetVoltage": {
            "laser_488": np.array(0.1),
            "laser_633": np.array(0.2),
        },
        "optoSampleRate": np.array(2000),
    }
    monkeypatch.setattr(waveforms.npc_stim, "get_h5_stim_data", lambda _: stim_data)
    monkeypatch.setattr(waveforms.npc_stim, "get_num_trials", lambda _: 2)

    generated = waveforms.generate_opto_waveforms(stim_data)

    assert generated[0] is not None
    assert generated[1] is not None
    assert generated[0].samples[0] == pytest.approx(0.2)
    assert generated[1].samples[0] == pytest.approx(0.4)
