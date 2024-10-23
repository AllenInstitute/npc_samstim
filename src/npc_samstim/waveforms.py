from __future__ import annotations

import concurrent.futures
import dataclasses
import datetime
import enum
import functools
import logging
from collections.abc import Iterable
from typing import Any, Callable, Literal, Protocol

import DynamicRoutingTask.TaskUtils
import npc_ephys
import npc_io
import npc_stim
import npc_sync
import numba
import numpy as np
import numpy.typing as npt
import scipy.signal
import tqdm
import zarr

logger = logging.getLogger(__name__)


class WaveformModality(enum.Enum):
    SOUND = enum.auto()
    OPTO = enum.auto()

    def __str__(self) -> str:
        return self.name.lower()

    @classmethod
    def from_factory(cls, s: Any) -> WaveformModality:
        if isinstance(s, WaveformModality):
            return s
        s = str(s)
        if any(
            label in s.lower()
            for label in ("sound", "audio", "tone", "noise", "acoustic")
        ):
            return cls.SOUND
        if any(label in s.lower() for label in ("opto", "optic")):
            return cls.OPTO
        raise ValueError(f"Could not determine modality from {s!r}")


class Waveform(Protocol):
    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def modality(self) -> WaveformModality:
        raise NotImplementedError

    @property
    def samples(self) -> npt.NDArray[np.float64]:
        raise NotImplementedError

    @property
    def sampling_rate(self) -> float:
        raise NotImplementedError

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sampling_rate

    @property
    def timestamps(self) -> npt.NDArray[np.float64]:
        return np.linspace(0.0, self.duration, len(self.samples), endpoint=False)

    def __eq__(self, other) -> bool:
        try:
            return (
                np.array_equal(self.samples, other.samples)
                and self.sampling_rate == other.sampling_rate
            )
        except (AttributeError, TypeError):
            return NotImplemented

    def __hash__(self) -> int:
        return hash((self.samples.tobytes(), self.sampling_rate))


class SimpleWaveform(Waveform):
    """
    >>> waveform = SimpleWaveform(name='test', modality='opto',sampling_rate=1, samples=np.array([1, 2, 3]))
    >>> waveform.duration
    3.0
    >>> waveform.timestamps
    array([0., 1., 2.])
    """

    def __init__(
        self,
        name: str,
        modality: str | WaveformModality,
        sampling_rate: float,
        samples: npt.NDArray[np.float64],
    ) -> None:
        self._samples = samples
        self._sampling_rate = sampling_rate
        self._name = name.replace(" ", "_")
        self._modality = WaveformModality.from_factory(modality)

    @property
    def name(self) -> str:
        return self._name

    @property
    def modality(self) -> WaveformModality:
        return self._modality

    @property
    def samples(self) -> npt.NDArray[np.float64]:
        return self._samples

    @property
    def sampling_rate(self) -> float:
        return self._sampling_rate


class LazyWaveform(Waveform):
    """Pass a function with args and kwargs used to generate the waveform
    on-demand, to avoid carrying around many not-so-small arrays in memory.

    If the function is wrapped with functools.cache or similar, then we
    waveforms available immediately and stored only once for each unique
    parameter set.

    >>> waveform = LazyWaveform(name='test', modality='opto', sampling_rate=1, fn=lambda dtype: np.array([1, 2, 3], dtype=dtype), dtype=np.float64)
    >>> waveform.samples
    array([1., 2., 3.])
    >>> waveform.duration
    3.0
    >>> waveform.timestamps
    array([0., 1., 2.])
    """

    def __init__(
        self,
        name: str,
        modality: str | WaveformModality,
        sampling_rate: float,
        fn: Callable[..., npt.NDArray[np.float64]],
        **kwargs,
    ) -> None:
        self._name = name.replace(" ", "_")
        self._modality = WaveformModality.from_factory(modality)
        self._sampling_rate = sampling_rate
        self._fn = fn
        self._kwargs = kwargs
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                # convert to tuple to make hashable (for caching)
                self._kwargs[k] = tuple(v)

    @property
    def name(self) -> str:
        return self._name

    @property
    def modality(self) -> WaveformModality:
        return self._modality

    @property
    def sampling_rate(self) -> float:
        return self._sampling_rate

    @property
    def samples(self) -> npt.NDArray[np.float64]:
        return self._fn(**self._kwargs)


@dataclasses.dataclass(frozen=True, eq=True)
class StimPresentation:
    """
    Info about a waveform-stimulus when it was triggered: its sample values (ideal, not actual), the time it was
    sent, the expected duration, etc.

    >>> presentation = StimPresentation(
    ...     trial_idx=0,
    ...     waveform=SimpleWaveform(name="test", modality="sound", sampling_rate=1, samples=np.array([1, 2, 3])),
    ...     trigger_time_on_sync=0,
    ...     )
    >>> presentation.duration
    3.0
    """

    trial_idx: int
    waveform: Waveform
    trigger_time_on_sync: float

    @property
    def duration(self) -> float:
        return self.waveform.duration


class StimRecording(Protocol):
    """Timing information about a waveform-stimulus as recorded."""

    @property
    def name(self) -> str:
        """Descriptive name - will be used as key in `nwb.stimuli` dict"""
        raise NotImplementedError

    @property
    def modality(self) -> WaveformModality:
        raise NotImplementedError

    @property
    def onset_time_on_sync(self) -> float:
        raise NotImplementedError

    @property
    def offset_time_on_sync(self) -> float:
        raise NotImplementedError

    @property
    def latency(self) -> float | None:
        raise NotImplementedError


class FlexStimRecording(StimRecording):
    """Information about an actual recording of a waveform-stimulus, mainly for
    obtaining onset and offset of the stimulus.

    >>> presentation = StimPresentation(
    ...     trial_idx=0,
    ...     waveform=SimpleWaveform(
    ...         name="test",
    ...         modality="sound",
    ...         sampling_rate=1,
    ...         samples=np.array([1, 2, 3]),
    ...     ),
    ...     trigger_time_on_sync=0,
    ... )
    >>> recording = FlexStimRecording(presentation=presentation, latency=0.1)
    >>> recording.onset_time_on_sync
    0.1
    >>> recording.offset_time_on_sync
    3.1

    >>> recorded_waveform = SimpleWaveform(
    ...     name="test",
    ...     modality="sound",
    ...     sampling_rate=1,
    ...     samples=np.array([1, 2, 3]),
    ... )
    >>> recording = FlexStimRecording(waveform=recorded_waveform, onset_time_on_sync=0.1)
    >>> recording.offset_time_on_sync
    3.1
    """

    def __init__(
        self,
        name: str | None = None,
        modality: str | WaveformModality | None = None,
        presentation: StimPresentation | None = None,
        waveform: Waveform | None = None,
        trigger_time_on_sync: float | None = None,
        latency: float | None = None,
        onset_time_on_sync: float | None = None,
        offset_time_on_sync: float | None = None,
    ) -> None:
        if not (name or presentation or waveform):
            raise ValueError(
                "At least one of `name`, `presentation`, `waveform` must be provided"
            )
        if not (presentation or waveform):
            raise ValueError(
                "At least one of `presentation`, `waveform` must be provided"
            )
        if not (presentation or waveform):
            raise ValueError(
                "At least one of `presentation` or `waveform` must be provided"
            )
        if latency is None and onset_time_on_sync is None:
            raise ValueError(
                "At least one of `latency` or `onset_time_on_sync` must be provided"
            )
        if not (presentation or waveform) and offset_time_on_sync is None:
            raise ValueError(
                "At least one of `presentation`, `waveform`, `offset_time_on_sync` must be provided"
            )
        # minimum attrs:
        self.presentation = presentation
        self.waveform = waveform
        self.trigger_time_on_sync = trigger_time_on_sync
        self._name = None if name is None else name.replace(" ", "_")
        self._modality = (
            None if modality is None else WaveformModality.from_factory(modality)
        )

        # attrs that can potentially be derived from other attrs:
        self._latency = latency
        self._onset_time_on_sync = onset_time_on_sync
        self._offset_time_on_sync = offset_time_on_sync

    @property
    def name(self) -> str:
        if self._name is not None:
            return self._name
        if self.waveform is not None:
            return self.waveform.name
        assert self.presentation is not None
        return self.presentation.waveform.name

    @property
    def modality(self) -> WaveformModality:
        if self._modality is not None:
            return self._modality
        if self.waveform is not None:
            return self.waveform.modality
        assert self.presentation is not None
        return self.presentation.waveform.modality

    @property
    def latency(self) -> float | None:
        if self._latency is not None:
            return self._latency
        assert self._onset_time_on_sync is not None
        if self.presentation is not None:
            return self.onset_time_on_sync - self.presentation.trigger_time_on_sync
        if self.trigger_time_on_sync is not None:
            return self.onset_time_on_sync - self.trigger_time_on_sync
        logger.warning("No trigger time available - cannot calculate latency")
        return None

    @property
    def onset_time_on_sync(self) -> float:
        if self._onset_time_on_sync is not None:
            return self._onset_time_on_sync
        assert self.latency is not None
        if self.presentation is not None:
            return np.nansum([self.presentation.trigger_time_on_sync, self.latency])
        assert self.trigger_time_on_sync is not None
        return np.nansum([self.trigger_time_on_sync, self.latency])

    @property
    def offset_time_on_sync(self) -> float:
        if self._offset_time_on_sync is not None:
            return self._offset_time_on_sync
        if self.waveform is not None:
            return self.onset_time_on_sync + self.duration
        assert self.presentation is not None
        return self.onset_time_on_sync + self.presentation.duration

    @property
    def duration(self) -> float:
        if self.waveform is not None:
            return self.waveform.duration
        return self.offset_time_on_sync - self.onset_time_on_sync


class NullRecording(FlexStimRecording):
    """A recording which didn't yield any useful information"""

    pass


def get_waveforms_from_stim_file(
    stim_file_or_dataset: npc_stim.StimPathOrDataset,
    waveform_type: Literal["sound", "audio", "opto"],
) -> tuple[Waveform | None, ...]:
    if any(s in waveform_type for s in ("sound", "audio")):
        return get_audio_waveforms_from_stim_file(stim_file_or_dataset)
    if "opto" in waveform_type:
        return get_opto_waveforms_from_stim_file(stim_file_or_dataset)
    raise ValueError(
        f"Unexpected value: {waveform_type = }. Should be 'sound' or 'opto'."
    )


def get_audio_waveforms_from_stim_file(
    stim_file_or_dataset: npc_stim.StimPathOrDataset,
) -> tuple[Waveform | None, ...]:
    """
    >>> path = 's3://aind-ephys-data/ecephys_662892_2023-08-21_12-43-45/behavior/RFMapping_662892_20230821_124434.hdf5'
    >>> waveforms = get_audio_waveforms_from_stim_file(path)
    >>> next(w for w in waveforms if w is not None).duration
    0.25
    """
    stim_data = npc_stim.get_h5_stim_data(stim_file_or_dataset)

    trialSoundArray: list[npt.NDArray] | None = stim_data.get("trialSoundArray", None)
    if (
        trialSoundArray is None
        or len(trialSoundArray) == 0
        or all(a.size == 0 for a in trialSoundArray)
    ):
        print("trialSoundArray empty; regenerating sound arrays")
        return generate_sound_waveforms(stim_data)

    # extract saved waveforms
    waveforms: list[Waveform | None] = [None] * npc_stim.get_num_trials(stim_data)
    for idx in range(len(waveforms)):
        if any(trialSoundArray[idx]):
            if sound_type := stim_data.get("trialSoundType"):
                name = sound_type[idx].decode()
            elif (noise := stim_data.get("trialAMNoiseFreq")) and ~np.isnan(noise[idx]):
                name = "AM_noise"
            elif (noise := stim_data.get("trialNoiseFreq")) and ~np.isnan(
                noise[idx]
            ).any():
                # older sessions use `trialNoiseFreq`, which contains two values
                name = "bandpass_filtered_noise"
            elif (
                tone := stim_data.get("trialToneFreq")
                or stim_data.get("trialSoundFreq")
            ) and ~np.isnan(tone[idx]):
                name = "tone"
            elif (sound_type := stim_data.get("soundType")) is not None:
                name = sound_type[()].decode()
            else:
                raise ValueError(
                    f"Could not determine sound type for trial {idx} in {stim_file_or_dataset}"
                )
            waveforms[idx] = SimpleWaveform(
                name=name,
                modality=WaveformModality.SOUND,
                sampling_rate=stim_data["soundSampleRate"][()],
                samples=trialSoundArray[idx],
            )
    return tuple(waveforms)


def get_opto_waveforms_from_stim_file(
    stim_file_or_dataset: npc_stim.StimPathOrDataset,
) -> tuple[Waveform | None, ...]:
    stim_data = npc_stim.get_h5_stim_data(stim_file_or_dataset)
    if "trialOptoDur" not in stim_data or len(stim_data["trialOptoDur"]) == 0:
        raise ValueError(
            f"trialOptoDur is empty - no opto waveforms to generate from {stim_file_or_dataset}"
        )
    return generate_opto_waveforms(stim_data)


@functools.wraps(DynamicRoutingTask.TaskUtils.makeSoundArray)
@functools.cache
def get_cached_sound_waveform(*args, **kwargs) -> npt.NDArray[np.float64]:
    # any unhashable args/kwargs (incl np.ndarray) will raise TypeError
    return DynamicRoutingTask.TaskUtils.makeSoundArray(*args, **kwargs)


@functools.wraps(DynamicRoutingTask.TaskUtils.getOptoPulseWaveform)
@functools.cache
def get_cached_opto_pulse_waveform(*args, **kwargs) -> npt.NDArray[np.float64]:
    # any unhashable args/kwargs (incl np.ndarray) will raise TypeError
    return DynamicRoutingTask.TaskUtils.getOptoPulseWaveform(*args, **kwargs)


def generate_sound_waveforms(
    stim_file_or_dataset: npc_stim.StimPathOrDataset,
) -> tuple[Waveform | None, ...]:
    """
    >>> path = 's3://aind-ephys-data/ecephys_668755_2023-08-31_12-33-31/behavior/DynamicRouting1_668755_20230831_131418.hdf5'
    >>> waveforms = generate_sound_waveforms(path)
    >>> next(w for w in waveforms if w is not None).duration
    0.5
    """
    stim_data = npc_stim.get_h5_stim_data(stim_file_or_dataset)

    nTrials = npc_stim.get_num_trials(stim_data)
    trialSoundDur = stim_data["trialSoundDur"][:nTrials]
    trialSoundFreq = stim_data["trialSoundFreq"][:nTrials]
    if "trialSoundSeed" in stim_data:
        trialSoundSeed = stim_data["trialSoundSeed"][:nTrials]
    else:
        logger.debug(
            "trialSoundSeed not found; likely older (2022) recording; setting to None"
        )
        trialSoundSeed = [None] * nTrials
    trialSoundType = stim_data["trialSoundType"][:nTrials]
    trialSoundVolume = stim_data["trialSoundVolume"][:nTrials]
    trialSoundAM = stim_data["trialSoundAM"][:nTrials]
    soundSampleRate = stim_data["soundSampleRate"][()]
    soundHanningDur = stim_data["soundHanningDur"][()]

    waveforms: list[Waveform | None] = [None] * nTrials
    for idx in range(len(waveforms)):
        if trialSoundType[idx].decode() == "":
            continue
        if trialSoundType[idx].decode() == "tone":
            # accounts for a quirk of how the trial sound frequencies are saved
            freq = trialSoundFreq[idx][0]
        else:
            freq = trialSoundFreq[idx]

        waveforms[idx] = LazyWaveform(
            name=trialSoundType[idx].decode(),
            modality=WaveformModality.SOUND,
            sampling_rate=soundSampleRate,
            fn=get_cached_sound_waveform,
            soundType=trialSoundType[idx].decode(),
            sampleRate=soundSampleRate,
            dur=trialSoundDur[idx],
            hanningDur=soundHanningDur,
            vol=trialSoundVolume[idx],
            freq=freq,
            AM=trialSoundAM[idx],
            seed=trialSoundSeed[idx],
        )

    return tuple(waveforms)


def generate_opto_waveforms(
    stim_file_or_dataset: npc_stim.StimPathOrDataset, device_index: int | None = None
) -> tuple[Waveform | None, ...]:
    """
    >>> path = 's3://aind-ephys-data/ecephys_662892_2023-08-21_12-43-45/behavior/OptoTagging_662892_20230821_125915.hdf5'
    >>> waveforms = generate_opto_waveforms(path)
    >>> next(w for w in waveforms if w is not None).duration
    0.2025
    """
    stim_data = npc_stim.get_h5_stim_data(stim_file_or_dataset)

    nTrials = npc_stim.get_num_trials(stim_data)

    trialOptoDur = stim_data["trialOptoDur"][:nTrials].squeeze()
    trialOptoVoltage = stim_data["trialOptoVoltage"][:nTrials].squeeze()

    # TODO update `trialOptoDelay` to accommodate multiple devices (task only)
    # Sam says: there is a trialOptoDelay value for each device (because the
    # analog output has to start synchronously for each laser but you might
    # want one laser to actually turn on later than the other one)
    if "trialOptoDelay" in stim_data:
        trialOptoDelay = stim_data["trialOptoDelay"][:nTrials]
    elif "optoDelay" in stim_data:
        trialOptoDelay = np.ones(nTrials) * stim_data["optoDelay"][()]
    else:
        trialOptoDelay = np.zeros(nTrials)

    if "trialOptoOffRamp" in stim_data:
        trialOptoOffRamp = stim_data["trialOptoOffRamp"][:nTrials]
    elif "optoOffRamp" in stim_data:
        trialOptoOffRamp = np.ones(nTrials) * stim_data["optoOffRamp"]
    else:
        trialOptoOffRamp = np.zeros(nTrials)

    if "trialOptoOnRamp" in stim_data:
        trialOptoOnRamp = stim_data["trialOptoOnRamp"][:nTrials]
    elif "optoOnRamp" in stim_data:
        trialOptoOnRamp = np.ones(nTrials) * stim_data["optoOnRamp"]
    else:
        trialOptoOnRamp = np.zeros(nTrials)

    if "trialOptoSinFreq" in stim_data:
        trialOptoSinFreq = stim_data["trialOptoSinFreq"][:nTrials]
    elif "optoSinFreq" in stim_data:
        trialOptoSinFreq = np.ones(nTrials) * stim_data["optoSinFreq"]
    else:
        trialOptoSinFreq = np.zeros(nTrials)

    if "optoSampleRate" in stim_data.keys():
        optoSampleRate = stim_data["optoSampleRate"][()]
    else:
        optoSampleRate = 2000

    if "trialOptoOnsetFrame" in stim_data.keys():
        trialOptoOnsetFrame = stim_data["trialOptoOnsetFrame"][:nTrials]
    else:
        trialOptoOnsetFrame = np.ones(nTrials) * stim_data["trialOptoOnsetFrame"]

    def device(array: npt.NDArray) -> npt.NDArray:
        if array.ndim > 1:
            return array[:, device_index or 0]
        return array

    waveforms: list[Waveform | None] = [None] * nTrials
    for trialnum in range(0, nTrials):
        if any(
            np.isnan(v[trialnum]) or v[trialnum] == 0
            for v in (trialOptoDur, trialOptoVoltage, trialOptoOnsetFrame)
        ):
            continue

        if trialOptoSinFreq[trialnum] != 0:
            name = "sine"
        else:
            name = "square"

        waveform = LazyWaveform(
            name=name,
            modality=WaveformModality.OPTO,
            sampling_rate=optoSampleRate,
            fn=get_cached_opto_pulse_waveform,
            sampleRate=optoSampleRate,
            amp=device(trialOptoVoltage)[trialnum],
            dur=device(trialOptoDur)[trialnum],
            delay=device(trialOptoDelay)[trialnum],
            freq=device(trialOptoSinFreq)[trialnum],
            onRamp=device(trialOptoOnRamp)[trialnum],
            offRamp=device(trialOptoOffRamp)[trialnum],
        )
        assert waveform is not None and waveform.samples.any()
        waveforms[trialnum] = waveform

    return tuple(waveforms)


def find_envelope(s, t, dmin=1, dmax=1):
    # locals min
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1

    # global min of dmin-chunks of locals min
    lmin = lmin[
        [i + np.argmin(s[lmin[i : i + dmin]]) for i in range(0, len(lmin), dmin)]
    ]
    # global max of dmax-chunks of locals max
    lmax = lmax[
        [i + np.argmax(s[lmax[i : i + dmax]]) for i in range(0, len(lmax), dmax)]
    ]

    # upsample envelope to original sampling rate
    s_min = np.interp(t, t[lmin], s[lmin])
    s_max = np.interp(t, t[lmax], s[lmax])

    return s_min, s_max


@numba.njit(parallel=True)
def _xcorr(v, w, t) -> tuple[float, float]:
    c = np.correlate(v, w)
    return t[np.argmax(c)], np.max(c)


def xcorr(
    nidaq_data: npt.NDArray[np.int16],
    nidaq_timing: npc_ephys.EphysTimingInfo,
    nidaq_channel: int,
    presentations: Iterable[StimPresentation | None],
    use_envelope: bool = False,
    padding_sec: float = 0.15,
    **kwargs,
) -> tuple[StimRecording | None, ...]:
    num_presentations = len(tuple(presentations))
    waveform_modality = next(
        p for p in presentations if p is not None
    ).waveform.modality
    recordings: list[StimRecording | None] = [None] * num_presentations
    padding_samples = int(padding_sec * nidaq_timing.sampling_rate)
    xcorr_values: list[float] = []

    for idx, presentation in tqdm.tqdm(
        iterable=enumerate(presentations),
        desc=f"aligning {waveform_modality.name.lower()} waveforms",
        unit="trial",
        total=num_presentations,
        ncols=80,
        ascii=False,
    ):
        if presentation is None:
            continue
        trigger_time_on_nidaq = (
            presentation.trigger_time_on_sync - nidaq_timing.start_time
        )
        onset_sample_on_nidaq = round(
            trigger_time_on_nidaq * nidaq_timing.sampling_rate
        )
        offset_sample_on_nidaq = round(
            (trigger_time_on_nidaq + presentation.duration) * nidaq_timing.sampling_rate
        )

        nidaq_times = (
            np.arange(
                (offset_sample_on_nidaq + padding_samples)
                - (onset_sample_on_nidaq - padding_samples)
            )
            / (nidaq_timing.sampling_rate)
            - padding_sec
        )
        nidaq_samples = nidaq_data[
            onset_sample_on_nidaq
            - padding_samples : offset_sample_on_nidaq
            + padding_samples,
            nidaq_channel,
        ]
        if not nidaq_samples.any():
            if ~np.isnan(xcorr_values).any():
                logger.warning(
                    f"Requested range {onset_sample_on_nidaq} to {offset_sample_on_nidaq} on {nidaq_channel=} is out of bounds: {nidaq_data.shape=}"
                )
            logger.warning(
                f"No sound recording for trial {idx} aud stim ({presentation.trigger_time_on_sync=} s) - setting latency=np.nan"
            )
            xcorr_values.append(np.nan)
            recordings[idx] = NullRecording(
                presentation=presentation,
                latency=np.nan,
            )
            continue

        interp_waveform_times = np.arange(
            0,
            presentation.duration,
            1 / nidaq_timing.sampling_rate,
        )
        interp_waveform_samples = np.interp(
            interp_waveform_times,
            presentation.waveform.timestamps,
            presentation.waveform.samples,
        )

        if use_envelope is False:
            lag, xcorr = _xcorr(nidaq_samples, interp_waveform_samples, nidaq_times)

        elif use_envelope is True:
            _, nidaq_samples_max = find_envelope(nidaq_samples, nidaq_times)
            _, interp_waveform_samples_max = find_envelope(
                interp_waveform_samples, interp_waveform_times
            )

            lag, xcorr = _xcorr(
                nidaq_samples_max, interp_waveform_samples_max, nidaq_times
            )

        # TODO: upsample option
        # interp_nidaq_times = np.arange(
        #     nidaq_times[0],
        #     nidaq_times[-1],
        #     1 / presentation.waveform.sampling_rate,
        # )
        # interp_nidaq_samples = np.interp(
        #     interp_nidaq_times,
        #     nidaq_times,
        #     nidaq_samples,
        # )

        # _,interp_nidaq_samples_max=find_envelope(interp_nidaq_samples,interp_nidaq_times)
        # _,waveform_samples_max=find_envelope(presentation.waveform.samples,presentation.waveform.timestamps)

        # lag, xcorr = _xcorr(interp_nidaq_samples_max, waveform_samples_max, interp_nidaq_times)

        recordings[idx] = FlexStimRecording(
            presentation=presentation,
            latency=lag,
        )

        xcorr_values.append(xcorr)
        # to verify:
        """
        import matplotlib.pyplot as plt
        norm_nidaq_samples = (nidaq_samples - np.mean(nidaq_samples)) / max(abs((nidaq_samples - np.mean(nidaq_samples))))
        norm_waveform_samples = (interp_waveform_samples - np.mean(interp_waveform_samples)) / max(abs((interp_waveform_samples - np.mean(interp_waveform_samples))))
        plt.plot(nidaq_times, norm_nidaq_samples)
        plt.plot(interp_waveform_times + recordings[-1].latency, norm_waveform_samples / max(abs(norm_waveform_samples)))
        plt.title(f"{recordings[-1].latency = }")
        """
    logger.info(
        f"Cross-correlation values: {np.nanmax(xcorr_values)=}, {np.nanmin(xcorr_values)=}, {np.nanmean(xcorr_values)=}"
    )
    return tuple(recordings)


def get_stim_latencies_from_nidaq_recording(
    stim_file_or_dataset: npc_stim.StimPathOrDataset,
    sync: npc_sync.SyncPathOrDataset,
    recording_dirs: Iterable[npc_io.PathLike],
    waveform_type: Literal["sound", "audio", "opto"],
    nidaq_device_name: str | None = None,
    correlation_method: Callable[
        [
            npt.NDArray[np.int16],
            npc_ephys.EphysTimingInfo,
            int,
            Iterable[StimPresentation | None],
            bool,
        ],
        tuple[StimRecording | None, ...],
    ] = xcorr,
    correlation_method_kwargs: dict[str, Any] | None = None,
    use_envelope: bool = False,
) -> tuple[StimRecording | None, ...]:
    """
    >>> stim = 's3://aind-ephys-data/ecephys_668755_2023-08-31_12-33-31/behavior/DynamicRouting1_668755_20230831_131418.hdf5'
    >>> sync = 's3://aind-ephys-data/ecephys_668755_2023-08-31_12-33-31/behavior/20230831T123331.h5'
    >>> recording_dirs = (
    ...     's3://aind-ephys-data/ecephys_668755_2023-08-31_12-33-31/ecephys_clipped/Record Node 102/experiment2/recording1',
    ...     's3://aind-ephys-data/ecephys_668755_2023-08-31_12-33-31/ecephys_clipped/Record Node 103/experiment2/recording1',
    ... )
    >>> recordings = get_stim_latencies_from_nidaq_recording(stim, sync, recording_dirs, waveform_type='sound') # doctest:+ELLIPSIS
    >>> latency = next(_ for _ in recordings if _ is not None).latency
    >>> assert 0 < latency < 0.1
    """
    sync = npc_sync.get_sync_data(sync)
    if not nidaq_device_name:
        nidaq_device = npc_ephys.get_pxi_nidaq_info(recording_dirs)
    else:
        nidaq_device = next(
            npc_ephys.get_ephys_timing_on_pxi(
                recording_dirs=recording_dirs, only_devices_including=nidaq_device_name
            )
        )

    nidaq_timing: npc_ephys.EphysTimingInfoOnSync = next(
        npc_ephys.get_ephys_timing_on_sync(
            sync=sync,
            recording_dirs=recording_dirs,
            devices=(nidaq_device,),
        )
    )

    nidaq_data = npc_ephys.get_pxi_nidaq_data(
        *recording_dirs,
        device_name=nidaq_device.device.name,
    )

    nidaq_channel = get_nidaq_channel_for_stim_onset(
        waveform_type, date=sync.start_time.date()
    )

    stim = npc_stim.get_h5_stim_data(stim_file_or_dataset)

    vsyncs = npc_stim.assert_stim_times(
        npc_stim.get_stim_frame_times(stim, sync=sync, frame_time_type="vsync")[stim]
    )

    num_trials = npc_stim.get_num_trials(stim)
    trigger_frames = npc_stim.get_stim_trigger_frames(stim)

    presentations: list[StimPresentation | None] = [None] * num_trials
    waveforms = get_waveforms_from_stim_file(stim, waveform_type)
    for idx, waveform in enumerate(waveforms):
        if waveform is None:
            continue
        # padding should be done by correlation method, when reading data
        presentations[idx] = StimPresentation(
            trial_idx=idx,
            waveform=waveform,
            trigger_time_on_sync=float(vsyncs[trigger_frames[idx]]),
        )

    # run the correlation of presentations with nidaq data
    recordings = correlation_method(
        nidaq_data,
        nidaq_timing,
        nidaq_channel,
        presentations,
        use_envelope,
        **(correlation_method_kwargs or {}),
    )

    return recordings


def get_waveforms_from_nidaq_recording(
    start_times_on_sync: Iterable[float],
    duration_sec: float,
    sync: npc_sync.SyncPathOrDataset,
    recording_dirs: Iterable[npc_io.PathLike],
    waveform_type: Literal["sound", "audio", "opto"],
    nidaq_device_name: str | None = None,
    resampling_factor: int | float | None = None,
) -> tuple[SimpleWaveform | None, ...]:
    """
    resulting length of samples will be original * resampling_factor, if not None

    >>> stim = 's3://aind-ephys-data/ecephys_668755_2023-08-31_12-33-31/behavior/DynamicRouting1_668755_20230831_131418.hdf5'
    >>> sync = 's3://aind-ephys-data/ecephys_668755_2023-08-31_12-33-31/behavior/20230831T123331.h5'
    >>> recording_dirs = (
    ...     's3://aind-ephys-data/ecephys_668755_2023-08-31_12-33-31/ecephys_clipped/Record Node 102/experiment2/recording1',
    ...     's3://aind-ephys-data/ecephys_668755_2023-08-31_12-33-31/ecephys_clipped/Record Node 103/experiment2/recording1',
    ... )
    >>> w = get_waveforms_from_nidaq_recording([100, 500], 1, sync, recording_dirs, 'sound', resampling_factor=.1) # doctest:+ELLIPSIS
    """
    start_times_on_sync = np.fromiter(start_times_on_sync, dtype=float)
    sync = npc_sync.get_sync_data(sync)
    if not nidaq_device_name:
        nidaq_device = npc_ephys.get_pxi_nidaq_info(recording_dirs)
    else:
        nidaq_device = next(
            npc_ephys.get_ephys_timing_on_pxi(
                recording_dirs=recording_dirs, only_devices_including=nidaq_device_name
            )
        )

    nidaq_timing: npc_ephys.EphysTimingInfoOnSync = next(
        npc_ephys.get_ephys_timing_on_sync(
            sync=sync,
            recording_dirs=recording_dirs,
            devices=(nidaq_device,),
        )
    )

    nidaq_data = npc_ephys.get_pxi_nidaq_data(
        *recording_dirs,
        device_name=nidaq_device.device.name,
    )
    nidaq_channel = get_nidaq_channel_for_stim_onset(
        waveform_type, date=sync.start_time.date()
    )
    # convert times on sync to times on nidaq
    nidaq_start_samples = np.around(
        (start_times_on_sync - nidaq_timing.start_time) * nidaq_timing.sampling_rate
    ).astype(int)
    nidaq_duration_samples = round(duration_sec * nidaq_timing.sampling_rate)

    def _get_waveform(
        start_sample: int,
        start_time: float,
    ) -> SimpleWaveform | None:
        nidaq_samples = nidaq_data[
            start_sample : start_sample + nidaq_duration_samples,
            nidaq_channel,
        ]
        if not nidaq_samples.any():
            logger.warning(
                f"Requested range {start_sample} to {start_sample + nidaq_duration_samples} on {nidaq_channel=} is out of bounds: {nidaq_data.shape=}"
            )
            return None
        else:
            if resampling_factor:
                nidaq_samples = scipy.signal.resample(
                    nidaq_samples,
                    int(len(nidaq_samples) * resampling_factor),
                )
                sampling_rate = nidaq_timing.sampling_rate * resampling_factor
            else:
                sampling_rate = nidaq_timing.sampling_rate
            return SimpleWaveform(
                name=f"{start_time:.3f} s",
                modality=WaveformModality.from_factory(waveform_type),
                sampling_rate=sampling_rate,
                samples=nidaq_samples,  # type: ignore[arg-type]
            )

    waveforms: list[SimpleWaveform | None]
    use_threading = isinstance(nidaq_data, zarr.Array)
    if use_threading:
        waveforms = [None] * len(nidaq_start_samples)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_idx = {}
            for idx, (start_sample, start_time) in enumerate(
                zip(nidaq_start_samples, start_times_on_sync)
            ):
                future = executor.submit(
                    _get_waveform, start_sample=start_sample, start_time=start_time
                )
                future_to_idx[future] = idx
            for future in tqdm.tqdm(
                iterable=concurrent.futures.as_completed(future_to_idx),
                desc=f"fetching data from {nidaq_device.device.name}",
                unit="segments",
                total=len(nidaq_start_samples),
                ncols=80,
                ascii=False,
            ):
                idx = future_to_idx[future]
                waveforms[idx] = future.result()
        if all(w is None for w in waveforms):
            logger.warning(
                f"No sound recording for any of the requested ranges {nidaq_start_samples} to {nidaq_start_samples + nidaq_duration_samples} on {nidaq_channel=} - setting latency=np.nan"
            )
    else:
        waveforms = []
        for start_sample, start_time in tqdm.tqdm(
            iterable=zip(nidaq_start_samples, start_times_on_sync),
            desc=f"fetching data from {nidaq_device.device.name}",
            unit="segments",
            total=len(nidaq_start_samples),
            ncols=80,
            ascii=False,
        ):
            waveforms.append(_get_waveform(start_sample, start_time))

    assert len(waveforms) == len(start_times_on_sync)
    return tuple(waveforms)


class MissingSyncLineError(IndexError):
    pass


def get_stim_latencies_from_sync(
    stim_file_or_dataset: npc_stim.StimPathOrDataset,
    sync: npc_sync.SyncPathOrDataset,
    waveform_type: Literal["sound", "audio", "opto"],
    line_index_or_label: int | str | None = None,
) -> tuple[StimRecording | None, ...]:
    """
    >>> stim = 's3://aind-ephys-data/ecephys_668755_2023-08-31_12-33-31/behavior/DynamicRouting1_668755_20230831_131418.hdf5'
    >>> sync = 's3://aind-ephys-data/ecephys_668755_2023-08-31_12-33-31/behavior/20230831T123331.h5'
    >>> latencies = get_stim_latencies_from_sync(stim, sync, waveform_type='sound')
    >>> assert 0 < next(_.latency for _ in latencies if _ is not None) < 0.1
    """
    stim = npc_stim.get_h5_stim_data(stim_file_or_dataset)
    sync = npc_sync.get_sync_data(sync)
    if line_index_or_label is None:
        line_index_or_label = sync.get_line_for_stim_onset(waveform_type)
    if not sync.get_rising_edges(line_index_or_label).any():
        raise MissingSyncLineError(
            f"No edges found for {line_index_or_label = } in {sync = }"
        )
    vsyncs = npc_stim.assert_stim_times(
        npc_stim.get_stim_frame_times(stim, sync=sync, frame_time_type="vsync")[stim]
    )
    trigger_times = tuple(
        vsyncs[idx] if idx is not None else None
        for idx in npc_stim.get_stim_trigger_frames(stim, stim_type=waveform_type)
    )
    stim_onsets = sync.get_rising_edges(line_index_or_label, units="seconds")
    recordings: list[StimRecording | None] = [None] * len(trigger_times)
    for idx, (trigger_time, waveform) in enumerate(
        zip(trigger_times, get_waveforms_from_stim_file(stim, waveform_type))
    ):
        if waveform is None:
            continue
        assert trigger_time
        onset_following_trigger = stim_onsets[
            np.searchsorted(stim_onsets, trigger_time, side="right")
        ]
        recordings[idx] = FlexStimRecording(
            presentation=StimPresentation(
                trial_idx=idx,
                waveform=waveform,
                trigger_time_on_sync=float(trigger_time),
            ),
            latency=onset_following_trigger - trigger_time,
        )
    return tuple(recordings)


def get_nidaq_channel_for_stim_onset(
    waveform_type: str | Literal["sound", "audio", "opto"],
    date: datetime.date | None = None,
) -> int:
    if any(label in waveform_type for label in ("aud", "sound")):
        return 1
    elif "opto" in waveform_type:
        return 5
    else:
        raise ValueError(f"Unexpected value: {waveform_type = }")


if __name__ == "__main__":
    from npc_samstim import testmod

    testmod()
