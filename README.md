# npc_samstim

Tools for processing stim data from Sam's DynamicRoutingTask, compatible with data in the cloud.

[![PyPI](https://img.shields.io/pypi/v/npc_samstim.svg?label=PyPI&color=blue)](https://pypi.org/project/npc_samstim/)
[![Python version](https://img.shields.io/pypi/pyversions/npc_samstim)](https://pypi.org/project/npc_samstim/)

[![Coverage](https://img.shields.io/codecov/c/github/AllenInstitute/npc_samstim?logo=codecov)](https://app.codecov.io/github/AllenInstitute/npc_samstim)
[![CI/CD](https://img.shields.io/github/actions/workflow/status/AllenInstitute/npc_samstim/publish.yml?label=CI/CD&logo=github)](https://github.com/AllenInstitute/npc_samstim/actions/workflows/publish.yml)
[![GitHub issues](https://img.shields.io/github/issues/AllenInstitute/npc_samstim?logo=github)](https://github.com/AllenInstitute/npc_samstim/issues)

# Usage
```bash
conda create -n npc_samstim python>=3.9
conda activate npc_samstim
pip install npc_samstim
```

## Python
```python
>>> import npc_samstim
```

## Sound onset times without a sync line
Sessions recorded before 2023-08-31 have no "sound running" line on sync, so sound
onsets are found in the NI-DAQ recording of the speaker signal with
`get_stim_latencies_from_nidaq_recording(..., waveform_type='sound')`.

The default method (`waveforms.align_sound`, core in `sound_alignment.py`):
- models playback (NI-DAQ AO zero-order hold at 80 MHz/1667 = 47990.4 Hz, or a sound card's
  band-limited output) and calibrates the effective rate per stim file,
- finds each onset with a robust envelope correlation, refined by carrier correlation for
  noise,
- maps NI-DAQ samples to sync time through all matched barcodes (not just the first and last),
- returns NaN latency when the recording doesn't match the waveform.

Validated against the sync line in 33 sessions (21,069 trials):
- AM-noise onsets: p95 error ~10 us
- tone onsets: p95 ~0.12 ms, and at most 0.29 ms for any single frequency

The previous raw cross-correlation (`xcorr_legacy`, still used for opto, or via `legacy=True`)
was more than 1 ms wrong for 85% of tone trials, by up to ~100 ms.
See `outputs/aud-onset-xcorr/report.pdf` and `experiments/aud_xcorr/`.

# Development
See instructions in https://github.com/AllenInstitute/npc_samstim/CONTRIBUTING.md and the original template: https://github.com/AllenInstitute/copier-pdm-npc/blob/main/README.md