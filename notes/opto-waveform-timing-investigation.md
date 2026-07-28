# Opto waveform timing investigation

Date: 2026-07-27

## Scope

Investigated optogenetic stimulus onset and offset timing across:

- `npc_samstim.waveforms.get_stim_latencies_from_sync`
- `npc_samstim.waveforms.get_stim_latencies_from_nidaq_recording`
- `DynamicRoutingTask.TaskUtils.getOptoPulseWaveform`
- `DynamicRoutingTask.OptoTagging`
- DynamicRoutingTask's opto hardware-marker construction

## Discoveries

### DynamicRoutingTask command-buffer construction

`getOptoPulseWaveform`:

- allocates samples for `dur + onRamp + offRamp`;
- adds one terminal sample;
- prepends programmed delay samples;
- creates each ramp with `int(ramp_duration * sample_rate)` samples.

At the standard 2 kHz opto sample rate, a 1 ms ramp has only two samples.
OptoTagging uses nominal durations of 10 or 200 ms and 1 ms onset and offset
ramps.

TaskControl generates the opto hardware marker by setting it high wherever the
analog opto command is greater than zero. Ramp quantization and offset voltage
therefore affect the marker's rising and falling edges.

Upstream sources:

- <https://github.com/samgale/DynamicRoutingTask/blob/master/TaskUtils.py#L63-L84>
- <https://github.com/samgale/DynamicRoutingTask/blob/master/OptoTagging.py#L21-L25>
- <https://github.com/samgale/DynamicRoutingTask/blob/master/TaskControl.py#L854-L875>

### Previous `npc_samstim` behavior

`get_stim_latencies_from_sync` used only the marker's rising edge. It calculated
the offset as:

```text
recorded onset + complete generated command-buffer duration
```

This treated delay, ramp padding, and the terminal sample as though they all
occurred after the already-recorded onset.

Deterministic reproductions at 2 kHz showed:

| Command | Previous offset error |
|---|---:|
| 10 ms OptoTagging pulse, 1 ms ramps, zero offset | +1.5 ms |
| 200 ms OptoTagging pulse, 1 ms ramps, zero offset | +1.5 ms |
| Pulse without ramps | +1.0 ms |
| 50 ms delay and 100 ms off-ramp | +51.0 ms |
| 10 ms OptoTagging pulse with positive offset | +0.5 ms |

The 50 ms delay was effectively counted twice.

### Waveform regeneration mismatch

DynamicRoutingTask and OptoTagging pass the laser's saved offset voltage to
`getOptoPulseWaveform`. `npc_samstim.generate_opto_waveforms` previously omitted
that argument. Consequently, reconstructed ramp samples and positive-output
boundaries could differ from the command sent to hardware.

This discrepancy matters when the dedicated sync marker is unavailable and
NI-DAQ cross-correlation uses the reconstructed waveform.

### Internal marker edges

A sinusoidal command can contain internal zero crossings, depending on its
offset voltage. Selecting the first falling edge after onset could truncate the
stimulus. The final pairing rule therefore uses the falling edge nearest the
reconstructed final active boundary, constrained to occur before the next
stimulus onset.

## Implemented behavior

### Dedicated opto marker available

- Use the recorded rising edge as onset.
- Reconstruct the expected positive-output interval from the saved waveform.
- Among falling edges after onset and before the next onset, use the edge
  nearest the expected final active boundary.
- Do not derive offset from the complete command-buffer length.

### Rising edge available but no matching falling edge

- Use recorded onset plus that trial's nominal `trialOptoDur`.
- Emit a warning identifying the trial and fallback duration.
- Do not borrow a falling edge belonging to the next trial.

### Dedicated marker unavailable, NI-DAQ recording available

- Cross-correlate the complete command buffer, as before.
- Treat the correlation lag as the command-buffer alignment.
- Report onset and offset from the positive-output interval within the aligned
  buffer.
- This prevents programmed delay and terminal padding from being reported as
  active opto time.

### No marker or NI-DAQ recording

The existing caller fallback remains unchanged: script/frame-derived onset plus
nominal `trialOptoDur`.

### Offset-voltage reconstruction

Generated opto waveforms now include:

- scalar offset voltage used by OptoTagging;
- a single saved device offset;
- per-trial offsets selected from multiple device entries using
  `trialOptoDevice`.

Missing offset metadata continues to mean zero, preserving compatibility with
older files.

## Assumptions and choices

1. **Active opto means positive analog command.**
   This matches TaskControl's hardware-marker rule. It describes commanded
   output, not independently measured optical emission.

2. **Recorded marker edges are authoritative when available.**
   Ramp quantization may make measured duration differ from nominal
   `trialOptoDur`; that difference reflects the command actually marked by
   TaskControl.

3. **Nominal duration is the conservative missing-offset fallback.**
   If no corresponding falling edge was recorded, actual marker duration
   cannot be recovered reliably.

4. **A completely missing marker still raises `MissingSyncLineError`.**
   This preserves the existing flow that attempts NI-DAQ alignment and then
   script timing.

5. **Positive-output boundaries use sample-hold semantics.**
   Onset is the first positive sample time. Offset is the boundary after the
   final positive sample.

6. **The change does not modify DynamicRoutingTask hardware generation.**
   The two-sample 1 ms ramp remains in upstream task code. A genuine smoother
   hardware ramp would require changing the waveform construction or sample
   rate there.

## Verification

- Added deterministic regression tests for:
  - recorded falling-edge offsets;
  - missing falling-edge nominal fallback;
  - preventing use of the next trial's edge;
  - internal zero crossings;
  - NI-DAQ alignment with programmed delay and ramp padding;
  - scalar OptoTagging offset voltage;
  - per-trial, per-device offset voltage.
- All eight local repository tests passed.
- Full source type checking passed.
- Test lint and diff checks passed.
- No temporary debug instrumentation remains.
- After S3 credentials were refreshed, all ten S3-backed doctests in
  `src/npc_samstim/waveforms.py` passed in 2 minutes 56 seconds.

### Real OptoTagging validation

Validated both OptoTagging epochs from session
`662892_2023-08-21_12-43-45`:

- `OptoTagging_662892_20230821_125915.hdf5`
- `OptoTagging_662892_20230821_143937.hdf5`
- sync: `20230821T124345.h5`

The files contain:

- 300 trials each;
- nominal durations of 10 and 200 ms;
- 1 ms onset and offset ramps;
- a saved offset voltage of `0.17085009376826207 V`;
- no explicit sample-rate dataset, so the task default of 2 kHz applies.

Regenerated waveforms have:

| Nominal duration | Full command buffer | Positive-output interval |
|---:|---:|---:|
| 10 ms | 12.5 ms | 12 ms |
| 200 ms | 202.5 ms | 202 ms |

Results across all 600 trials:

- every trial produced one recording;
- all onsets were strictly increasing;
- recorded durations were 12 or 202 ms, with some values 0.01 ms longer due
  to sync-clock quantization;
- the previous `recorded onset + full buffer duration` calculation placed
  every offset 0.49 or 0.50 ms after the recorded falling edge;
- the repaired calculation uses the recorded falling edge and therefore
  removes that systematic error.

## Validation limitation

The DynamicRouting task in the validated session had no opto trials, so it
could not provide a real opto NI-DAQ fallback example. That path is covered by
the deterministic delayed/ramped command-buffer regression test, while the
repository's existing S3-backed NI-DAQ doctests also passed. A real
DynamicRouting session containing opto trials but lacking the dedicated opto
marker would provide the strongest remaining integration check.

## Files changed

- `src/npc_samstim/waveforms.py`
- `tests/test_waveforms.py`

The pre-existing modification to `.vscode/settings.json` was not changed.
