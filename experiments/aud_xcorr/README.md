# Audio-onset alignment experiment

Validation of NI-DAQ-based sound onset detection against the sync "sound running"
line, and development of `npc_samstim.sound_alignment`. The write-up is in
`outputs/aud-onset-xcorr/report.pdf`, with its source in `report.md`.

Run the scripts from this directory with the repo's venv. `_creds.py` exports AWS
credentials resolved by boto3, because aiobotocore can't spawn a `credential_process`
on Windows.

| Step | Script | Output |
|---|---|---|
| list sessions on S3 | `discover.py` | `sessions.json` |
| cache NI-DAQ ch1 segments, waveforms, sync ground truth | `extract.py` | `cache/*.npz` (~6.4 GB, not committed) |
| cache barcode pairs (sync vs NI-DAQ) | `fetch_barcodes.py` | `cache/*__barcodes.npz` |
| check that the offline legacy replica equals the library | `check_baseline.py` | `results_check_baseline.txt` |
| legacy methods + preliminary new method | `evaluate.py` | `results/trials.csv` |
| clock-mapping variants | `eval_clock.py`, `clockmap.py` | stdout |
| tone sub-sample-phase grid density | `exp_frac.py` | stdout |
| shipped pipeline on all files | `evaluate_final.py` | `results/final_trials.csv`, `results/final_calibration.csv` |
| figures and tables | `analyze.py` | `../../outputs/aud-onset-xcorr/` |
| public API on S3 vs sync | `e2e_check.py` | `results/e2e_trials_final.csv` |
| render the PDF (headless Edge) | `render_report.py` | `report.pdf` |

`methods.py` holds the offline replica of the legacy method (`baseline`, `baseline_env`)
and the prototype estimators (v2–v4) that led to the library implementation.
`explore_*.py`, `diag*.py`, `fit_analog.py` and `per_freq.py` are the exploratory steps
referred to in the report: signals, tone failure, spectra, analog-model fit, error
structure, and older sessions.
