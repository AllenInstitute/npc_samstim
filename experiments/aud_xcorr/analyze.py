"""Figures + summary tables for the report, from results/final_trials.csv and the cache.

Outputs go to ../../outputs/aud-onset-xcorr/ (figures/*.png, tables/*.csv, numbers.json).
"""
from __future__ import annotations

import glob
import json
import pathlib
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import methods as M  # noqa: E402
from npc_samstim import sound_alignment as SA  # noqa: E402

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).parent
OUT = HERE.parent.parent / "outputs" / "aud-onset-xcorr"
FIG = OUT / "figures"
TAB = OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
NUM: dict = {}
MIN_R_ENV, MIN_R_CAR = 0.5, 0.3

d = pd.read_csv(HERE / "results" / "final_trials.csv")
cal = pd.read_csv(HERE / "results" / "final_calibration.csv")
d["accepted"] = (d.r_env >= MIN_R_ENV) & ((d.name == "tone") | (d.r_car >= MIN_R_CAR)) & d.final.notna()  # carrier gate: aperiodic only
d["kind"] = np.where(d.name == "tone", "tone", "AM noise")
d["group"] = d.stim.str.replace("DynamicRouting1", "DR task").str.replace("RFMapping", "RF mapping") + " / " + d.kind
v = d[d.has_sync & d.gt_on.notna()].copy()
o = d[~d.has_sync].copy()
NUM["n_validation_sessions"] = int(v.session.nunique())
NUM["n_validation_files"] = int(v.groupby(["session", "stim"]).ngroups)
NUM["n_validation_trials"] = int(len(v))
NUM["validation_date_range"] = [str(v.date.min()), str(v.date.max())]
NUM["n_old_sessions"] = int(o.session.nunique())
NUM["n_old_files"] = int(o.groupby(["session", "stim"]).ngroups)
NUM["n_old_trials"] = int(len(o))
NUM["old_date_range"] = [str(o.date.min()), str(o.date.max())]

# ---------------------------------------------------------------- LOSO offset
noise_ok = v[(v.kind == "AM noise") & v.accepted]
sess_med = noise_ok.assign(e=(noise_ok.final - noise_ok.gt_on) * 1e6).groupby("session").e.median()
NUM["offset_all_sessions_us"] = float(sess_med.median())
NUM["offset_session_median_range_us"] = [float(sess_med.min()), float(sess_med.max())]
loso = {s: -float(sess_med.drop(s).median()) * 1e-6 for s in sess_med.index}
v["offset"] = v.session.map(loso).fillna(-NUM["offset_all_sessions_us"] * 1e-6)
v["new"] = v.final + v.offset  # final pipeline: barcode clock + constant correction (LOSO)
v["new_linear_clock"] = v.final_linear_clock + v.offset
v.loc[~v.accepted, ["new", "new_linear_clock"]] = np.nan

METHODS = {
    "legacy": "legacy xcorr (current)",
    "legacy_env": "legacy xcorr, use_envelope=True",
    "legacy_barcode_clock": "legacy xcorr + barcode clock",
    "new_linear_clock": "new aligner, linear clock",
    "new": "new aligner + barcode clock (shipped)",
}
for m in METHODS:
    v["e_" + m] = (v[m] - v.gt_on) * 1e6


def summarize(e: pd.Series) -> dict:
    a = e.abs()
    n = len(e)
    a_ok = a.dropna()
    return dict(
        n=n, rejected=int(e.isna().sum()),
        median_us=float(e.median()), p50_abs_us=float(a_ok.median()), p95_abs_us=float(a_ok.quantile(0.95)),
        p99_abs_us=float(a_ok.quantile(0.99)), max_abs_us=float(a_ok.max()),
        frac_gt_100us=float((a_ok > 100).mean()), frac_gt_1ms=float((a_ok > 1000).mean()),
    )


rows = []
for g, gg in v.groupby("group"):
    for m, label in METHODS.items():
        rows.append(dict(group=g, method=label, **summarize(gg["e_" + m])))
summary = pd.DataFrame(rows)
summary.to_csv(TAB / "validation_summary.csv", index=False)

# excluding the one file with no speaker signal (reported separately)
bad_files = (
    v.groupby(["session", "stim"]).apply(lambda g: (~g.accepted).mean()).loc[lambda x: x > 0.5]
)
NUM["files_mostly_rejected"] = [f"{s} {st}: {x:.0%} rejected" for (s, st), x in bad_files.items()]
NUM["rejected_trials_validation"] = int((~v.accepted).sum())
NUM["rejected_trials_validation_outside_bad_files"] = int(
    (~v.accepted & ~v.set_index(["session", "stim"]).index.isin(bad_files.index)).sum()
)
# do rejected trials coincide with legacy gross errors? (i.e. is rejection justified)
rej = v[~v.accepted]
NUM["rejected_trials_legacy_gt1ms_frac"] = float(((rej.e_legacy.abs() > 1000)).mean()) if len(rej) else None
acc = v[v.accepted]
NUM["accepted_gross_errors_gt1ms"] = int((acc.e_new.abs() > 1000).sum())

# ---------------------------------------------------------------- tone by freq
tones = v[v.kind == "tone"].copy()
tones["khz"] = (tones.freq / 1000).round().astype(int)
tf = tones.groupby("khz").apply(
    lambda g: pd.Series(
        dict(
            n=len(g),
            legacy_p50=g.e_legacy.abs().median(), legacy_p95=g.e_legacy.abs().quantile(0.95),
            legacy_frac_gt1ms=(g.e_legacy.abs() > 1000).mean(),
            env_p95=g.e_legacy_env.abs().quantile(0.95),
            new_p50=g.e_new.abs().median(), new_p95=g.e_new.abs().quantile(0.95),
            new_max=g.e_new.abs().max(), new_frac_gt1ms=(g.e_new.abs() > 1000).mean(),
        )
    )
)
tf.to_csv(TAB / "tone_by_frequency.csv")

# ---------------------------------------------------------------- figures
# Fig 1: raw signals for one session (noise + tone), sync marker
meta, T = M.load(M.CACHE / "ecephys_668755_2023-08-31_12-33-31__RFMapping.npz")
tone_t = next(t for t in T if t.name == "tone" and abs(M.np.argmax(np.abs(np.fft.rfft(t.wf))) * t.wsr / len(t.wf) - 12000) < 60)
noise_t = next(t for t in T if t.name != "tone")
model = SA.PlaybackModel(47990.208, "zoh", 48000)
fig, axes = plt.subplots(2, 3, figsize=(11, 5), gridspec_kw=dict(width_ratios=[2, 1, 1]))
for row, t, label in ((0, noise_t, "AM noise"), (1, tone_t, "12 kHz tone")):
    st = (M._seg_times(t) - t.gt_on) * 1e3
    ax = axes[row, 0]
    ax.plot(st, t.seg, lw=0.3, color="C0")
    ax.axvline(0, color="r", lw=0.8, label="sync 'sound running' rising edge")
    ax.axvline((t.trigger - t.gt_on) * 1e3, color="k", ls=":", lw=0.8, label="trigger (vsync)")
    ax.set_xlim(-12, t.dur * 1e3 + 8)
    ax.set_title(f"{label}: NI-DAQ ch1 (speaker copy)")
    ax.set_xlabel("time re. sync marker (ms)")
    ax.set_ylabel("ADC counts")
    if row == 0:
        ax.legend(fontsize=7, loc="upper right")
    est = SA.align_sound_waveform(t.seg, t.seg_t0, t.sr, t.wf, model, t.trigger - 0.02, t.trigger + 0.25).onset_time
    for col, (lo, hi) in ((1, (-0.5, 2.0)), (2, (100, 100.6))):
        ax = axes[row, col]
        k = int(np.floor((est - t.seg_t0) * t.sr)) - 200
        frac = (est - (t.seg_t0 + k / t.sr)) * t.sr
        w = SA._TemplateFactory(t.wf, model, t.sr)(frac=frac, n=len(t.seg) - k)
        v_ = t.seg[k:].astype(float)
        n = min(len(w), len(v_))
        g_ = np.polyfit(w[:n], v_[:n], 1)
        tt = (t.seg_t0 + (k + np.arange(n)) / t.sr - est) * 1e3
        sel = (tt >= lo) & (tt <= hi)
        ax.plot(tt[sel], v_[:n][sel], ".-", ms=2, lw=0.5, label="recorded")
        ax.plot(tt[sel], np.polyval(g_, w[:n][sel]), ".-", ms=2, lw=0.5, label="ZOH model @ 47990 Hz")
        ax.set_title(f"zoom {lo}..{hi} ms")
        ax.set_xlabel("ms re. estimated onset")
        if row == 0 and col == 2:
            ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "fig1_signals.png", dpi=150)
plt.close(fig)

# Fig 2: why tones fail: legacy correlation vs new envelope correlation, one tone trial
t = tone_t
v_raw, nidaq_times = M._baseline_window(t)
wt, w_lin = M._interp_waveform(t)
import scipy.signal  # noqa: E402

c = scipy.signal.correlate(v_raw.astype(float), w_lin, mode="valid", method="fft")
lag_ms = (nidaq_times[: len(c)] + t.trigger - t.gt_on) * 1e3
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
axes[0].plot(lag_ms, c / np.abs(c).max(), lw=0.4)
axes[0].axvline(0, color="r", lw=0.8)
axes[0].axvline(lag_ms[np.argmax(c)], color="k", ls="--", lw=0.8)
axes[0].set_title(f"legacy: raw xcorr (linear-interp template)\nargmax error = {lag_ms[np.argmax(c)]:.2f} ms")
axes[0].set_xlabel("candidate onset re. sync marker (ms)")
# new: envelope NCC curve
templates = SA._TemplateFactory(t.wf, model, t.sr)
n_rms = 30
ev = SA._moving_rms(t.seg.astype(float), n_rms)
pad = 2 * n_rms
ncc = SA._SlidingNCC(ev, templates.n + 2 * pad)
best = None
for f in np.arange(16) / 16:
    ew = SA._moving_rms(np.r_[np.zeros(pad), templates(f), np.zeros(pad)], n_rms)
    r = ncc(ew)
    on_ms = (t.seg_t0 + (np.arange(len(r)) + pad + f) / t.sr - t.gt_on) * 1e3
    if best is None or r.max() > best[0].max():
        best = (r, on_ms)
r, on_ms = best
axes[1].plot(on_ms, r, lw=0.6)
axes[1].axvline(0, color="r", lw=0.8)
axes[1].set_xlim(lag_ms.min(), lag_ms.max())
axes[1].set_title("new, stage A: envelope NCC\n(DAC model, best sub-sample phase)")
axes[1].set_xlabel("candidate onset re. sync marker (ms)")
# carrier comb near 0
taus = np.arange(-0.6e-3, 0.6e-3, 2e-6)
rs = []
for tau in taus:
    on = t.gt_on + tau
    k = int(np.floor((on - t.seg_t0) * t.sr))
    fr = (on - (t.seg_t0 + k / t.sr)) * t.sr
    w = templates(fr)
    seg = t.seg[k : k + len(w)].astype(float)
    rs.append(np.corrcoef(seg, w)[0, 1])
axes[2].plot(taus * 1e6, rs, lw=0.6)
axes[2].axvline(0, color="r", lw=0.8)
axes[2].set_title("carrier NCC near the true onset:\npeaks every cycle, nearly equal")
axes[2].set_xlabel("candidate onset re. sync marker (us)")
fig.tight_layout()
fig.savefig(FIG / "fig2_tone_failure.png", dpi=150)
plt.close(fig)

# Fig 3: DAC rate evidence from marker durations
recs = []
for f in sorted(glob.glob(str(M.CACHE / "*.npz"))):
    if f.endswith("__barcodes.npz"):
        continue
    z = np.load(f)
    meta = json.loads(str(z["meta"]))
    if not meta["has_sync_line"]:
        continue
    dur = z["gt_off"] - z["gt_on"]
    ok = np.isfinite(dur)
    recs.append(dict(file=pathlib.Path(f).name, fs=float(np.median((z["wf_len"][ok] - 1) / dur[ok]))))
rate = pd.DataFrame(recs)
NUM["marker_rate_hz_median"] = float(rate.fs.median())
NUM["marker_rate_hz_range"] = [float(rate.fs.min()), float(rate.fs.max())]
fig, ax = plt.subplots(1, 2, figsize=(9, 3))
ppm = lambda f: (f / 48000 - 1) * 1e6  # noqa: E731
ax[0].hist(ppm(rate.fs), bins=np.arange(-203.5, -199.0, 0.1))
ax[0].axvline(ppm(80e6 / 1667), color="r", label="80 MHz / 1667 = 47990.40 Hz")
ax[0].set_xlabel("(N-1) / marker high-time, ppm re. 48 kHz (per stim file)")
ax[0].set_ylabel("stim files")
ax[0].legend(fontsize=7)
ax[0].set_title("effective AO sample rate from sync marker")
cc = cal.copy()
for i, (mode, g) in enumerate(cc.groupby("mode")):
    for recon, gg in g.groupby("recon"):
        ax[1].scatter(gg.ppm, gg.cal_r, s=8, marker="o" if recon == "zoh" else "^", label=f"{mode} / {recon}", color=f"C{i}" if recon == "zoh" else f"C{i+3}")
ax[1].axvline((80e6 / 1667 / 48000 - 1) * 1e6, color="r", lw=0.6)
ax[1].set_ylim(0.6, 1.01)
ax[1].set_xlim(-230, 40)
ax[1].annotate("1 file with no speaker signal (r=0.02, +124 ppm) off-axis", (-200, 0.62), fontsize=6)
ax[1].set_xlabel("calibrated playback rate (ppm re. 48 kHz)")
ax[1].set_ylabel("median carrier r")
ax[1].set_title("per-file playback calibration")
ax[1].legend(fontsize=6, loc="center")
fig.tight_layout()
fig.savefig(FIG / "fig3_playback_rate.png", dpi=150)
plt.close(fig)
calsum = cal.groupby(["mode", "recon"]).agg(
    files=("ppm", "size"), ppm_median=("ppm", "median"), ppm_min=("ppm", "min"), ppm_max=("ppm", "max"),
    r_median=("cal_r", "median"), r_min=("cal_r", "min"),
)
calsum.to_csv(TAB / "calibration_summary.csv")

# Fig 4: error distributions by method (validation, accepted for new)
groups = sorted(v.group.unique())
fig, axes = plt.subplots(1, len(groups), figsize=(11, 3.3), sharey=True)
bins = np.logspace(0, 5.5, 60)
for ax, g in zip(axes, groups):
    gg = v[v.group == g]
    for m, col in (("legacy", "C3"), ("legacy_env", "C1"), ("new", "C0")):
        a = gg["e_" + m].abs().dropna().clip(lower=1)
        ax.hist(a, bins=bins, histtype="step", color=col, label=METHODS[m], density=False)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(1000, color="k", ls=":", lw=0.6)
    ax.set_title(g)
    ax.set_xlabel("|onset error| (us)")
axes[0].set_ylabel("trials")
axes[0].legend(fontsize=6, loc="upper right")
fig.tight_layout()
fig.savefig(FIG / "fig4_error_hist.png", dpi=150)
plt.close(fig)

# Fig 5: tone errors by frequency
fig, ax = plt.subplots(figsize=(8, 3.2))
x = tf.index.values
ax.semilogy(x, tf.legacy_p95, "o-", color="C3", label="legacy p95")
ax.semilogy(x, tf.env_p95, "o-", color="C1", label="legacy use_envelope p95")
ax.semilogy(x, tf.new_p95, "o-", color="C0", label="new p95")
ax.semilogy(x, tf.new_p50, "o--", color="C0", label="new median")
ax.semilogy(x, 1e6 / (x * 1000), "k:", label="one carrier period")
ax.set_xlabel("tone frequency (kHz)")
ax.set_ylabel("|onset error| (us)")
ax.legend(fontsize=7, ncol=2)
ax.set_title("tone onset error vs. frequency (RF mapping, validation sessions)")
fig.tight_layout()
fig.savefig(FIG / "fig5_tone_freq.png", dpi=150)
plt.close(fig)

# Fig 6: clock wander: error vs time for 4 sessions, linear vs barcode clock (noise only)
ex = ["676909_2023-12-12", "681532_2023-10-17", "667252_2023-09-27", "668755_2023-08-31"]
fig, axes = plt.subplots(1, 4, figsize=(11, 3), sharey=True)
for ax, s in zip(axes, ex):
    gg = v[v.session.str.contains(s) & (v.kind == "AM noise")]
    t0 = gg.trigger.min()
    ax.plot((gg.trigger - t0) / 60, gg.e_new_linear_clock, ".", ms=1.5, color="C3", label="linear clock (npc_ephys)")
    ax.plot((gg.trigger - t0) / 60, gg.e_new, ".", ms=1.5, color="C0", label="barcode-smoothed clock")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(s, fontsize=8)
    ax.set_xlabel("session time (min)")
axes[0].set_ylabel("onset error (us), AM noise")
axes[0].legend(fontsize=6, markerscale=5)
fig.tight_layout()
fig.savefig(FIG / "fig6_clock.png", dpi=150)
plt.close(fig)

# Fig 7: per-session median error by method (AM noise)
fig, ax = plt.subplots(figsize=(9, 3.6))
sess = sorted(v.session.unique())
for m, col in (("legacy", "C3"), ("new_linear_clock", "C1"), ("new", "C0")):
    med = [v[(v.session == s) & (v.kind == "AM noise")]["e_" + m].median() for s in sess]
    ax.plot(range(len(sess)), med, "o", ms=3, color=col, label=METHODS[m])
ax.axhline(0, color="k", lw=0.5)
ax.set_ylim(-120, 120)
ax.annotate("legacy, 686176 2023-12-06 (no speaker signal in DR task): -4.7 ms, off-axis", (0, -112), fontsize=6)
ax.set_xticks(range(len(sess)))
ax.set_xticklabels([s[8:25] for s in sess], rotation=90, fontsize=5)
ax.set_ylabel("median error (us), AM noise")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "fig7_session_bias.png", dpi=150)
plt.close(fig)

# Fig 8: old sessions: legacy vs new disagreement + latency distributions
o["dleg"] = (o.legacy - o.final) * 1e6
o_ok = o[o.accepted]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
for (mode, kind), gg in o_ok.groupby(["mode", "kind"]):
    a = gg.dleg.abs().clip(lower=1)
    axes[0].hist(a, bins=np.logspace(0, 5.5, 50), histtype="step", label=f"{mode} / {kind} (n={len(gg)})")
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].axvline(1000, color="k", ls=":", lw=0.6)
axes[0].set_xlabel("|legacy - new| (us)")
axes[0].set_title("sessions without sync sound line")
axes[0].legend(fontsize=6)
for mode, gg in o_ok.groupby("mode"):
    axes[1].hist((gg.final - gg.trigger) * 1e3, bins=np.arange(0, 120, 1), histtype="step", label=mode)
axes[1].set_xlabel("sound latency re. trigger vsync (ms), new method")
axes[1].set_title("latency by sound mode")
axes[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "fig8_old_sessions.png", dpi=150)
plt.close(fig)
old_tab = o_ok.groupby(["mode", "kind"]).dleg.apply(
    lambda x: pd.Series(dict(n=len(x), p50_abs_us=x.abs().median(), p95_abs_us=x.abs().quantile(0.95), frac_gt_1ms=(x.abs() > 1000).mean()))
).unstack()
old_tab.to_csv(TAB / "old_sessions_legacy_vs_new.csv")
NUM["old_rejected"] = int((~o.accepted).sum())
NUM["old_rejected_by_file"] = {f"{a} {b}": int(c) for (a, b), c in o[~o.accepted].groupby(["session", "stim"]).size().items()}
NUM["old_latency_ms_by_mode"] = {
    m: [float(x) for x in np.percentile((g.final - g.trigger) * 1e3, [1, 50, 99])] for m, g in o_ok.groupby("mode")
}
NUM["affected_old_files_with_tones"] = int(o.groupby(["session", "stim"]).apply(lambda g: (g.kind == "tone").any()).sum())

# runtime
NUM["sec_per_trial_by_kind"] = {k: float(g.sec.median()) for k, g in d.groupby("kind")}
NUM["calibration_sec_median"] = float(cal.sec_calibration.median())

(OUT / "numbers.json").write_text(json.dumps(NUM, indent=1, default=str))
pd.set_option("display.width", 200)
print(summary.round(2).to_string())
print(tf.round(1).to_string())
print(calsum.round(2).to_string())
print(old_tab.round(3).to_string())
print(json.dumps(NUM, indent=1, default=str))
