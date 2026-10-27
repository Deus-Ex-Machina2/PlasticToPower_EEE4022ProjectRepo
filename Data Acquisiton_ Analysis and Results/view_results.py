"""
view_results.py - browse and plot saved cavity perturbation runs
================================================================

Pick one or more result JSON files in a file dialog; the program tabulates
them in the console and plots them.

    python view_results.py                  # opens the file picker
    python view_results.py a.json b.json    # or name files directly

Two kinds of figure, each overlaying every compatible file selected:

  Full insertion (known_sample / unknown_sample)
      |S11| in dB against frequency, zoomed on the resonances.
      Colour identifies the file, line style the cavity state
      (empty / quartz tube / sample).

  Partial insertion
      The regression the permittivity actually comes from: x against the
      effective volume fraction v, and y/2 against v, each with its fitted
      line, its residuals, the single-point drift diagnostic, and a small
      panel of the per-depth S11 traces.

Raw S11 traces are read from the CSV files saved beside each JSON
(<base>_<label>.csv). A run whose CSVs have been moved still tabulates;
only its traces are missing.

Dependencies: numpy, matplotlib, tkinter (all already needed by the
acquisition side). CPM.py is NOT imported - this reads stored results and
re-plots them, it does not recompute anything.
"""

import csv
import glob
import math
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

# ============================================================
# SETTINGS
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

ZOOM_BANDWIDTHS = 6.0      # half-width of the zoomed axes, in bandwidths
R2_WARN = 0.98             # flag a partial-insertion fit below this
STATE_STYLE = {"empty": ":", "quartz": "--", "sample": "-"}


# ============================================================
# LOADING
# ============================================================

def pick_files(argv):
    """Return the list of JSON paths to open, from argv or a file dialog."""
    if len(argv) > 1:
        return [p for p in argv[1:] if p.lower().endswith(".json")]

    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        print("tkinter is not available - pass the JSON files on the command line.")
        return []

    root = tk.Tk()
    root.withdraw()
    paths = filedialog.askopenfilenames(
        title="Select one or more result JSON files",
        initialdir=RESULTS_DIR if os.path.isdir(RESULTS_DIR) else SCRIPT_DIR,
        filetypes=[("Result files", "*.json"), ("All files", "*.*")])
    root.destroy()
    return list(paths)


def classify(data):
    """
    Identify what kind of run a loaded JSON holds.

    The holder-series file carries no "workflow" key, so it is recognised
    by its shape instead; everything else is tagged.
    """
    workflow = data.get("workflow")
    if workflow in ("known_sample", "unknown_sample", "partial_insertion"):
        return workflow
    if "depths_mm" in data and "holder" in data:
        return "holder_series"
    if "quartz_tube" in data and "empty_cavity" in data:
        return "reference_file"
    return "unknown"


def read_trace_csv(path):
    """
    Read one saved sweep file.

    Columns are freq_Hz, S11_re_1, S11_im_1, S11_re_2, ... one pair per
    repeat sweep. Returns (freqs, [complex array per sweep]).
    """
    with open(path, newline="") as handle:
        rows = list(csv.reader(handle))
    header, body = rows[0], rows[1:]
    values = np.array([[float(c) for c in r] for r in body if r])
    freqs = values[:, 0]
    n_sweeps = (len(header) - 1) // 2
    traces = [values[:, 1 + 2 * k] + 1j * values[:, 2 + 2 * k]
              for k in range(n_sweeps)]
    return freqs, traces


def find_traces(json_path):
    """
    Locate the CSV sweep files saved beside a JSON run.

    Returns {label: (freqs, [traces])}, where the label is whatever follows
    the base name: "empty_linear", "sample_linear", "sample_12.00mm", ...
    """
    base = os.path.splitext(json_path)[0]
    out = {}
    for path in sorted(glob.glob(base + "_*.csv")):
        label = os.path.basename(path)[len(os.path.basename(base)) + 1:-4]
        try:
            out[label] = read_trace_csv(path)
        except (ValueError, IndexError, OSError) as err:
            print(f"  ! could not read {os.path.basename(path)}: {err}")
    return out


def load_run(path):
    """Load one JSON plus its companion traces into a single dict."""
    with open(path) as handle:
        data = json.load(handle)
    return {"path": path,
            "name": os.path.splitext(os.path.basename(path))[0],
            "data": data,
            "kind": classify(data),
            "traces": find_traces(path)}


# ============================================================
# SMALL HELPERS
# ============================================================

def db(s11):
    """|S11| in dB, guarding the log against exact zeros."""
    return 20.0 * np.log10(np.maximum(np.abs(s11), 1e-12))


def state_of(label):
    """Map a trace label onto a cavity state for line styling."""
    for state in ("empty", "quartz", "holder", "sample"):
        if label.startswith(state):
            return "quartz" if state == "holder" else state
    return "sample"


def iter_measurements(block):
    """
    Yield (sense, measurement dict) from a stored block.

    Full-insertion runs key their references by sense, e.g.
    {"linear": {...}}; partial-insertion runs store a single unkeyed
    measurement, {"f0_Hz": ...}. Both shapes appear in saved files.
    """
    if not block:
        return
    if "f0_Hz" in block:
        yield "-", block
    else:
        for sense, measurement in block.items():
            if isinstance(measurement, dict) and "f0_Hz" in measurement:
                yield sense, measurement


def fmt(value, spec="{:.5g}"):
    """Format a number, or a dash padded to the same column width."""
    if value is None:
        return "-".rjust(len(spec.format(0.0)))
    return spec.format(value)


def rule(width=96, char="-"):
    print(char * width)


def material_mean(data):
    """
    (eps', eps'', tan d) for any run, or (None, None, None).

    Full-insertion runs store per-sweep values under material_results;
    partial-insertion runs store a single fitted material_result.
    """
    if "material_result" in data:                   # partial insertion
        m = data["material_result"]
        return m.get("eps_real"), m.get("eps_imag"), m.get("tan_delta")

    results = data.get("material_results") or {}
    for table in results.values():                  # first sense present
        if "eps_r'" in table:
            return (table["eps_r'"]["mean"],
                    table.get("eps_r''", {}).get("mean"),
                    table.get("tan_delta", {}).get("mean"))
        if "mu_eff' (exact)" in table:              # magnetic mode
            return (table["mu_eff' (exact)"]["mean"],
                    table.get("mu_eff'' (exact)", {}).get("mean"), None)
    return None, None, None


def fit_fallback_count(data):
    """
    How many stored traces fell back to the half-power method.

    The comparison section records an "error" under one_port_fit whenever
    the complex fit refused a trace; extract_resonance falls back silently
    in that case, so a run can be labelled "fit" and not have used it.
    """
    comparison = data.get("resonance_method_comparison") or {}
    total = failed = 0
    for sweeps in comparison.values():
        for entry in sweeps:
            total += 1
            if "error" in entry.get("one_port_fit", {}):
                failed += 1
    return failed, total


# ============================================================
# CONSOLE TABLES
# ============================================================

def print_summary(runs):
    """One row per selected file."""
    print("\n" + "=" * 96)
    print(" SUMMARY")
    rule()
    head = (f"{'file':<34}{'type':<12}{'mode':<7}{'date':<12}"
            f"{'eps1':>9}{'eps2':>10}{'tan d':>10}")
    print(head)
    rule()
    for run in runs:
        data = run["data"]
        e1, e2, td = material_mean(data)
        cav = data.get("cavity") or {}
        kind = {"known_sample": "known", "unknown_sample": "unknown",
                "partial_insertion": "partial", "holder_series": "holder",
                "reference_file": "reference"}.get(run["kind"], run["kind"])
        print(f"{run['name'][:33]:<34}{kind:<12}{cav.get('mode','-'):<7}"
              f"{str(data.get('date',''))[:10]:<12}"
              f"{fmt(e1,'{:9.4f}')}{fmt(e2,'{:10.5f}')}{fmt(td,'{:10.5f}')}")
    rule()


def print_states(data):
    """Reference and measured resonances of a full-insertion run."""
    blocks = [("empty cavity", data.get("empty_cavity")),
              ("quartz tube", data.get("quartz_reference")),
              ("predicted", data.get("predicted")),
              ("measured", data.get("measured_average"))]
    print(f"  {'state':<16}{'sense':<9}{'f0 [GHz]':>12}{'BW [kHz]':>12}{'Q':>10}")
    for title, block in blocks:
        for sense, m in iter_measurements(block):
            print(f"  {title:<16}{sense:<9}{m['f0_Hz']/1e9:12.6f}"
                  f"{m['BW_Hz']/1e3:12.2f}{m['Q']:10.1f}")


def print_full_detail(run):
    """Per-sweep material values and the reference states."""
    data = run["data"]
    print(f"\n--- {run['name']}  ({run['kind']}) ---")
    cav = data.get("cavity") or {}
    print(f"  cavity: {cav.get('shape')} {cav.get('mode')}, "
          f"a = {cav.get('a_mm')} mm, h = {cav.get('h_mm')} mm")
    if "sample_volume_mm3" in data:
        print(f"  sample volume: {data['sample_volume_mm3']} mm^3")
    if "uses_holder" in data:
        print("  holder: " + ("quartz tube" if data["uses_holder"] else "none (bare)"))
    elif data.get("quartz_reference"):
        print("  holder: quartz tube  (file predates the two-reference change)")
    known = data.get("known_material")
    if known and known.get("eps") not in (None, "None"):
        print(f"  known eps = {known['eps']}, loss tangent = {known.get('loss_tangent')}")
    print_states(data)

    for sense, table in (data.get("material_results") or {}).items():
        print(f"\n  measured [{sense}]")
        for name, entry in table.items():
            values = ", ".join(f"{v:.5g}" for v in entry["values"])
            print(f"    {name:<20}{entry['mean']:>12.5g} +/- {entry['std']:<10.2g}"
                  f"  sweeps: {values}")

    # Accuracy against the known value, where there is one.
    e1, _, td = material_mean(data)
    if known and e1 is not None:
        try:
            eps_known = float(known["eps"])
            print(f"\n    eps' measured {e1:.4f} vs known {eps_known:.4f}"
                  f"   ({(e1 - eps_known) / eps_known * 100:+.2f} %)")
        except (TypeError, ValueError):
            pass


def print_partial_detail(run):
    """Per-depth table, fit parameters and the diagnostics that matter."""
    data = run["data"]
    pi = data.get("partial_insertion")
    if not pi:
        print(f"\n--- {run['name']}: no partial_insertion block ---")
        return

    print(f"\n--- {run['name']}  (partial insertion) ---")
    cav = data.get("cavity") or {}
    print(f"  cavity: {cav.get('shape')} {cav.get('mode')}, "
          f"a = {cav.get('a_mm')} mm, h = {cav.get('h_mm')} mm")
    print(f"  sample radius: {data.get('sample_radius_mm')} mm")
    print(f"  holder corrected: {data.get('holder_corrected')}"
          f"   (holder series: {data.get('holder_series_date') or 'none'})")
    print(f"  resonance method: {data.get('resonance_method')}"
          f"   unloaded Q0: {data.get('use_unloaded_q')}")

    depth = np.asarray(pi["depth"], float)
    v = np.asarray(pi["v_effective"], float)
    x = np.asarray(pi["x_freq_shift"], float)
    y = np.asarray(pi["y_inv_q_shift"], float)
    used = np.asarray(pi["used"], bool)
    sp1 = np.asarray(pi["single_point_eps_real"], float)
    height = float(cav.get("h_mm") or 1.0)

    res_x = x - (pi["slope_r"] * v - pi["beta_r"])
    res_y = y / 2.0 - (pi["slope_i"] * v - pi["beta_i"])

    print(f"\n  {'depth':>8}{'fill':>8}{'v':>12}{'x [ppm]':>11}{'y/2 [ppm]':>11}"
          f"{'res_x [ppm]':>13}{'eps1 pt':>10}  used")
    for i in range(len(depth)):
        print(f"  {depth[i]:8.2f}{depth[i]/height*100:7.1f}%{v[i]:12.4e}"
              f"{x[i]*1e6:11.1f}{y[i]/2*1e6:11.1f}{res_x[i]*1e6:13.2f}"
              f"{sp1[i]:10.4f}   {'*' if used[i] else ' '}")

    print(f"\n  window {pi['window_fill_fraction'][0]*100:.0f}-"
          f"{pi['window_fill_fraction'][1]*100:.0f} %   "
          f"{pi['n_points_used']} of {pi['n_points_total']} points used")
    print(f"  slope_r {pi['slope_r']:.6g} +/- {pi['slope_r_err']:.2g}    "
          f"beta_r {pi['beta_r']:.4g} +/- {pi['beta_r_err']:.2g}    "
          f"R2 {pi['r2_r']:.5f}")
    print(f"  slope_i {pi['slope_i']:.6g} +/- {pi['slope_i_err']:.2g}    "
          f"beta_i {pi['beta_i']:.4g} +/- {pi['beta_i_err']:.2g}    "
          f"R2 {pi['r2_i']:.5f}")
    print(f"  eps' = {pi['eps_real']:.5g} +/- {pi['eps_real_err']:.2g}    "
          f"eps'' = {pi['eps_imag']:.5g} +/- {pi['eps_imag_err']:.2g}    "
          f"tan d = {pi['tan_delta']:.5g}")

    for line in partial_warnings(run):
        print("  ! " + line)


def partial_warnings(run):
    """Checks a depth series should pass; returns a list of messages."""
    data = run["data"]
    pi = data["partial_insertion"]
    out = []

    x = np.asarray(pi["x_freq_shift"], float)
    depth = np.asarray(pi["depth"], float)
    order = np.argsort(depth)
    if np.any(np.diff(x[order]) <= 0):
        bad = [f"{depth[order][i+1]:.2f}" for i in range(len(x) - 1)
               if np.diff(x[order])[i] <= 0]
        out.append("x does not increase monotonically with depth - suspect "
                   "point(s) at " + ", ".join(bad) + " mm. Deeper insertion "
                   "must shift the resonance further down.")
    if pi["r2_r"] < R2_WARN:
        out.append(f"frequency fit R2 = {pi['r2_r']:.4f}, below {R2_WARN} - "
                   "the points are not on a straight line.")
    if abs(pi["beta_r"]) < 2 * pi["beta_r_err"]:
        out.append("beta_r is within 2 sigma of zero: the non-uniform-field "
                   "term is not resolved, so the correction may be unnecessary here.")
    if not data.get("holder_corrected"):
        out.append("no holder correction was applied - valid only if the sample "
                   "is self-supporting or the tube stays fully inserted.")

    failed, total = fit_fallback_count(data)
    if failed and data.get("resonance_method") == "fit":
        out.append(f"the one-port fit was selected but failed on {failed} of "
                   f"{total} traces, which silently fell back to the half-power "
                   "method - so Q0 and beta were not actually used.")

    drift = data.get("coupling_drift")
    if drift and drift.get("spread_percent", 0) > 2.0:
        out.append(f"coupling beta drifted {drift['spread_percent']:.1f} % across "
                   "the series; the coupling-unchanged assumption does not hold.")

    sp1 = np.asarray(pi["single_point_eps_real"], float)
    if len(sp1) > 1 and np.isfinite(sp1).all():
        swing = abs(sp1[-1] - sp1[0])
        if swing < 0.02 * abs(pi["eps_real"]):
            out.append("the single-point eps' barely drifts, so beta is small "
                       "and the conventional formula was already adequate.")
    return out


# ============================================================
# PLOTS
# ============================================================

def resonance_window(runs):
    """(f_lo, f_hi) covering every stored resonance, in Hz."""
    f0s, bws = [], []
    for run in runs:
        data = run["data"]
        blocks = [data.get("empty_cavity"), data.get("quartz_reference"),
                  data.get("measured_average")]
        blocks.append(data.get("measured_per_depth"))
        for block in blocks:
            for _, m in iter_measurements(block):
                f0s.append(m["f0_Hz"])
                bws.append(m["BW_Hz"])
    if not f0s:
        return None
    # Median, not max: one very lossy run has a bandwidth tens of times the
    # others and would otherwise stretch the window over the whole sweep.
    pad = float(np.clip(ZOOM_BANDWIDTHS * np.median(bws), 2e6, 80e6))
    return min(f0s) - pad, max(f0s) + pad


def plot_full_overlay(runs):
    """|S11| in dB for every full-insertion run, zoomed on the resonances."""
    fig, ax_zoom = plt.subplots(figsize=(11, 6))
    fig.canvas.manager.set_window_title("Full insertion - S11")

    colours = plt.cm.tab10(np.linspace(0, 1, 10))
    plotted = False

    for i, run in enumerate(runs):
        colour = colours[i % 10]
        for label, (freqs, traces) in sorted(run["traces"].items()):
            state = state_of(label)
            style = STATE_STYLE.get(state, "-")
            mean_db = db(np.mean(traces, axis=0))
            ax_zoom.plot(freqs / 1e9, mean_db, style, color=colour, lw=1.3,
                         label=f"{run['name'][:22]} · {label}")
            plotted = True

        # mark the measured resonance
        for _, m in iter_measurements(run["data"].get("measured_average")):
            f0_ghz = m["f0_Hz"] / 1e9
            ax_zoom.axvline(f0_ghz, color=colour, lw=0.8,
                            alpha=0.5, ls="-.")
            ax_zoom.annotate(
                f"{f0_ghz:.3f} GHz", xy=(f0_ghz, 0.98),
                xycoords=ax_zoom.get_xaxis_transform(),
                xytext=(3, 0), textcoords="offset points",
                rotation=90, ha="left", va="top", fontsize=8, color=colour,
                bbox={"facecolor": "white", "edgecolor": "none",
                      "alpha": 0.75, "pad": 1.0})

    if not plotted:
        ax_zoom.text(0.5, 0.5, "No S11 trace files found beside these runs",
                     ha="center", va="center", transform=ax_zoom.transAxes)

    ax_zoom.set_xlabel("frequency [GHz]")
    ax_zoom.set_ylabel(r"$|S_{11}|$ [dB]")
    ax_zoom.set_title("Zoomed on the resonances", fontsize=10)
    ax_zoom.grid(True, alpha=0.3)

    window = resonance_window(runs)
    if window:
        ax_zoom.set_xlim(window[0] / 1e9, window[1] / 1e9)

    handles, labels = ax_zoom.get_legend_handles_labels()
    bottom = 0.12
    if handles:
        ncol = 3
        rows = math.ceil(len(labels) / ncol)
        bottom = 0.11 + 0.035 * rows
        fig.legend(handles, labels, loc="lower center", ncol=ncol,
                   fontsize=8, frameon=False)
    fig.suptitle("Full-insertion runs — mean of the repeat sweeps "
                 "(dotted empty · dashed tube · solid sample)", fontsize=11)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.91,
                        bottom=bottom)
    return fig


def plot_partial_overlay(runs):
    """
    The regression that produces the permittivity, overlaid across runs.

    Left column: x and y/2 against the effective volume fraction, each with
    its fitted line and a residual strip. Right column: the single-point
    drift diagnostic, and a small panel of the per-depth S11 traces.
    """
    fig = plt.figure(figsize=(13, 9))
    fig.canvas.manager.set_window_title("Partial insertion - regression")
    # Two columns of two groups. Each left-hand group is a tall plot with a
    # tight residual strip under it, so the gap between groups stays clear.
    outer = gridspec.GridSpec(2, 2, width_ratios=[1.35, 1], figure=fig,
                              hspace=0.42, wspace=0.28,
                              left=0.08, right=0.97, top=0.86, bottom=0.12)
    top_left = outer[0, 0].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.06)
    bot_left = outer[1, 0].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.06)
    ax_x = fig.add_subplot(top_left[0])
    ax_xr = fig.add_subplot(top_left[1], sharex=ax_x)
    ax_y = fig.add_subplot(bot_left[0])
    ax_yr = fig.add_subplot(bot_left[1], sharex=ax_y)
    ax_sp = fig.add_subplot(outer[0, 1])
    ax_s11 = fig.add_subplot(outer[1, 1])

    colours = plt.cm.tab10(np.linspace(0, 1, 10))

    for i, run in enumerate(runs):
        data = run["data"]
        pi = data.get("partial_insertion")
        if not pi:
            continue
        colour = colours[i % 10]
        tag = run["name"][:20]

        v = np.asarray(pi["v_effective"], float)
        x = np.asarray(pi["x_freq_shift"], float)
        y2 = np.asarray(pi["y_inv_q_shift"], float) / 2.0
        used = np.asarray(pi["used"], bool)
        depth = np.asarray(pi["depth"], float)
        height = float((data.get("cavity") or {}).get("h_mm") or 1.0)

        line = np.linspace(0, v.max() * 1.08, 50)
        for ax, obs, slope, beta, res_label in (
                (ax_x, x, pi["slope_r"], pi["beta_r"], ax_xr),
                (ax_y, y2, pi["slope_i"], pi["beta_i"], ax_yr)):
            ax.plot(line * 1e3, (slope * line - beta) * 1e3, "-",
                    color=colour, lw=1.4, zorder=2)
            ax.plot(v[used] * 1e3, obs[used] * 1e3, "o", color=colour,
                    ms=6, zorder=3, label=f"{tag}")
            if (~used).any():
                ax.plot(v[~used] * 1e3, obs[~used] * 1e3, "o", mfc="none",
                        mec=colour, ms=6, zorder=3)
            # intercept -beta, where the line meets v = 0
            ax.plot(0, -beta * 1e3, "s", color=colour, ms=5, zorder=4)
            res_label.plot(v * 1e3, (obs - (slope * v - beta)) * 1e6, "o-",
                           color=colour, ms=4, lw=0.9)

        ax_sp.plot(depth / height * 100,
                   np.asarray(pi["single_point_eps_real"], float),
                   "o-", color=colour, ms=5, lw=1.1, label=tag)
        ax_sp.axhline(pi["eps_real"], color=colour, ls="--", lw=1.2, alpha=0.8)

        # per-depth S11, lighter the shallower the insertion
        depth_traces = sorted(
            ((lab, tr) for lab, tr in run["traces"].items()
             if lab.startswith("sample_")),
            key=lambda kv: _depth_of(kv[0]))
        for k, (label, (freqs, traces)) in enumerate(depth_traces):
            alpha = 0.3 + 0.7 * (k / max(1, len(depth_traces) - 1))
            ax_s11.plot(freqs / 1e9, db(np.mean(traces, axis=0)),
                        color=colour, lw=1.0, alpha=alpha)

    ax_x.set_ylabel(r"$x = (f_c-f_{cs})/f_c$   [$\times10^{-3}$]")
    ax_y.set_ylabel(r"$y/2$   [$\times10^{-3}$]")
    ax_xr.set_ylabel("resid\n[ppm]", fontsize=8)
    ax_yr.set_ylabel("resid\n[ppm]", fontsize=8)
    ax_yr.set_xlabel(r"effective volume fraction  $v = V'_s/V_{cav}$   [$\times10^{-3}$]")
    ax_x.set_title("Resonant-frequency shift vs effective volume fraction\n"
                   r"slope $\rightarrow\ \varepsilon'$; square: intercept $-\beta_r$",
                   fontsize=10, pad=8)
    ax_y.set_title("Inverse-Q shift vs effective volume fraction\n"
                   r"slope $\rightarrow\ \varepsilon''$; square: intercept $-\beta_i$",
                   fontsize=10, pad=8)

    ax_sp.set_xlabel("filling fraction [%]")
    ax_sp.set_ylabel(r"single-point $\varepsilon'$")
    ax_sp.set_title("Single-point permittivity vs insertion depth\n"
                    "dashed: permittivity from the regression",
                    fontsize=10, pad=8)

    ax_s11.set_xlabel("frequency [GHz]")
    ax_s11.set_ylabel(r"$|S_{11}|$ [dB]")
    ax_s11.set_title("Reflection spectra at each insertion depth\n"
                     "lighter traces: shallower insertion", fontsize=10, pad=8)
    window = resonance_window(runs)
    if window:
        ax_s11.set_xlim(window[0] / 1e9, window[1] / 1e9)

    for ax in (ax_x, ax_y, ax_xr, ax_yr, ax_sp, ax_s11):
        ax.grid(True, alpha=0.3)
    for ax in (ax_xr, ax_yr):
        ax.axhline(0, color="0.4", lw=0.8)
    for ax in (ax_x, ax_y):
        ax.axhline(0, color="0.6", lw=0.7)
        ax.set_xlim(left=0)
    plt.setp(ax_x.get_xticklabels(), visible=False)
    plt.setp(ax_y.get_xticklabels(), visible=False)

    handles, labels = ax_x.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)),
                   fontsize=9, frameon=False)

    modes = sorted({str((run["data"].get("cavity") or {}).get("mode"))
                    for run in runs if (run["data"].get("cavity") or {}).get("mode")})
    mode_label = f" ({', '.join(modes)})" if modes else ""
    fig.suptitle(f"Partial-insertion dielectric characterisation{mode_label}\n"
                 "Frequency and loss regressions, residuals and depth diagnostics "
                 "(open circles: excluded points)", fontsize=11)
    return fig


def _depth_of(label):
    """Numeric depth from a trace label such as 'sample_12.00mm'."""
    try:
        return float(label.rsplit("_", 1)[-1].replace("mm", ""))
    except ValueError:
        return 0.0


# ============================================================
# MAIN
# ============================================================

def main():
    paths = pick_files(sys.argv)
    if not paths:
        print("No files selected.")
        return

    runs = []
    for path in paths:
        try:
            runs.append(load_run(path))
        except (OSError, ValueError) as err:
            print(f"Could not read {path}: {err}")

    if not runs:
        return

    print_summary(runs)

    full = [r for r in runs if r["kind"] in ("known_sample", "unknown_sample")]
    partial = [r for r in runs if r["kind"] == "partial_insertion"]
    other = [r for r in runs if r not in full and r not in partial]

    for run in full:
        print_full_detail(run)
    for run in partial:
        print_partial_detail(run)
    for run in other:
        print(f"\n--- {run['name']}  ({run['kind']}) ---")
        print("  Not a sample run; nothing to plot. Keys: "
              + ", ".join(run["data"].keys()))

    figures = []
    if full:
        figures.append(plot_full_overlay(full))
    if partial:
        figures.append(plot_partial_overlay(partial))

    if figures:
        print(f"\nShowing {len(figures)} figure(s). Close them to exit.")
        plt.show()
    else:
        print("\nNothing to plot.")


if __name__ == "__main__":
    main()
