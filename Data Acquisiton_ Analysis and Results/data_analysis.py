"""
data_analysis.py - LibreVNA acquisition and cavity perturbation workflows
=========================================================================

Run from a terminal while LibreVNA-GUI is open:

    python data_analysis.py

Requirements
------------
* LibreVNA-GUI running, with its SCPI server enabled
  (Window -> Preferences -> General -> "SCPI Control", default port 19542).
* numpy.
* CPM.py in the same folder as this file.
* The LibreVNA repository cloned into ./LibreVNA (its small SCPI wrapper,
  libreVNA.py, is imported from the SCPI_Examples folder).

What the script does
--------------------
1. Connects to LibreVNA-GUI and checks that a VNA is attached.
2. Offers five workflows (see cavity_perturbation_planning_context.md
   and partial_insertion_plan.md):
     [1] Known sample   - empty cavity, empty quartz tube (reference),
                          prediction, sample measurement, comparison.
     [2] Unknown sample - uses the saved quartz tube reference.
     [3] Holder series  - the EMPTY quartz tube measured over a range of
                          insertion depths, saved to holder_series.json.
                          Needed once per cavity/tube, because the tube is
                          withdrawn with the sample and its perturbation
                          is therefore depth-dependent.
     [4] Partial insertion - the Orloff-corrected depth series. Measures
                          the sample at many depths and obtains the
                          permittivity from the SLOPE of the perturbation
                          against the effective volume fraction, which
                          projects out the non-uniform-field term. Use
                          this for high-loss samples that over-perturb
                          the cavity at full insertion.
     [5] Re-fit         - change the fit window on a saved run, offline.

   Workflows [1] and [2] are unchanged. [4] additionally writes a
   "resonance_method_comparison" section to its results file, holding all
   three resonance-extraction methods for every sweep, so they can be
   compared on real data.
3. Safety checks on every measurement:
     * the sweep has FINISHED before data is read (and all points arrived),
     * hardware status flags (PLL unlocked, ADC overload, unlevelled source),
     * the resonance is fully inside the sweep and the dip is deep enough,
     * N_REPEATS sweeps are compared: f0 and Q must agree within tolerance,
       otherwise the set is flagged (e.g. the cavity was bumped).
4. Pauses before every sweep so you can look at the LibreVNA-GUI yourself.

All lengths are entered in mm and volumes in mm^3 (only the ratio is used).
"""

import os
import sys
import json
import time
import datetime

import numpy as np

import CPM

# ------------------------------------------------------------
# Import the LibreVNA SCPI wrapper from the cloned repository
# ------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIBREVNA_DIR = os.path.join(SCRIPT_DIR, "LibreVNA", "Documentation",
                            "UserManual", "SCPI_Examples")
sys.path.insert(0, LIBREVNA_DIR)
from libreVNA import libreVNA  # noqa: E402  (import after sys.path change)


# ============================================================
# USER SETTINGS  (edit these as required)
# ============================================================
VNA_HOST = "localhost"
VNA_PORT = 19542
TRACE_NAME = "S11"            # trace read from the GUI

N_REPEATS = 3                 # sweeps per measurement set
F0_TOLERANCE_PPM = 50.0       # max spread (max-min)/mean of f0, in ppm
Q_TOLERANCE_FRAC = 0.05       # max spread (max-min)/mean of Q  (5 %)
MIN_DIP_DB = 3.0              # warn if the S11 dip is shallower than this
MIN_POINTS_IN_BW = 10         # warn if fewer sweep points fall inside the BW

SWEEP_TIMEOUT_S = 300.0       # give up waiting for a sweep after this
QUERY_TIMEOUT_S = 10.0        # socket timeout for normal queries

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
QUARTZ_FILE = os.path.join(RESULTS_DIR, "quartz_reference.json")
HOLDER_SERIES_FILE = os.path.join(RESULTS_DIR, "holder_series.json")

# Which resonance-extraction method feeds the results. Chosen at runtime by
# choose_resonance_method(); "half_power" keeps the original behaviour, so
# the two established workflows are unaffected unless the user changes it.
#   "half_power"      find_resonance, as specified in the handwritten notes
#   "half_power_norm" the same, divided by the measured off-resonance baseline
#   "fit"             fit_resonance, the one-port cavity model
RESONANCE_METHOD = "half_power"

# With "fit" the unloaded Q0 is available, so the perturbation can use
# D(1/Q0) directly instead of D(1/QL), which removes the assumption that
# the coupling is unchanged between reference and sample.
USE_UNLOADED_Q = True

DEFAULT_FIT_WINDOW = (0.30, 0.70)      # Orloff's default, as a fraction of fill


# ============================================================
# SMALL INPUT HELPERS
# ============================================================

def pause(message):
    """Wait for ENTER. Typing q quits the program."""
    answer = input("\n>>> " + message + "\n    Press ENTER to continue (q = quit): ")
    if answer.strip().lower() == "q":
        print("Quitting at user request.")
        sys.exit(0)


def ask_float(prompt, default=None):
    """Ask for a number. ENTER keeps the default (if one is given)."""
    while True:
        text = prompt if default is None else f"{prompt} [{default:g}]"
        answer = input(text + ": ").strip()
        if answer == "" and default is not None:
            return float(default)
        try:
            return float(answer)
        except ValueError:
            print("    Please enter a number (e.g. 2.45e9 or 12.5).")


def ask_yes_no(prompt, default=True):
    """Ask a y/n question."""
    hint = "[Y/n]" if default else "[y/N]"
    while True:
        answer = input(f"{prompt} {hint}: ").strip().lower()
        if answer == "":
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False


def ask_choice(prompt, options):
    """Show a numbered list and return the chosen option (string)."""
    print(prompt)
    for i, option in enumerate(options, start=1):
        print(f"   [{i}] {option}")
    while True:
        answer = input("Choice: ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1]
        print("    Please enter one of the numbers shown.")


# ============================================================
# VNA CONNECTION AND SETTINGS
# ============================================================

def connect_vna():
    """
    Connect to LibreVNA-GUI and make sure a VNA is attached.
    Exits the program with a clear message if anything is missing.
    """
    print(f"Connecting to LibreVNA-GUI at {VNA_HOST}:{VNA_PORT} ...")
    try:
        vna = libreVNA(VNA_HOST, VNA_PORT, timeout=QUERY_TIMEOUT_S)
    except Exception as err:
        print("ERROR:", err)
        print("Check that LibreVNA-GUI is open and the SCPI server is enabled.")
        sys.exit(1)

    print("GUI identifies as:", vna.query("*IDN?"))

    # Is a VNA device attached to the GUI?
    device = vna.query(":DEV:CONN?")
    if device == "Not connected":
        print("GUI is not connected to a VNA - trying to connect ...")
        vna.cmd(":DEV:CONN")
        device = vna.query(":DEV:CONN?")
    if device == "Not connected":
        print("ERROR: no LibreVNA device found. Check the USB cable/power.")
        sys.exit(1)
    print("Connected to device with serial:", device)

    vna.cmd(":DEV:MODE VNA")

    # Check the trace we need exists
    traces = vna.query(":VNA:TRACE:LIST?").split(",")
    if TRACE_NAME not in traces:
        print(f"ERROR: trace '{TRACE_NAME}' not found. Available: {traces}")
        sys.exit(1)

    # Report the calibration state (not enforced - user decides)
    print("Active calibration:", vna.query(":VNA:CAL:ACTIVE?"))
    return vna


def read_sweep_settings(vna):
    """Return the GUI's current sweep settings as a dictionary."""
    return {
        "start": float(vna.query(":VNA:FREQ:START?")),
        "stop": float(vna.query(":VNA:FREQ:STOP?")),
        "points": int(float(vna.query(":VNA:ACQ:POINTS?"))),
        "ifbw": float(vna.query(":VNA:ACQ:IFBW?")),
        "averages": int(float(vna.query(":VNA:ACQ:AVG?"))),
    }


def print_sweep_settings(settings):
    print(f"    start  = {settings['start'] / 1e9:.6f} GHz")
    print(f"    stop   = {settings['stop'] / 1e9:.6f} GHz")
    print(f"    points = {settings['points']}")
    if "ifbw" in settings:
        print(f"    IFBW   = {settings['ifbw']:g} Hz,  averages = {settings['averages']}")


def apply_sweep_settings(vna, start, stop, points):
    """Send start/stop/points to the VNA and read them back."""
    vna.cmd(":VNA:SWEEP FREQUENCY")
    vna.cmd(f":VNA:FREQ:START {start:.0f}")
    vna.cmd(f":VNA:FREQ:STOP {stop:.0f}")
    vna.cmd(f":VNA:ACQ:POINTS {int(points)}")
    print("Sweep settings now on the VNA:")
    print_sweep_settings(read_sweep_settings(vna))


def choose_sweep_settings(vna, suggestion=None):
    """
    Let the user keep the GUI settings, accept a suggestion, or type
    new values. Any calibration should be done in the GUI beforehand.
    """
    print("\nCurrent sweep settings in the GUI:")
    print_sweep_settings(read_sweep_settings(vna))

    options = ["Keep the current GUI settings"]
    if suggestion is not None:
        print("Suggested sweep settings:")
        print_sweep_settings(suggestion)
        options.append("Use the suggested settings")
    options.append("Enter settings manually")

    choice = ask_choice("Sweep settings:", options)
    if choice == "Use the suggested settings":
        apply_sweep_settings(vna, suggestion["start"], suggestion["stop"], suggestion["points"])
    elif choice == "Enter settings manually":
        start = ask_float("Start frequency (GHz)") * 1e9
        stop = ask_float("Stop frequency (GHz)") * 1e9
        points = int(ask_float("Number of points", 401))
        apply_sweep_settings(vna, start, stop, points)


def suggest_sweep(predicted):
    """
    Suggest a sweep around a predicted resonance:
    span = 10 x predicted BW (so both 3 dB points and some baseline are
    visible), 401 points (about 40 points inside the BW).
    """
    span = 10.0 * predicted.BW
    return {"start": predicted.f_res - span / 2.0,
            "stop": predicted.f_res + span / 2.0,
            "points": 401}


def set_live_view(vna):
    """Return the GUI to continuous sweeping so you can watch it live."""
    vna.cmd(":VNA:ACQ:SINGLE FALSE")
    vna.cmd(":VNA:ACQ:RUN")


# ============================================================
# SWEEP ACQUISITION AND CHECKS
# ============================================================

def check_hardware_status(vna):
    """Return a list of warnings from the VNA's hardware status flags."""
    warnings = []
    checks = {":DEV:STA:UNLO?": "a PLL is UNLOCKED",
              ":DEV:STA:ADCOVER?": "ADC OVERLOAD (reduce stimulus power)",
              ":DEV:STA:UNLEV?": "source output is UNLEVELLED"}
    for query, text in checks.items():
        try:
            if vna.query(query) == "TRUE":
                warnings.append(text)
        except Exception:
            pass    # older GUI versions may not support every query
    return warnings


def run_single_sweep(vna):
    """
    Trigger ONE new sweep, wait until it has FINISHED, and return the trace.

    Completion checks:
      * :VNA:ACQ:FIN? must return TRUE (averaging complete) before reading,
      * the number of returned points must equal the configured points.

    Returns
    -------
    freqs (numpy array, Hz), s11 (numpy complex array)
    """
    expected_points = int(float(vna.query(":VNA:ACQ:POINTS?")))

    # Single-sweep mode: "SINGLE TRUE" always triggers a fresh sweep.
    vna.cmd(":VNA:ACQ:SINGLE TRUE")
    vna.cmd(":VNA:ACQ:RUN")

    # Give the GUI a moment to register the new sweep (FIN? drops to FALSE),
    # so that an old "finished" flag is not mistaken for the new sweep.
    t_start = time.time()
    while vna.query(":VNA:ACQ:FIN?") == "TRUE" and time.time() - t_start < 2.0:
        time.sleep(0.05)

    # Now wait for this sweep to finish.
    print("    sweeping", end="", flush=True)
    while vna.query(":VNA:ACQ:FIN?") != "TRUE":
        if time.time() - t_start > SWEEP_TIMEOUT_S:
            raise TimeoutError("Sweep did not finish within SWEEP_TIMEOUT_S.")
        print(".", end="", flush=True)
        time.sleep(0.2)
    print(" done")

    data = vna.query(":VNA:TRACE:DATA? " + TRACE_NAME, timeout=QUERY_TIMEOUT_S)
    points = libreVNA.parse_VNA_trace_data(data)   # list of (freq, complex)

    freqs = np.array([p[0] for p in points])
    s11 = np.array([p[1] for p in points])
    if len(freqs) != expected_points:
        raise RuntimeError(f"Incomplete trace: {len(freqs)} of {expected_points} points.")
    return freqs, s11


def extract_resonance(freqs, s11):
    """
    Run the currently selected resonance-extraction method on one trace.

    Returns (Measurement, warnings). Falls back to the half-power method
    if the fit is unavailable or fails, saying so in the warnings, so a
    single awkward trace never aborts a long depth series.

    When RESONANCE_METHOD is "fit" and USE_UNLOADED_Q is set, the returned
    Measurement carries the UNLOADED Q0 rather than the loaded QL, because
    that is the quantity perturbation theory wants. The fitted Q0 and beta
    stay attached as .Q0_unloaded and .beta_coupling either way.
    """
    warnings = []
    if RESONANCE_METHOD == "half_power":
        return CPM.find_resonance(freqs, s11), warnings
    if RESONANCE_METHOD == "half_power_norm":
        base = CPM.baseline_level(freqs, s11)
        if abs(base - 1.0) > 0.05:
            warnings.append(
                f"off-resonance |S11| = {base:.3f}, well away from 1 - the "
                "calibration may not be holding at the reference plane")
        return CPM.find_resonance(freqs, s11, normalise=True), warnings

    # "fit"
    try:
        meas = CPM.fit_resonance(freqs, s11)
    except (ImportError, RuntimeError, ValueError) as err:
        warnings.append(f"one-port fit failed ({err}); used the half-power method")
        return CPM.find_resonance(freqs, s11), warnings

    if USE_UNLOADED_Q:
        unloaded = CPM.Measurement(meas.f_res, meas.f_res / meas.Q0_unloaded,
                                   s11_min_db=meas.s11_min_db)
        unloaded.Q0_unloaded = meas.Q0_unloaded
        unloaded.beta_coupling = meas.beta_coupling
        unloaded.fit_rms = meas.fit_rms
        meas = unloaded
    return meas, warnings


def choose_resonance_method():
    """
    Ask which resonance-extraction method should feed the results.

    All of them are recorded in the comparison section of the results
    file regardless; this chooses the one the permittivity is computed
    from. Asked once per partial-insertion run, as requested.
    """
    global RESONANCE_METHOD, USE_UNLOADED_Q

    print("\nResonance extraction. All methods are logged for comparison;")
    print("this picks the one the permittivity is actually computed from.")
    print("  - half-power is exact for the one-port cavity model and is")
    print("    what the handwritten notes specify;")
    print("  - the fit uses the whole trace, is ~10x better on shallow dips,")
    print("    and separates Q0 from the coupling beta, which removes the")
    print("    assumption that coupling is unchanged as the sample moves.")

    options = [
        "One-port cavity fit (recommended; needs SciPy)",
        "Half-absorbed-power, as in the notes",
        "Half-absorbed-power, normalised to the measured baseline",
    ]
    choice = ask_choice("Method:", options)
    if choice.startswith("One-port"):
        RESONANCE_METHOD = "fit"
        USE_UNLOADED_Q = ask_yes_no(
            "Use the fitted UNLOADED Q0 in the perturbation (recommended)?", True)
    elif choice.startswith("Half-absorbed-power, as"):
        RESONANCE_METHOD = "half_power"
    else:
        RESONANCE_METHOD = "half_power_norm"
    print(f"  -> using '{RESONANCE_METHOD}'"
          + (" with unloaded Q0" if RESONANCE_METHOD == "fit" and USE_UNLOADED_Q
             else ""))
    return RESONANCE_METHOD


def analyse_trace(freqs, s11):
    """
    Find the resonance in one trace and check its quality.
    Returns (Measurement, list of warnings). Raises ValueError if the
    resonance cannot be analysed (e.g. the bandwidth points fall outside
    the sweep).

    Which extraction method is used is set by RESONANCE_METHOD; see
    extract_resonance().
    """
    meas, warnings = extract_resonance(freqs, s11)

    if meas.s11_min_db > -MIN_DIP_DB:
        warnings.append(f"shallow dip ({meas.s11_min_db:.1f} dB) - check coupling / is this the resonance?")

    step = freqs[1] - freqs[0]
    if meas.BW / step < MIN_POINTS_IN_BW:
        warnings.append(f"only {meas.BW / step:.1f} points inside the BW - reduce span or add points")

    span = freqs[-1] - freqs[0]
    if meas.f_res - freqs[0] < 0.1 * span or freqs[-1] - meas.f_res < 0.1 * span:
        warnings.append("resonance is close to the edge of the sweep")

    return meas, warnings


def check_consistency(measurements):
    """
    Faulty-sweep check: compare f0 and Q of the repeated sweeps.
    Returns True if both spreads are within tolerance.
    """
    f0s = np.array([m.f_res for m in measurements])
    qs = np.array([m.quality_factor() for m in measurements])

    f_spread_ppm = (f0s.max() - f0s.min()) / f0s.mean() * 1e6
    q_spread = (qs.max() - qs.min()) / qs.mean()

    f_ok = f_spread_ppm <= F0_TOLERANCE_PPM
    q_ok = q_spread <= Q_TOLERANCE_FRAC
    print(f"    f0 spread = {f_spread_ppm:.1f} ppm (limit {F0_TOLERANCE_PPM:g})  "
          f"{'OK' if f_ok else 'FAIL'}")
    print(f"    Q  spread = {q_spread * 100:.2f} %   (limit {Q_TOLERANCE_FRAC * 100:g} %)  "
          f"{'OK' if q_ok else 'FAIL'}")
    return f_ok and q_ok


def measure_repeated(vna, label):
    """
    Take N_REPEATS sweeps of the current cavity state, analyse each one,
    check they agree, and let the user accept / retake / quit.

    Returns
    -------
    measurements : list of CPM.Measurement (one per sweep)
    traces       : list of (freqs, s11) tuples (raw data, for saving)
    """
    while True:                       # loops until the user accepts a set
        measurements = []
        traces = []
        failed = False

        for k in range(N_REPEATS):
            pause(f"[{label}] Sweep {k + 1}/{N_REPEATS}. Check the GUI if you wish.")
            status_warnings = check_hardware_status(vna)
            freqs, s11 = run_single_sweep(vna)

            try:
                meas, warnings = analyse_trace(freqs, s11)
            except ValueError as err:
                print("    ERROR analysing sweep:", err)
                failed = True
                break

            print(f"    f0 = {meas.f_res / 1e9:.6f} GHz   Q = {meas.quality_factor():.1f}   "
                  f"BW = {meas.BW / 1e3:.2f} kHz   S11min = {meas.s11_min_db:.1f} dB")
            for w in status_warnings + warnings:
                print("    WARNING:", w)

            measurements.append(meas)
            traces.append((freqs, s11))

        set_live_view(vna)            # let the user look at the live trace

        if not failed:
            print(f"Consistency check over {N_REPEATS} sweeps:")
            if check_consistency(measurements):
                if ask_yes_no("Accept this measurement set?", True):
                    return measurements, traces
            else:
                print("    The sweeps disagree - possible disturbance (e.g. the cavity was bumped).")

        choice = ask_choice("What next?", ["Retake all sweeps", "Accept anyway", "Quit"])
        if choice == "Accept anyway" and not failed:
            return measurements, traces
        if choice == "Quit":
            sys.exit(0)
        # otherwise loop round and retake


# ============================================================
# CAVITY / SAMPLE INPUT
# ============================================================

def input_cavity():
    """Ask the user for cavity shape, mode and dimensions (mm)."""
    shape_name = ask_choice("Cavity shape:", ["cylindrical", "rectangular"])
    mode_name = ask_choice("Resonant mode:", ["TM010", "TM110", "TM210"])
    if shape_name == "cylindrical":
        a = ask_float("Cavity radius (mm)")
    else:
        a = ask_float("Cavity side length, square cross-section (mm)")
    h = ask_float("Cavity height (mm)")
    return CPM.Cavity(CPM.Cavity.Shape(shape_name), CPM.Cavity.Mode(mode_name), a, h)


def cavity_to_dict(cavity):
    return {"shape": cavity.shape.value, "mode": cavity.mode.value,
            "a_mm": cavity.a, "h_mm": cavity.h}


def cavity_from_dict(d):
    return CPM.Cavity(CPM.Cavity.Shape(d["shape"]), CPM.Cavity.Mode(d["mode"]),
                      d["a_mm"], d["h_mm"])


def choose_senses(cavity):
    """
    Which polarisations to measure.
      TM010              -> ["linear"]
      TM110/TM210        -> ["+", "-"] for mu +/- alpha, or ["linear"]
    '+' and '-' are measured one after the other; the script pauses so the
    feed can be reconfigured between them.
    """
    if not cavity.is_magnetic():
        return ["linear"]
    print("\nFor a magnetised (gyromagnetic) sample, measure both circular senses")
    print("to obtain mu and alpha. Label '+' = the sense giving mu + alpha.")
    if ask_yes_no("Measure both circular senses (+ and -)?", True):
        return ["+", "-"]
    return ["linear"]


def choose_holder():
    """
    Ask whether the sample sits in the quartz tube or goes in bare.

    This decides the BASELINE the sample is differenced against. The empty
    cavity is measured either way, because it normalises the frequency
    shift in both cases.
    """
    print("\nSample holder:")
    print("  The empty cavity is measured either way - it normalises every")
    print("  frequency shift. A quartz tube additionally needs its own empty")
    print("  measurement, which becomes the baseline so the tube cancels.")
    choice = ask_choice("How is the sample held?",
                        ["In the quartz tube",
                         "Bare - no holder (standard formula)"])
    return choice.startswith("In the quartz")


def sense_sign(sense):
    """Convert '+', '-', 'linear' into +1, -1, 0."""
    return {"+": 1, "-": -1, "linear": 0}[sense]


def configure_polarisation(sense):
    """Pause so the user can set the feed for the given sense."""
    if sense != "linear":
        pause(f"Configure the cavity feed for the '{sense}' circular sense.")


def input_known_material(cavity, senses):
    """Ask for the known material values."""
    if not cavity.is_magnetic():
        eps_r = ask_float("Known eps_r' (real permittivity)")
        tan_d = ask_float("Known loss tangent", 0.0)
        return CPM.Material(eps=eps_r, loss_tangent=tan_d)

    mu_r = ask_float("Known mu' (real permeability)")
    mu_i = ask_float("Known mu'' (magnetic loss, positive)", 0.0)
    alpha = None
    if senses == ["+", "-"]:
        a_r = ask_float("Known alpha' (off-diagonal term)", 0.0)
        a_i = ask_float("Known alpha''", 0.0)
        alpha = complex(a_r, -a_i)
    return CPM.Material(mu=complex(mu_r, -mu_i), alpha=alpha)


# ============================================================
# RESULT CALCULATION AND PRINTING
# ============================================================

def mean_std(values):
    """Mean and sample standard deviation of a list of numbers."""
    values = np.asarray(values, dtype=float)
    std = values.std(ddof=1) if len(values) > 1 else 0.0
    return float(values.mean()), float(std)


def evaluate_sweeps(measurements, reference, sample, cavity, empty_cavity=None):
    """
    Material values for each sweep, plus mean and std.

    reference is the baseline (empty tube, or the empty cavity for a bare
    sample); empty_cavity normalises the frequency shift and defaults to
    reference when not supplied.

    Returns a dict of { name: {"values": [...], "mean": m, "std": s} }.
    """
    table = {}

    def add(name, value):
        table.setdefault(name, {"values": []})["values"].append(float(value))

    for meas in measurements:
        result = CPM.measure(meas, reference, sample, cavity, empty_cavity)
        if cavity.is_magnetic():
            add("mu_eff' (notes)", result["notes"].real)
            add("mu_eff'' (notes)", -result["notes"].imag)
            add("mu_eff' (exact)", result["exact"].real)
            add("mu_eff'' (exact)", -result["exact"].imag)
        else:
            add("eps_r'", result.eps.real)
            add("eps_r''", -result.eps.imag)
            add("tan_delta", result.loss_tangent)

    for name in table:
        table[name]["mean"], table[name]["std"] = mean_std(table[name]["values"])
    return table


def print_table(title, table):
    print(f"\n--- {title} ---")
    for name, entry in table.items():
        values = ", ".join(f"{v:.5g}" for v in entry["values"])
        print(f"  {name:18s} = {entry['mean']:.5g} +/- {entry['std']:.2g}   (sweeps: {values})")


def combine_senses(tables):
    """Use the '+' and '-' mean results to separate mu and alpha (both methods)."""
    combined = {}
    for method in ("notes", "exact"):
        mu_plus = complex(tables["+"][f"mu_eff' ({method})"]["mean"],
                          -tables["+"][f"mu_eff'' ({method})"]["mean"])
        mu_minus = complex(tables["-"][f"mu_eff' ({method})"]["mean"],
                           -tables["-"][f"mu_eff'' ({method})"]["mean"])
        mat = CPM.separate_circular(mu_plus, mu_minus)
        combined[method] = {"mu'": mat.mu.real, "mu''": -mat.mu.imag,
                            "alpha'": mat.alpha.real, "alpha''": -mat.alpha.imag}
    print("\n--- Separated tensor components (from '+' and '-' means) ---")
    for method, vals in combined.items():
        print(f"  [{method}] " + "   ".join(f"{k} = {v:.5g}" for k, v in vals.items()))
    return combined


def measurement_to_dict(meas):
    return {"f0_Hz": meas.f_res, "BW_Hz": meas.BW, "Q": meas.quality_factor()}


# ============================================================
# SAVING / LOADING
# ============================================================

def ask_filename():
    """Ask for a base filename (no extension). Blank = timestamp."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    while True:
        name = input("\nFile name to save results (no extension, blank = timestamp): ").strip()
        if name == "":
            name = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
        path = os.path.join(RESULTS_DIR, name)
        if not os.path.exists(path + ".json") or ask_yes_no("File exists - overwrite?", False):
            return path


def save_traces_csv(path, traces):
    """Save raw sweeps: columns freq_Hz, re_1, im_1, re_2, im_2, ..."""
    columns = [traces[0][0]]
    header = ["freq_Hz"]
    for i, (freqs, s11) in enumerate(traces, start=1):
        columns += [s11.real, s11.imag]
        header += [f"S11_re_{i}", f"S11_im_{i}"]
    np.savetxt(path, np.column_stack(columns), delimiter=",",
               header=",".join(header), comments="")


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print("Saved:", path)


def save_quartz_reference(cavity, empty, quartz):
    """
    Store BOTH references for later use.

    empty  : {sense: Measurement} - the empty cavity. Always needed, because
             it normalises every frequency shift.
    quartz : {sense: Measurement} or None - the empty quartz tube, the
             baseline the sample is differenced against. None for a bare
             sample, in which case the empty cavity serves as both.
    """
    data = {"date": datetime.datetime.now().isoformat(timespec="seconds"),
            "cavity": cavity_to_dict(cavity),
            "uses_holder": quartz is not None,
            "empty_cavity": {s: measurement_to_dict(m) for s, m in empty.items()},
            "quartz_tube": (None if quartz is None else
                            {s: measurement_to_dict(m) for s, m in quartz.items()})}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if os.path.exists(QUARTZ_FILE) and not ask_yes_no(
            "A reference file already exists. Overwrite it?", True):
        return
    save_json(QUARTZ_FILE, data)


def load_quartz_reference():
    """
    Return (cavity, empty, quartz, date) or None if no file.

    empty is always a {sense: Measurement} dict; quartz is the same or None
    when the stored run used no holder. Files written before the two-
    reference change have no "uses_holder" key; they are read as
    holder runs, and their empty-cavity block was already being saved.
    """
    if not os.path.exists(QUARTZ_FILE):
        return None
    with open(QUARTZ_FILE) as f:
        data = json.load(f)

    def as_meas(block):
        return {s: CPM.Measurement(d["f0_Hz"], d["BW_Hz"]) for s, d in block.items()}

    empty = as_meas(data["empty_cavity"])
    quartz_block = data.get("quartz_tube")
    quartz = as_meas(quartz_block) if quartz_block else None
    return cavity_from_dict(data["cavity"]), empty, quartz, data["date"]


# ============================================================
# REFERENCE MEASUREMENTS (empty cavity, then empty quartz tube)
# ============================================================

def measure_references(vna, cavity, senses, raw, use_holder=None):
    """
    Measure the references: the empty cavity always, the empty quartz tube
    as well when one is used.

    Returns (empty, quartz). quartz is None for a bare sample, in which
    case the empty cavity is both the normalising reference and the
    baseline. Raw traces are added to the 'raw' dictionary.
    """
    if use_holder is None:
        use_holder = choose_holder()

    empty, quartz = {}, ({} if use_holder else None)

    pause("STEP 2: Make sure the cavity is EMPTY (no tube, no sample).")
    for sense in senses:
        configure_polarisation(sense)
        choose_sweep_settings(vna)
        meas_list, traces = measure_repeated(vna, f"empty cavity, {sense}")
        empty[sense] = CPM.average_measurements(meas_list)
        raw[f"empty_{sense}"] = traces

    if use_holder:
        pause("STEP 3: Insert the EMPTY quartz tube.")
        for sense in senses:
            configure_polarisation(sense)
            choose_sweep_settings(vna)
            meas_list, traces = measure_repeated(vna, f"empty quartz tube, {sense}")
            quartz[sense] = CPM.average_measurements(meas_list)
            raw[f"quartz_{sense}"] = traces

    for sense in senses:
        line = (f"  [{sense}] empty: f0 = {empty[sense].f_res / 1e9:.6f} GHz, "
                f"Q = {empty[sense].quality_factor():.1f}")
        if use_holder:
            line += (f"   |  tube: f0 = {quartz[sense].f_res / 1e9:.6f} GHz, "
                     f"Q = {quartz[sense].quality_factor():.1f}")
            x_tube = ((empty[sense].f_res - quartz[sense].f_res)
                      / empty[sense].f_res)
            line += f"   |  x_tube = {x_tube:.3e}"
        else:
            line += "   |  no holder - empty cavity is the baseline"
        print(line)

    save_quartz_reference(cavity, empty, quartz)
    return empty, quartz


def baseline_for(empty, quartz, sense):
    """The measurement the sample is differenced against, for one sense."""
    return empty[sense] if quartz is None else quartz[sense]


# ============================================================
# SAMPLE MEASUREMENT (shared by both workflows)
# ============================================================

def measure_sample(vna, cavity, sample, empty, quartz, senses, raw,
                   suggestions=None):
    """
    Measure the sample for every sense: N_REPEATS sweeps each, then the
    material property per sweep and its mean/std.

    empty  : the empty cavity, which normalises the frequency shift.
    quartz : the empty tube baseline, or None for a bare sample.

    Returns {sense: table} and the averaged Measurements.
    """
    tables, averaged = {}, {}
    for sense in senses:
        configure_polarisation(sense)
        suggestion = suggestions[sense] if suggestions else None
        choose_sweep_settings(vna, suggestion)
        meas_list, traces = measure_repeated(vna, f"sample, {sense}")
        raw[f"sample_{sense}"] = traces

        averaged[sense] = CPM.average_measurements(meas_list)
        tables[sense] = evaluate_sweeps(meas_list, baseline_for(empty, quartz, sense),
                                        sample, cavity, empty[sense])
        print_table(f"Measured ({sense})", tables[sense])
    return tables, averaged


def save_run(results, raw):
    """Ask for a filename; save JSON summary and one CSV per measurement set."""
    base = ask_filename()
    save_json(base + ".json", results)
    for name, traces in raw.items():
        csv_path = f"{base}_{name.replace('+', 'plus').replace('-', 'minus')}.csv"
        save_traces_csv(csv_path, traces)
    print("Raw traces saved alongside the JSON file.")


# ============================================================
# WORKFLOW 1: KNOWN SAMPLE
# ============================================================

def known_sample_workflow(vna):
    raw = {}

    # Step 1.1 - cavity
    print("\nSTEP 1: Cavity description")
    cavity = input_cavity()
    senses = choose_senses(cavity)

    # Steps 2-3 - references
    empty, quartz = measure_references(vna, cavity, senses, raw)

    # Step 4 - known sample
    print("\nSTEP 4: Known sample properties")
    volume = ask_float("Sample volume (mm^3)")
    sample = CPM.Sample(volume, input_known_material(cavity, senses))

    # Step 5 - prediction and suggested sweep
    print("\nSTEP 5: Prediction")
    predictions, suggestions = {}, {}
    for sense in senses:
        base = baseline_for(empty, quartz, sense)
        pred = CPM.predict(sample, cavity, base, sense_sign(sense), empty[sense])
        predictions[sense] = pred
        suggestions[sense] = suggest_sweep(pred)
        shift = pred.f_res - base.f_res
        print(f"  [{sense}] fs = {pred.f_res / 1e9:.6f} GHz (shift {shift / 1e6:+.4f} MHz),  "
              f"Qs = {pred.quality_factor():.1f},  BW = {pred.BW / 1e3:.2f} kHz")

    # Step 6 - calibration by user
    pause("STEP 6: Perform any VNA calibration in the GUI now if required.")

    # Step 7-8 - insert sample and measure
    pause("STEP 7: Insert the SAMPLE"
          + (" (in the quartz tube)." if quartz is not None else " (bare, no holder)."))
    tables, averaged = measure_sample(vna, cavity, sample, empty, quartz,
                                      senses, raw, suggestions)

    # Compare with the known values
    print("\n--- Predicted vs measured resonance ---")
    for sense in senses:
        p, m = predictions[sense], averaged[sense]
        print(f"  [{sense}] f: {p.f_res / 1e9:.6f} vs {m.f_res / 1e9:.6f} GHz,   "
              f"Q: {p.quality_factor():.1f} vs {m.quality_factor():.1f}")
    combined = combine_senses(tables) if senses == ["+", "-"] else None

    # Step 9 - save
    mat = sample.material
    results = {
        "workflow": "known_sample",
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "cavity": cavity_to_dict(cavity),
        "sample_volume_mm3": volume,
        "known_material": {"eps": str(mat.eps), "loss_tangent": mat.loss_tangent,
                           "mu": str(mat.mu), "alpha": str(mat.alpha)},
        "uses_holder": quartz is not None,
        "empty_cavity": {s: measurement_to_dict(m) for s, m in empty.items()},
        "quartz_reference": (None if quartz is None else
                             {s: measurement_to_dict(m) for s, m in quartz.items()}),
        "predicted": {s: measurement_to_dict(m) for s, m in predictions.items()},
        "measured_average": {s: measurement_to_dict(m) for s, m in averaged.items()},
        "material_results": tables,
        "separated_mu_alpha": combined,
    }
    save_run(results, raw)


# ============================================================
# WORKFLOW 2: UNKNOWN SAMPLE
# ============================================================

def unknown_sample_workflow(vna):
    raw = {}

    # Steps 2-3 - cavity and saved quartz reference
    loaded = load_quartz_reference()
    if loaded is None:
        print("No reference file found - the references must be measured first.")
        cavity = input_cavity()
        senses = choose_senses(cavity)
        empty, quartz = measure_references(vna, cavity, senses, raw)
    else:
        cavity, empty, quartz, date = loaded
        senses = list(empty.keys())
        print(f"\nLoaded references from {date}:")
        print(f"  cavity {cavity.shape.value}, {cavity.mode.value}, a = {cavity.a} mm, h = {cavity.h} mm")
        print("  holder: " + ("quartz tube" if quartz is not None else "none (bare sample)"))
        for s in senses:
            line = (f"  [{s}] empty: f0 = {empty[s].f_res / 1e9:.6f} GHz, "
                    f"Q = {empty[s].quality_factor():.1f}")
            if quartz is not None:
                line += (f"   |  tube: f0 = {quartz[s].f_res / 1e9:.6f} GHz, "
                         f"Q = {quartz[s].quality_factor():.1f}")
            print(line)
        if not ask_yes_no("Use these references?", True):
            cavity = input_cavity()
            senses = choose_senses(cavity)
            empty, quartz = measure_references(vna, cavity, senses, raw)

    volume = ask_float("\nSample volume (mm^3)")
    sample = CPM.Sample(volume)

    # Steps 4-7 - sweep settings, insert, N_REPEATS sweeps, average
    pause("Insert the SAMPLE"
          + (" (in the quartz tube)." if quartz is not None else " (bare, no holder)."))
    tables, averaged = measure_sample(vna, cavity, sample, empty, quartz, senses, raw)
    combined = combine_senses(tables) if senses == ["+", "-"] else None

    results = {
        "workflow": "unknown_sample",
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "cavity": cavity_to_dict(cavity),
        "sample_volume_mm3": volume,
        "uses_holder": quartz is not None,
        "empty_cavity": {s: measurement_to_dict(m) for s, m in empty.items()},
        "quartz_reference": (None if quartz is None else
                             {s: measurement_to_dict(m) for s, m in quartz.items()}),
        "measured_average": {s: measurement_to_dict(m) for s, m in averaged.items()},
        "material_results": tables,
        "separated_mu_alpha": combined,
    }
    save_run(results, raw)


# ============================================================
# WORKFLOW 3: PARTIAL INSERTION (Orloff-corrected)
# ============================================================
#
# Why: a high-loss sample at full insertion over-perturbs the cavity and
# the S11 dip collapses. Withdrawing it reduces the perturbation, but the
# single-point formula then degrades because the field is non-uniform
# near the sample's leading edge and the insertion hole.
#
# Orloff's answer is to measure at MANY depths. With the effective volume
# fraction v = V'_s/V_cavity, the perturbation equations are straight
# lines whose SLOPES give the permittivity and whose INTERCEPTS absorb the
# non-uniform-field term beta. See CPM.py "PARTIAL INSERTION" and the
# notebook section of the same name.
#
# Because the quartz tube is withdrawn WITH the sample, its own
# perturbation changes with depth. It is measured once as its own depth
# series (holder_series_workflow) and interpolated onto the sample's
# depths, so a sample run still needs only N_REPEATS sweeps per depth.


def ask_depths(cavity):
    """Ask for the insertion depths of a series, in mm."""
    print(f"\nInsertion depths, in mm, measured from the end plate the sample "
          f"enters.\nCavity height is {cavity.h:g} mm (= 100 % filling).")
    print("Orloff fits between about 30 % and 70 % filling; take a few points "
          "outside\nthat range too, so the departure from linearity can be seen.")

    if ask_yes_no("Generate an evenly spaced list?", True):
        lo = ask_float("  lowest depth (mm)", round(0.20 * cavity.h, 2))
        hi = ask_float("  highest depth (mm)", round(0.80 * cavity.h, 2))
        n = int(ask_float("  number of depths", 9))
        depths = list(np.linspace(lo, hi, max(2, n)))
    else:
        depths = []
        print("  Enter depths one per line; blank line to finish.")
        while True:
            text = input("  depth (mm): ").strip()
            if text == "":
                break
            try:
                depths.append(float(text))
            except ValueError:
                print("    Please enter a number.")

    depths = sorted(d for d in depths if 0.0 < d <= cavity.h)
    if len(depths) < 3:
        print("Need at least 3 valid depths.")
        return ask_depths(cavity)
    print("  depths (mm): " + ", ".join(f"{d:.2f}" for d in depths))
    print("  filling     : " + ", ".join(f"{d / cavity.h * 100:.0f}%" for d in depths))
    return depths


def measure_depth_series(vna, cavity, depths, label, raw, raw_prefix):
    """
    Walk a list of insertion depths, taking N_REPEATS sweeps at each.

    Returns (list of Measurement, dict of per-depth comparison data).
    The sweep suggestion for each depth comes from the previous one, since
    the resonance walks steadily as the sample goes in.
    """
    measurements = []
    comparisons = {}
    previous = None

    for i, depth in enumerate(depths):
        fill = depth / cavity.h * 100.0
        pause(f"[{label}] Set the insertion depth to {depth:.2f} mm "
              f"({fill:.0f} % filling).  Point {i + 1} of {len(depths)}.")

        suggestion = None
        if previous is not None:
            span = max(10.0 * previous.BW, 5.0 * previous.BW)
            suggestion = {"start": previous.f_res - span / 2.0,
                          "stop": previous.f_res + span / 2.0,
                          "points": 401}
        choose_sweep_settings(vna, suggestion)

        meas_list, traces = measure_repeated(vna, f"{label}, {depth:.2f} mm")
        raw[f"{raw_prefix}_{depth:.2f}mm"] = traces

        averaged = CPM.average_measurements(meas_list)
        measurements.append(averaged)
        previous = averaged

        # All extraction methods, on every sweep, for the comparison section.
        comparisons[f"{depth:.3f}"] = [
            CPM.compare_resonance_methods(f, s) for f, s in traces]

        print(f"    depth {depth:6.2f} mm  ({fill:5.1f} %)   "
              f"f0 = {averaged.f_res / 1e9:.6f} GHz   Q = {averaged.quality_factor():.1f}")

    return measurements, comparisons


# ------------------------------------------------------------
# Holder (quartz tube) depth series
# ------------------------------------------------------------

def holder_series_workflow(vna):
    """
    Measure the EMPTY quartz tube over a range of insertion depths.

    Needed because the tube is withdrawn together with the sample, so its
    perturbation is depth-dependent and cannot be removed by a single
    fixed reference (see CPM.correct_for_holder and Orloff eqs 12-13).

    Saved to holder_series.json and reused by every later sample run, so
    this only has to be done once per cavity/tube combination.
    """
    raw = {}
    print("\nSTEP 1: Cavity description")
    cavity = input_cavity()
    if cavity.mode != CPM.Cavity.Mode.TM010:
        print("Partial insertion is implemented for TM010 only - see CPM.py.")
        return
    choose_resonance_method()

    pause("STEP 2: Make sure the cavity is EMPTY (no tube, no sample).")
    choose_sweep_settings(vna)
    empty_list, empty_traces = measure_repeated(vna, "empty cavity")
    empty = CPM.average_measurements(empty_list)
    raw["empty"] = empty_traces
    print(f"  empty cavity: f0 = {empty.f_res / 1e9:.6f} GHz, "
          f"Q = {empty.quality_factor():.1f}")

    pause("STEP 3: Insert the EMPTY quartz tube. It will now be measured at "
          "a range of depths.")
    depths = ask_depths(cavity)
    holder, comparisons = measure_depth_series(
        vna, cavity, depths, "empty tube", raw, "holder")

    data = {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "resonance_method": RESONANCE_METHOD,
        "use_unloaded_q": USE_UNLOADED_Q,
        "cavity": cavity_to_dict(cavity),
        "empty_cavity": measurement_to_dict(empty),
        "depths_mm": list(depths),
        "holder": [measurement_to_dict(m) for m in holder],
        "resonance_method_comparison": comparisons,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if os.path.exists(HOLDER_SERIES_FILE) and not ask_yes_no(
            "A holder series already exists. Overwrite it?", True):
        print("Not saved.")
    else:
        save_json(HOLDER_SERIES_FILE, data)

    base = ask_filename()
    save_json(base + ".json", data)
    for name, traces in raw.items():
        save_traces_csv(f"{base}_{name}.csv", traces)


def load_holder_series():
    """Return (cavity, empty_cavity Measurement, depths, holder list) or None."""
    if not os.path.exists(HOLDER_SERIES_FILE):
        return None
    with open(HOLDER_SERIES_FILE) as f:
        data = json.load(f)
    cavity = cavity_from_dict(data["cavity"])
    empty = CPM.Measurement(data["empty_cavity"]["f0_Hz"],
                            data["empty_cavity"]["BW_Hz"])
    depths = [float(d) for d in data["depths_mm"]]
    holder = [CPM.Measurement(d["f0_Hz"], d["BW_Hz"]) for d in data["holder"]]
    return cavity, empty, depths, holder, data.get("date", "unknown")


def interpolate_holder(depths_h, holder, depth):
    """
    Holder measurement at an arbitrary depth, by interpolating the series.

    f and 1/Q are both smooth in depth, so each is interpolated linearly.
    Extrapolation is refused: the holder series must bracket the sample
    depths, otherwise the correction is guesswork.
    """
    depths_h = np.asarray(depths_h, dtype=float)
    order = np.argsort(depths_h)
    depths_h = depths_h[order]
    f_h = np.array([holder[i].f_res for i in order])
    inv_q_h = np.array([1.0 / holder[i].quality_factor() for i in order])

    if depth < depths_h[0] - 1e-9 or depth > depths_h[-1] + 1e-9:
        raise ValueError(
            f"depth {depth:.2f} mm is outside the holder series "
            f"({depths_h[0]:.2f} to {depths_h[-1]:.2f} mm); extend the holder "
            "series or keep the sample depths inside it.")

    f_i = float(np.interp(depth, depths_h, f_h))
    inv_q_i = float(np.interp(depth, depths_h, inv_q_h))
    return CPM.Measurement(f_i, f_i * inv_q_i)


# ------------------------------------------------------------
# Fit, report, refit
# ------------------------------------------------------------

def report_fit(result, cavity):
    """Print the fit, its residuals, and the single-point comparison."""
    print("\n--- Partial-insertion fit ---")
    print(result.summary())

    print("\n  depth    fill       v        x (ppm)   y/2 (ppm)   "
          "resid_x   single-pt eps'")
    res_map = {}
    used_idx = np.flatnonzero(result.used)
    for k, idx in enumerate(used_idx):
        res_map[int(idx)] = result.residual_r[k]
    for i, d in enumerate(result.depth):
        mark = "*" if result.used[i] else " "
        r = f"{res_map[i] * 1e6:9.2f}" if i in res_map else "        -"
        print(f" {mark}{d:7.2f} {d / cavity.h * 100:6.1f}%  {result.v[i]:.4e}"
              f" {result.x[i] * 1e6:10.2f} {result.y[i] / 2 * 1e6:10.2f}"
              f" {r}   {result.single_point_eps_real[i]:10.4f}")
    print("  (* = inside the fit window; residuals and x,y in ppm)")

    drift = (result.single_point_eps_real[-1] - result.single_point_eps_real[0])
    print(f"\n  Single-point eps' drifts {result.single_point_eps_real[0]:.4f} "
          f"-> {result.single_point_eps_real[-1]:.4f} across the series "
          f"({drift:+.4f}).")
    if abs(drift) < 0.02 * abs(result.eps_real):
        print("  That is flat, so beta is small and the conventional")
        print("  single-point formula was already adequate for this sample.")
    else:
        print("  That drift is the signature of a non-zero beta - the")
        print("  correction is doing real work here.")

    if abs(result.beta_r) < 2 * result.beta_r_err:
        print(f"  beta_r is within 2 sigma of zero, so the non-uniform-field")
        print(f"  term is not resolved above the noise.")


def refit_interactively(series, cavity):
    """Fit, show the result, and let the user adjust the window and refit."""
    window = DEFAULT_FIT_WINDOW
    while True:
        try:
            result = series.fit(window=window)
        except ValueError as err:
            print("  Cannot fit:", err)
            if not ask_yes_no("Choose a different window?", True):
                raise
            window = (ask_float("  window lower bound (fraction of fill)", 0.0),
                      ask_float("  window upper bound (fraction of fill)", 1.0))
            continue

        report_fit(result, cavity)
        choice = ask_choice("\nWhat next?", ["Accept this fit",
                                             "Change the window and refit",
                                             "Use every point"])
        if choice == "Accept this fit":
            return result
        if choice == "Use every point":
            window = (0.0, 1.0)
        else:
            window = (ask_float("  window lower bound (fraction of fill)",
                                window[0]),
                      ask_float("  window upper bound (fraction of fill)",
                                window[1]))


def report_coupling_drift(measurements, depths):
    """
    Report how the coupling coefficient moved across the series.

    Only available when the one-port fit was used. This is a direct test
    of the assumption that the external Q cancels in D(1/Q): if beta
    drifts, that assumption fails for the half-power method, and a 5 %
    drift costs about 3 % in eps''.
    """
    betas = [getattr(m, "beta_coupling", None) for m in measurements]
    if any(b is None for b in betas):
        return None
    lo, hi = min(betas), max(betas)
    spread = (hi - lo) / np.mean(betas) * 100.0
    print(f"\n  Coupling beta across the series: {lo:.4f} to {hi:.4f} "
          f"({spread:.1f} % spread)")
    if spread > 2.0:
        print("  beta is NOT constant with insertion depth. The assumption that")
        print("  coupling cancels in D(1/Q) does not hold here - which is why")
        print("  this run used the fitted unloaded Q0 instead of the loaded Q.")
    else:
        print("  beta is effectively constant, so the coupling assumption holds")
        print("  for this cavity and the loaded-Q route would also have been safe.")
    return {"beta": [float(b) for b in betas],
            "depths_mm": [float(d) for d in depths],
            "spread_percent": float(spread)}


def partial_insertion_workflow(vna):
    """Orloff-corrected depth-series measurement of a dielectric sample."""
    raw = {}

    loaded = load_holder_series()
    use_holder = False
    if loaded is not None:
        cavity, empty, depths_h, holder, date = loaded
        print(f"\nFound a holder (quartz tube) series from {date}:")
        print(f"  cavity {cavity.shape.value}, {cavity.mode.value}, "
              f"a = {cavity.a} mm, h = {cavity.h} mm")
        print(f"  {len(depths_h)} depths from {min(depths_h):.2f} to "
              f"{max(depths_h):.2f} mm")
        use_holder = ask_yes_no("Use it to correct for the tube?", True)

    if not use_holder:
        print("\nNo holder correction will be applied. That is correct only if")
        print("the sample is self-supporting, or the tube stays fully inserted")
        print("while only the sample column height changes.")
        if not ask_yes_no("Continue without a holder correction?", False):
            print("Run the holder-series workflow first.")
            return
        print("\nSTEP 1: Cavity description")
        cavity = input_cavity()
        if cavity.mode != CPM.Cavity.Mode.TM010:
            print("Partial insertion is implemented for TM010 only - see CPM.py.")
            return
        choose_resonance_method()
        pause("STEP 2: Make sure the cavity is EMPTY.")
        choose_sweep_settings(vna)
        empty_list, empty_traces = measure_repeated(vna, "empty cavity")
        empty = CPM.average_measurements(empty_list)
        raw["empty"] = empty_traces
        depths_h, holder = None, None
    else:
        choose_resonance_method()

    r_sample = ask_float("\nSample (powder column) radius (mm)")

    pause("Load the SAMPLE into the tube and prepare to insert.")
    depths = ask_depths(cavity)
    if use_holder:
        outside = [d for d in depths
                   if d < min(depths_h) - 1e-9 or d > max(depths_h) + 1e-9]
        if outside:
            print(f"  WARNING: {len(outside)} depth(s) fall outside the holder "
                  f"series and cannot be corrected: "
                  + ", ".join(f"{d:.2f}" for d in outside))
            if not ask_yes_no("Drop those depths and continue?", True):
                return
            depths = [d for d in depths if d not in outside]

    sample_meas, comparisons = measure_depth_series(
        vna, cavity, depths, "sample", raw, "sample")

    # --- assemble the series ---
    series = CPM.InsertionSeries(cavity, r_sample, empty)
    for depth, meas in zip(depths, sample_meas):
        holder_at_depth = (interpolate_holder(depths_h, holder, depth)
                           if use_holder else None)
        series.add(depth, meas, holder_at_depth)

    result = refit_interactively(series, cavity)
    coupling = report_coupling_drift(sample_meas, depths)

    results = {
        "workflow": "partial_insertion",
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "cavity": cavity_to_dict(cavity),
        "sample_radius_mm": r_sample,
        "resonance_method": RESONANCE_METHOD,
        "use_unloaded_q": USE_UNLOADED_Q,
        "holder_corrected": use_holder,
        "holder_series_date": loaded[4] if loaded else None,
        "empty_cavity": measurement_to_dict(empty),
        "measured_per_depth": {f"{d:.3f}": measurement_to_dict(m)
                               for d, m in zip(depths, sample_meas)},
        "partial_insertion": result.to_dict(),
        "material_result": {
            "eps_real": result.eps_real, "eps_real_err": result.eps_real_err,
            "eps_imag": result.eps_imag, "eps_imag_err": result.eps_imag_err,
            "tan_delta": result.tan_delta,
        },
        # Requested separately: only written for a partial-insertion run.
        "resonance_method_comparison": comparisons,
        "coupling_drift": coupling,
    }
    save_run(results, raw)


def refit_saved_run():
    """Re-fit a saved partial-insertion run with a different window, no VNA."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    files = sorted(f for f in os.listdir(RESULTS_DIR) if f.endswith(".json"))
    runs = []
    for name in files:
        try:
            with open(os.path.join(RESULTS_DIR, name)) as f:
                data = json.load(f)
        except (ValueError, OSError):
            continue
        if data.get("workflow") == "partial_insertion":
            runs.append((name, data))
    if not runs:
        print("No saved partial-insertion runs found in ./results.")
        return

    name = ask_choice("Which run?", [n for n, _ in runs])
    data = dict(runs)[name]
    cavity = cavity_from_dict(data["cavity"])
    pi = data["partial_insertion"]

    # Rebuild the series from the ALREADY CORRECTED observables, so the
    # holder correction is not applied twice.
    empty = CPM.Measurement(data["empty_cavity"]["f0_Hz"],
                            data["empty_cavity"]["BW_Hz"])
    f_c = empty.f_res
    inv_q_c = 1.0 / empty.quality_factor()
    series = CPM.InsertionSeries(cavity, data["sample_radius_mm"], empty)
    for depth, x, y in zip(pi["depth"], pi["x_freq_shift"], pi["y_inv_q_shift"]):
        f_s = f_c * (1.0 - x)
        q_s = 1.0 / (inv_q_c + y)
        series.add(depth, CPM.Measurement(f_s, f_s / q_s))

    print(f"\nLoaded {name}: {len(pi['depth'])} depths, previous window "
          f"{pi['window_fill_fraction'][0] * 100:.0f}-"
          f"{pi['window_fill_fraction'][1] * 100:.0f} %, "
          f"eps' = {data['material_result']['eps_real']:.4f}")

    result = refit_interactively(series, cavity)
    if ask_yes_no("Save the refit alongside the original?", True):
        data["partial_insertion_refit"] = result.to_dict()
        data["material_result_refit"] = {
            "eps_real": result.eps_real, "eps_real_err": result.eps_real_err,
            "eps_imag": result.eps_imag, "eps_imag_err": result.eps_imag_err,
            "tan_delta": result.tan_delta,
        }
        save_json(os.path.join(RESULTS_DIR, name), data)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print(" Cavity perturbation measurement - LibreVNA")
    print("=" * 60)
    vna = connect_vna()

    try:
        while True:
            choice = ask_choice("\nSelect workflow:",
                                ["Known sample (calibration / validation)",
                                 "Unknown sample (measurement)",
                                 "Holder series (empty quartz tube vs depth)",
                                 "Partial insertion (depth series, Orloff)",
                                 "Re-fit a saved partial-insertion run",
                                 "Quit"])
            if choice.startswith("Known"):
                known_sample_workflow(vna)
            elif choice.startswith("Unknown"):
                unknown_sample_workflow(vna)
            elif choice.startswith("Holder"):
                holder_series_workflow(vna)
            elif choice.startswith("Partial"):
                partial_insertion_workflow(vna)
            elif choice.startswith("Re-fit"):
                refit_saved_run()
            else:
                break
    finally:
        # Always leave the GUI sweeping continuously
        set_live_view(vna)


if __name__ == "__main__":
    main()
