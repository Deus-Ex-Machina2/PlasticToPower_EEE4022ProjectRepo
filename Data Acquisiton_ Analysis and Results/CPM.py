"""
CPM.py - Cavity Perturbation Method (analysis module)
=====================================================

Pure analysis code: no VNA communication happens here. The VNA control and
the measurement workflows live in data_analysis.py, which imports this file.

Contents
--------
    Material, Cavity, Sample, Measurement, Perturbation   (data classes)
    find_resonance()        f0 and half-absorbed-power BW from an S11 trace
    fit_resonance()         f0 and Q from a complex one-port resonator fit
    average_measurements()  mean f0 and Q of repeated sweeps
    TM010/TM110/TM210 _predict()   known sample  -> expected f and Q
    TM010/TM110/TM210 _measure()   measured f, Q -> material property
    separate_circular()     (mu + alpha), (mu - alpha) -> mu and alpha
    predict(), measure()    choose the correct function from cavity.mode

    --- partial insertion (Orloff et al. 2014), see PARTIAL INSERTION below ---
    effective_volume()      V'_s for a rod partially inserted to depth h
    effective_volume_annulus()   the same for a tube wall (the holder)
    effective_filling()     v = V'_s / V_cavity
    correct_for_holder()    remove a holder that moves with the sample
    InsertionPoint, InsertionSeries, PartialInsertionResult
    fit_partial_insertion() depth series -> complex permittivity

Conventions and assumptions (read before using)
-----------------------------------------------
1.  Complex material values use the engineering convention
        eps = eps' - j*eps''      mu = mu' - j*mu''      alpha = alpha' - j*alpha''
    so a lossy material has a POSITIVE eps'' / mu''. In Python this is
        eps = eps_real - 1j * eps_imag

2.  Perturbation theory (small-sample, first-order). The complex frequency
    shift of the cavity is
        d_omega / omega = -(f0 - fs)/f0  +  (j/2) * (1/Qs - 1/Q0)
    and for every mode here it is written as
        -(f0 - fs)/f0 + (j/2) * D(1/Q)  =  -C * Vr * F(material)
    where
        C   = mode coupling coefficient     (see MODE_COEFFICIENTS)
        Vr  = V_sample / V_cavity           (volume filling ratio)
        F   = (eps - 1)                     for TM010 (electric field on axis)
        F   = (mu - 1)/(mu + 1)             for TM110 / TM210 (magnetic field)
    Splitting into real and imaginary parts gives, with
        x  = (f0 - fs)/f0      (normalised frequency shift, positive = downwards)
        y  = 1/Qs - 1/Q0       (change in inverse Q, positive = more loss)
    the two working equations
        Re(F) =  x / (C*Vr)
        Im(F) = -y / (2*C*Vr)    ->   F'' = y / (2*C*Vr)

3.  TWO REFERENCES are used, and they are different things:

        empty cavity   f_c, Q_c. Always measured. It NORMALISES the
                       frequency shift, because first-order perturbation
                       theory is written relative to the unperturbed cavity.

        baseline       what the sample is DIFFERENCED against. With a
                       quartz tube this is the empty-tube measurement, so
                       the tube's own perturbation cancels. With a bare
                       sample it is the empty cavity itself.

    so, writing f_ch for the baseline and f_chs for the loaded cavity,

        x = (f_ch - f_chs) / f_c          y = 1/Q_chs - 1/Q_ch

    With no holder f_ch = f_c and this is the textbook (f_c - f_cs)/f_c.

    Normalising by the baseline instead of the empty cavity inflates x by
    1/(1 - x_holder), a bias of roughly +0.2 to +0.6 % in eps' for a quartz
    tube, always positive. eps'' is unaffected: 1/Q is additive and no
    reference frequency divides it. Passing empty_cavity=None everywhere
    restores the old single-reference behaviour.

4.  Q is the LOADED Q obtained from the S11 half-absorbed-power bandwidth
    (see find_resonance; this is NOT a fixed -3 dB level). It is assumed
    that the coupling (external Q) does not change when the sample is
    inserted, so it cancels in D(1/Q) = 1/Qs - 1/Q0.

    fit_resonance() lifts this assumption: it fits the one-port cavity
    model and returns the UNLOADED Q0 and the coupling beta separately,
    so D(1/Q0) needs no cancellation. This matters most for partial
    insertion, where the sample slides past the coupling loop.

5.  Lengths and volumes only need consistent units (mm and mm^3 are used
    in data_analysis.py). Only the ratio Vr is used. The partial-insertion
    functions take lengths in the SAME unit as the cavity dimensions,
    because they form the dimensionless product k*r.

6.  Partial insertion replaces the geometric filling factor Vr by an
    EFFECTIVE filling factor v = V'_s/V_cavity and adds an intercept
    (the non-uniform-field term). The single-point equations above are
    the special case v -> Vr, intercept -> 0. Nothing else changes; in
    particular the mode coefficients C are unaltered. The derivation is
    in CPMCavityCode/CPM_params_code.ipynb, section "Partial Insertion
    and the Effective Volume".

Dependencies
------------
NumPy only, for everything in the original single-point workflow. SciPy is
imported lazily inside fit_resonance() alone, so the lab PC needs it only if
the complex-resonator fit is actually used.
"""

import numpy as np
from enum import Enum


# ============================================================
# MODE COUPLING COEFFICIENTS
# ============================================================
# Bessel values are hard-coded (computed with scipy.special) so that
# scipy is not needed on the lab PC.
#   x01 = 2.404825557695773   first zero of J0   -> TM010
#   x11 = 3.831705970207512   first zero of J1   -> TM110
J1_AT_X01 = 0.5191474972894669     # J1(x01)
J0_AT_X11 = -0.402759395702553     # J0(x11)

MODE_COEFFICIENTS = {
    # C010 = 1 / (2 J1(x01)^2) = 1.8552   (literature: eps' = 1 + 0.539 (Vc/Vs) x)
    "TM010": 1.0 / (2.0 * J1_AT_X01 ** 2),
    # C110 = 1 / (2 J0(x11)^2) = 3.0823
    "TM110": 1.0 / (2.0 * J0_AT_X11 ** 2),
    # C210 = 16/5, as given in the handwritten notes.
    # OPEN ITEM: verify against the source derivation (no Bessel term here).
    "TM210": 16.0 / 5.0,
}


# ============================================================
# MATERIAL
# ============================================================

class Material:
    """
    Electromagnetic properties of a material.

    Parameters
    ----------
    mu : complex or float
        Relative permeability, mu' - j mu''.
    eps : complex or float
        Relative permittivity, eps' - j eps''.
    loss_tangent : float or None
        Dielectric loss tangent eps''/eps', if known.
    alpha : complex or None
        Off-diagonal (gyromagnetic) permeability tensor term, alpha' - j alpha''.

    Notes
    -----
    Properties which are unknown or not applicable should be
    represented by None rather than an arbitrary numerical value.
    """

    def __init__(self, mu=None, eps=None, loss_tangent=None, alpha=None):
        self.mu = mu
        self.eps = eps
        self.loss_tangent = loss_tangent
        self.alpha = alpha


# ============================================================
# CAVITY
# ============================================================

class Cavity:
    """
    Resonant cavity.

    Parameters
    ----------
    shape : Cavity.Shape
    mode : Cavity.Mode
        Resonant mode used for the measurement (sets the coefficient C).
    length : float
        Radius (cylindrical) or side length of the square cross-section
        (rectangular).
    height : float
        Cavity height.
    """

    class Shape(Enum):
        CYLINDRICAL = "cylindrical"
        RECTANGULAR = "rectangular"

    class Mode(Enum):
        TM010 = "TM010"    # E-field max on axis   -> permittivity
        TM110 = "TM110"    # H-field max on axis   -> permeability
        TM210 = "TM210"    # H-field               -> permeability

    def __init__(self, shape, mode, length, height):
        if not isinstance(shape, self.Shape):
            raise TypeError("shape must be a Cavity.Shape")
        if not isinstance(mode, self.Mode):
            raise TypeError("mode must be a Cavity.Mode")

        self.shape = shape
        self.mode = mode

        # Geometry
        self.a = length
        self.h = height

    def get_volume(self):
        """Return cavity volume (units of length^3)."""
        if self.shape == self.Shape.CYLINDRICAL:
            return np.pi * self.a ** 2 * self.h
        elif self.shape == self.Shape.RECTANGULAR:
            # Assumes a square cross-section of side a.
            return self.a ** 2 * self.h

    def get_coefficient(self):
        """Return the mode coupling coefficient C for this cavity's mode."""
        return MODE_COEFFICIENTS[self.mode.value]

    def is_magnetic(self):
        """True if the mode measures permeability (TM110 / TM210)."""
        return self.mode != self.Mode.TM010


# ============================================================
# SAMPLE
# ============================================================

class Sample:
    """
    Physical sample placed inside the cavity.

    The material may be None when the material properties are
    unknown and are intended to be inferred from a measurement.
    """

    def __init__(self, volume, material=None):
        self.volume = volume
        self.material = material


# ============================================================
# MEASUREMENT
# ============================================================

class Measurement:
    """
    Experimental cavity perturbation measurement (one resonance).

    Parameters
    ----------
    resonant_f : float
        Resonant frequency in Hz.
    BW : float
        Half-absorbed-power bandwidth in Hz (see find_resonance).
    s11_min_db : float or None
        Depth of the S11 dip at resonance (dB), kept for sanity checks.
    """

    def __init__(self, resonant_f, BW, s11_min_db=None):
        self.f_res = resonant_f
        self.BW = BW
        self.s11_min_db = s11_min_db

    def quality_factor(self):
        """Loaded quality factor Q = f0 / BW."""
        return self.f_res / self.BW


def find_resonance(freqs, s11, normalise=False):
    """
    Find the resonant frequency and half-absorbed-power bandwidth from an
    S11 trace.

    Method (from the handwritten notes):
        1. Resonance = minimum of |S11| in dB.
        2. Bandwidth level:  S11_BW = 10 log10( (1 + 10^(S11_f0/10)) / 2 )
           This is NOT a fixed -3 dB. The power absorbed by the cavity is
           1 - |S11|^2; the level above is where that absorbed power has
           fallen to half its value at resonance, which depends on how deep
           the dip is. It tends to -3.01 dB only in the limit of a very deep
           dip (|S11(f0)| -> 0), and moves up for a shallow one.
        3. Walk left and right from the minimum until S11 rises above that
           level, then linearly interpolate between the two bracketing points.
    The minimum position is refined with a parabola through its 3 nearest
    points, which gives sub-point resolution on f0.

    For shallow, heavily damped dips (lossy samples) see fit_resonance(),
    which uses the whole trace rather than the points near the minimum.

    Parameters
    ----------
    freqs : 1-D array of frequencies (Hz)
    s11   : 1-D array of complex S11 values (linear, not dB)
    normalise : bool, default False
        Divide the trace by its off-resonance baseline (baseline_level())
        before locating the bandwidth points.

        DEFAULT IS FALSE, and for calibrated data it should stay False.

        The level formula is EXACT for the one-port cavity model (see
        s11_model): the absorbed power 1 - |S11|^2 is a true Lorentzian
        in Q0*delta, and this level is exactly where it halves, giving
        QL = Q0/(1+beta) with no model error.

        That exactness assumes |S11| -> 1 off resonance, which a good
        calibration delivers. It is worth knowing what happens if the
        calibration drifts: on synthetic traces an off-resonance
        baseline of 0.98 biases Q low by about 4 %, and 0.90 by about
        22 %. Setting normalise=True divides that baseline out.

        The better use of this is as a DIAGNOSTIC rather than a fix. If
        baseline_level() returns something materially different from
        1.00, re-calibrate rather than paper over it - a baseline that
        far off usually means something else is wrong too.

    Returns
    -------
    Measurement

    Raises
    ------
    ValueError if both bandwidth points are not inside the sweep.
    """
    freqs = np.asarray(freqs, dtype=float)
    s11 = np.asarray(s11)
    if normalise:
        s11 = s11 / baseline_level(freqs, s11)
    s11_db = 20.0 * np.log10(np.abs(s11)) #NOTE Used 20log(S11) as S11 is a voltage ratio

    i_min = int(np.argmin(s11_db))
    s_min = s11_db[i_min]
    level = 10.0 * np.log10((1.0 + 10.0 ** (s_min / 10.0)) / 2.0) #NOTE: Here 10log(linear) is used since looking for half power point

    # --- lower bandwidth point: walk left ---
    i = i_min
    while i > 0 and s11_db[i] < level:
        i -= 1
    if s11_db[i] < level:
        raise ValueError("Lower 3 dB point is outside the sweep - widen the span.")
    # interpolate between point i (above level) and i+1 (below level)
    f_low = freqs[i] + (level - s11_db[i]) * (freqs[i + 1] - freqs[i]) / (s11_db[i + 1] - s11_db[i])

    # --- upper bandwidth point: walk right ---
    i = i_min
    last = len(freqs) - 1
    while i < last and s11_db[i] < level:
        i += 1
    if s11_db[i] < level:
        raise ValueError("Upper 3 dB point is outside the sweep - widen the span.")
    f_high = freqs[i - 1] + (level - s11_db[i - 1]) * (freqs[i] - freqs[i - 1]) / (s11_db[i] - s11_db[i - 1])

    # --- refine f0 with a parabola through the minimum and its neighbours ---
    f0 = freqs[i_min]
    if 0 < i_min < last:
        y1, y2, y3 = s11_db[i_min - 1], s11_db[i_min], s11_db[i_min + 1]
        denom = y1 - 2.0 * y2 + y3
        if denom != 0:
            step = freqs[i_min + 1] - freqs[i_min]       # assumes a linear sweep
            f0 = freqs[i_min] + 0.5 * (y1 - y3) / denom * step

    return Measurement(f0, f_high - f_low, s11_min_db=s_min)


def average_measurements(measurements):
    """
    Average repeated measurements: mean f0 and mean Q (BW is then f0/Q).
    """
    f_mean = np.mean([m.f_res for m in measurements])
    q_mean = np.mean([m.quality_factor() for m in measurements])
    out = Measurement(f_mean, f_mean / q_mean)

    # Carry through the extras that fit_resonance() attaches, so the
    # coupling can be tracked across a depth series. Averaged only when
    # every input has them, i.e. when the fit was used throughout.
    for attr in ("Q0_unloaded", "beta_coupling", "fit_rms"):
        values = [getattr(m, attr, None) for m in measurements]
        if all(v is not None for v in values):
            setattr(out, attr, float(np.mean(values)))
    return out


# ============================================================
# PERTURBATION
# ============================================================

class Perturbation:
    """
    Generic cavity perturbation quantities (x, y and C*Vr).

    This class deliberately contains no assumptions about whether
    the experiment is magnetic, dielectric, ferromagnetic,
    gyromagnetic, etc. The mode-specific formulae are in the
    TMxxx_predict / TMxxx_measure functions below.

    TWO REFERENCES
    --------------
    These are different things and the distinction matters:

      reference    the BASELINE the sample is differenced against. With a
                   quartz tube this is the empty-tube measurement; with a
                   bare sample it is the empty cavity.

      empty_cavity the NORMALISING frequency, always the empty cavity.

    First-order perturbation shifts are additive over disjoint regions
    relative to the UNPERTURBED cavity, so the sample's own shift is

        x = (f_holder - f_sample) / f_empty

    and not (f_holder - f_sample)/f_holder. Dividing by the holder instead
    inflates x by 1/(1 - x_holder); for a quartz tube that is a bias of
    about +0.2 to +0.6 % in eps', always positive because the tube always
    pulls the resonance down. eps'' is unaffected either way, because 1/Q
    is additive and no reference frequency divides it.

    With no holder, reference and empty_cavity are the same measurement and
    this reduces to the textbook x = (f_c - f_cs)/f_c.

    Parameters
    ----------
    cavity : Cavity
    sample : Sample
    reference_measurement : Measurement
        The baseline to difference against: the empty quartz tube when one
        is used, otherwise the empty cavity.
    coefficient : float or None
        Mode coefficient C. If None it is taken from cavity.mode.
    empty_cavity : Measurement or None
        The empty cavity, used only to normalise the frequency shift. If
        None it falls back to reference_measurement, which reproduces the
        pre-correction behaviour exactly - so old callers are unchanged.
    """

    def __init__(self, cavity, sample, reference_measurement, coefficient=None,
                 empty_cavity=None):
        self.cavity = cavity
        self.sample = sample
        self.reference = reference_measurement
        self.empty_cavity = (empty_cavity if empty_cavity is not None
                             else reference_measurement)
        if coefficient is None:
            coefficient = cavity.get_coefficient()
        self.coefficient = coefficient

    def uses_holder(self):
        """True when the baseline is something other than the empty cavity."""
        return self.reference is not self.empty_cavity

    def get_volume_ratio(self):
        """Return Vr = sample volume / cavity volume (the filling factor)."""
        return self.sample.volume / self.cavity.get_volume()

    def get_scale(self):
        """Return C * Vr, the factor that appears in every formula."""
        return self.coefficient * self.get_volume_ratio()

    def frequency_shift(self, measurement):
        """
        x = (f_reference - f_sample) / f_empty_cavity.

        Positive when the resonance moves down. The numerator differences
        the sample against its baseline (tube or empty cavity); the
        denominator always normalises by the empty cavity, because that is
        what first-order perturbation theory is written relative to.
        """
        return ((self.reference.f_res - measurement.f_res)
                / self.empty_cavity.f_res)

    def quality_shift(self, measurement):
        """
        y = 1/Qs - 1/Q_reference : positive when the sample adds loss.

        Exact, with or without a holder: 1/Q is additive over the cavity,
        holder and sample, so differencing against the holder leaves the
        sample's own loss with no approximation.
        """
        return 1.0 / measurement.quality_factor() - 1.0 / self.reference.quality_factor()

    def predicted_measurement(self, x, y):
        """
        Convert predicted (x, y) back into a Measurement (fs, BW).

        Inverse of frequency_shift / quality_shift, so a predict -> measure
        round trip returns the input exactly.
        """
        fs = self.reference.f_res - x * self.empty_cavity.f_res
        inv_qs = 1.0 / self.reference.quality_factor() + y
        return Measurement(fs, fs * inv_qs)          # BW = fs / Qs


# ============================================================
# PREDICTION  (known sample -> expected fs and Qs)
# ============================================================

def _complex_eps(material):
    """Return complex eps = eps' - j eps'' using eps and/or the loss tangent."""
    eps = complex(material.eps)
    if eps.imag == 0 and material.loss_tangent is not None:
        eps = complex(eps.real, -eps.real * material.loss_tangent)
    return eps


def TM010_predict(sample, cavity, reference, empty_cavity=None):
    """
    Predict the loaded-cavity resonance for a known dielectric sample.

        x = C * Vr * (eps' - 1)
        y = 2 * C * Vr * eps''

    reference is the baseline (empty quartz tube, or the empty cavity for
    a bare sample); empty_cavity normalises the frequency shift and
    defaults to reference. See Perturbation for why they differ.

    Returns a Measurement with the predicted fs and BW.
    """
    pert = Perturbation(cavity, sample, reference, empty_cavity=empty_cavity)
    eps = _complex_eps(sample.material)
    x = pert.get_scale() * (eps.real - 1.0)
    y = 2.0 * pert.get_scale() * (-eps.imag)          # -Im(eps) = eps''
    return pert.predicted_measurement(x, y)


def _magnetic_predict(sample, cavity, reference, sense, empty_cavity=None):
    """
    Shared TM110 / TM210 prediction.

    The sample sees mu_eff = mu + sense*alpha (sense = +1, -1 or 0 for
    linear polarisation / no alpha). Then
        F = (mu_eff - 1) / (mu_eff + 1)          (complex)
        x =  C * Vr * Re(F)
        y = -2 * C * Vr * Im(F)
    """
    pert = Perturbation(cavity, sample, reference, empty_cavity=empty_cavity)
    mat = sample.material
    mu_eff = complex(mat.mu)
    if mat.alpha is not None:
        mu_eff = mu_eff + sense * complex(mat.alpha)
    F = (mu_eff - 1.0) / (mu_eff + 1.0)
    x = pert.get_scale() * F.real
    y = -2.0 * pert.get_scale() * F.imag
    return pert.predicted_measurement(x, y)


def TM110_predict(sample, cavity, reference, sense=0, empty_cavity=None):
    """Predict resonance for a known magnetic sample in TM110 (sense: +1, -1, 0)."""
    return _magnetic_predict(sample, cavity, reference, sense, empty_cavity)


def TM210_predict(sample, cavity, reference, sense=0, empty_cavity=None):
    """Predict resonance for a known magnetic sample in TM210 (sense: +1, -1, 0)."""
    return _magnetic_predict(sample, cavity, reference, sense, empty_cavity)


# ============================================================
# MEASUREMENT  (measured fs, Qs -> material property)
# ============================================================

def TM010_measure(measurement, reference, sample, cavity, empty_cavity=None):
    """
    Complex permittivity from a TM010 measurement.

        eps'  = x / (C*Vr) + 1
        eps'' = y / (2*C*Vr)
        tan_d = eps'' / eps'

    Returns a Material with eps (complex) and loss_tangent filled in.
    """
    pert = Perturbation(cavity, sample, reference, empty_cavity=empty_cavity)
    x = pert.frequency_shift(measurement)
    y = pert.quality_shift(measurement)

    eps_real = x / pert.get_scale() + 1.0
    eps_imag = y / (2.0 * pert.get_scale())
    return Material(eps=complex(eps_real, -eps_imag),
                    loss_tangent=eps_imag / eps_real)


def _magnetic_measure(measurement, reference, sample, cavity, empty_cavity=None):
    """
    Shared TM110 / TM210 inversion for the effective permeability mu_eff
    (= mu + alpha, mu - alpha, or mu for linear polarisation).

    With  F' = x/(C Vr)  and  F'' = y/(2 C Vr):

    Method 1 - handwritten notes (real part FIRST, then imaginary):
        mu'  = (1 + F') / (1 - F')
        mu'' = F'' * (mu' + 1)^2 / 2
        (the mu'' line is the first-order expansion
         d/dmu [(mu-1)/(mu+1)] = 2/(mu+1)^2, valid for mu'' << mu' + 1)

    Method 2 - exact complex inversion (check):
        F   = F' - j F''
        mu  = (1 + F) / (1 - F)

    Returns
    -------
    dict with keys "notes" and "exact", each a complex mu' - j mu''.
    """
    pert = Perturbation(cavity, sample, reference, empty_cavity=empty_cavity)
    F_real = pert.frequency_shift(measurement) / pert.get_scale()
    F_imag = pert.quality_shift(measurement) / (2.0 * pert.get_scale())

    # Method 1: notes (real part must be solved first)
    mu_real = (1.0 + F_real) / (1.0 - F_real)
    mu_imag = F_imag * (mu_real + 1.0) ** 2 / 2.0
    mu_notes = complex(mu_real, -mu_imag)

    # Method 2: exact
    F = complex(F_real, -F_imag)
    mu_exact = (1.0 + F) / (1.0 - F)

    return {"notes": mu_notes, "exact": mu_exact}


def TM110_measure(measurement, reference, sample, cavity, empty_cavity=None):
    """Effective permeability from a TM110 measurement (see _magnetic_measure)."""
    return _magnetic_measure(measurement, reference, sample, cavity, empty_cavity)


def TM210_measure(measurement, reference, sample, cavity, empty_cavity=None):
    """Effective permeability from a TM210 measurement (see _magnetic_measure)."""
    return _magnetic_measure(measurement, reference, sample, cavity, empty_cavity)


def separate_circular(mu_plus, mu_minus):
    """
    Separate the diagonal (mu) and off-diagonal (alpha) tensor terms from
    the two circular-polarisation measurements:

        mu_plus  = mu + alpha        mu_minus = mu - alpha
        mu    = (mu_plus + mu_minus) / 2
        alpha = (mu_plus - mu_minus) / 2

    Works for complex values, so mu'' and alpha'' are separated too.
    Which physical sense (RHCP/LHCP) is '+' is a labelling choice made in
    data_analysis.py and must be kept consistent.
    """
    mu = (mu_plus + mu_minus) / 2.0
    alpha = (mu_plus - mu_minus) / 2.0
    return Material(mu=mu, alpha=alpha)


# ============================================================
# DISPATCHERS  (choose the right function from cavity.mode)
# ============================================================

def predict(sample, cavity, reference, sense=0, empty_cavity=None):
    """Call TM010/TM110/TM210_predict according to cavity.mode."""
    if cavity.mode == Cavity.Mode.TM010:
        return TM010_predict(sample, cavity, reference, empty_cavity)
    elif cavity.mode == Cavity.Mode.TM110:
        return TM110_predict(sample, cavity, reference, sense, empty_cavity)
    else:
        return TM210_predict(sample, cavity, reference, sense, empty_cavity)


def measure(measurement, reference, sample, cavity, empty_cavity=None):
    """Call TM010/TM110/TM210_measure according to cavity.mode."""
    if cavity.mode == Cavity.Mode.TM010:
        return TM010_measure(measurement, reference, sample, cavity, empty_cavity)
    elif cavity.mode == Cavity.Mode.TM110:
        return TM110_measure(measurement, reference, sample, cavity, empty_cavity)
    else:
        return TM210_measure(measurement, reference, sample, cavity, empty_cavity)


# ============================================================
# ============================================================
# PARTIAL INSERTION  (Orloff et al., IEEE T-MTT 62(9) 2149, 2014)
# ============================================================
# ============================================================
#
# Why this exists
# ---------------
# A high-loss sample at full insertion over-perturbs the cavity: the S11
# dip collapses and first-order perturbation stops being valid. Inserting
# the sample only part of the way reduces the perturbation continuously,
# but the single-point formula then degrades, because the field is not
# uniform near the sample's leading edge or near the insertion hole.
#
# Orloff splits that into two separate objects:
#
#   V'_s  the EFFECTIVE SAMPLE VOLUME (his eq. 3). Pure modal geometry:
#         the sample volume weighted by the unperturbed cavity mode shape,
#         normalised to the field maximum. No material property in it.
#
#   beta  the NON-UNIFORM-FIELD TERM (his eq. 7). Depolarisation field,
#         hole fringing, sample-to-wall boundary. This is where the
#         polarisation physics lives, and it is NEVER COMPUTED. It is
#         approximately constant over a window of insertion depths, so it
#         appears as the INTERCEPT of a straight line and is discarded.
#
# With v = V'_s/V_cavity and the mode coefficient C unchanged from
# MODE_COEFFICIENTS, the two working equations are straight lines:
#
#       x = (f0 - fs)/f0      =  C (eps' - 1) v  -  beta_r
#       y/2 = (1/Qs - 1/Q0)/2 =  C  eps''     v  -  beta_i
#
# so a DEPTH SERIES, not a single measurement, gives
#
#       eps'  = 1 + slope_r / C
#       eps'' =     slope_i / C
#       tan d = slope_i / (slope_r + C)
#
# Setting beta = 0 and v -> Vr recovers TM010_measure exactly.
#
# Derivation, with every step checked against the rest of the model:
# CPMCavityCode/CPM_params_code.ipynb, "Partial Insertion and the
# Effective Volume".
#
# Scope: TM010 only, as agreed. TM110/TM210 raise NotImplementedError.
#
# Units: r_s, h and the cavity dimensions must share one unit (mm is used
# throughout data_analysis.py), because only the product k*r and the ratio
# V'_s/V_c are used.


# ------------------------------------------------------------
# Bessel functions without SciPy
# ------------------------------------------------------------
# Inside a cavity the argument is bounded by k*a, i.e. x01 = 2.405 for
# TM010 and x11 = 3.832 for TM110, so the ascending power series is exact
# to machine precision in a few terms. This keeps the module NumPy-only.

X01 = 2.404825557695773       # first zero of J0
X11 = 3.831705970207512       # first zero of J1

_BESSEL_MAX_ARG = 8.0         # series is reliable well past any cavity use
_BESSEL_TERMS = 40


def _besselj01(x):
    """
    J0(x) and J1(x) by ascending series, for 0 <= x <= _BESSEL_MAX_ARG.

    J0 = sum_m (-1)^m (x/2)^(2m) / (m!)^2
    J1 = sum_m (-1)^m (x/2)^(2m+1) / (m! (m+1)!)

    Returns (J0, J1) as floats or arrays, matching the input.
    """
    x = np.asarray(x, dtype=float)
    if np.any(np.abs(x) > _BESSEL_MAX_ARG):
        raise ValueError(
            f"Bessel argument exceeds {_BESSEL_MAX_ARG}; this series is only "
            "used for arguments bounded by k*a inside a cavity. Check units: "
            "r_s and the cavity radius must be in the same unit."
        )
    half = x / 2.0
    half_sq = half ** 2

    j0 = np.zeros_like(x)
    term = np.ones_like(x)                 # m = 0 term of J0
    for m in range(_BESSEL_TERMS):
        if m:
            term = -term * half_sq / (m * m)
        j0 = j0 + term

    j1 = np.zeros_like(x)
    term = half.copy()                     # m = 0 term of J1
    for m in range(_BESSEL_TERMS):
        if m:
            term = -term * half_sq / (m * (m + 1))
        j1 = j1 + term

    return j0, j1


def _mode_wavenumber(cavity):
    """Transverse wavenumber k of the cavity mode, in 1/(length unit of a)."""
    if cavity.mode == Cavity.Mode.TM010:
        return X01 / cavity.a
    if cavity.mode == Cavity.Mode.TM110:
        return X11 / cavity.a
    return np.sqrt(5.0) * np.pi / cavity.a          # TM210, square


def _require_tm010(cavity, what):
    if cavity.mode != Cavity.Mode.TM010:
        raise NotImplementedError(
            f"{what} is implemented for TM010 only. The TM110/TM210 magnetic "
            "analogue follows the same structure with |H|^2 in place of "
            "|E|^2, but has not been derived or validated yet - see the "
            "notebook section 'Partial Insertion and the Effective Volume'."
        )


# ------------------------------------------------------------
# EFFECTIVE VOLUME
# ------------------------------------------------------------

def radial_form_factor(cavity, r_s):
    """
    Transverse form factor F = J0(k r_s)^2 + J1(k r_s)^2  (TM010).

    This is V'_s divided by the geometric inserted volume: the correction
    for the cavity field varying ACROSS the rod. It equals 1 for a
    vanishingly thin rod and falls as 1 - (k r_s)^2 / 4.
    """
    _require_tm010(cavity, "radial_form_factor")
    k = _mode_wavenumber(cavity)
    j0, j1 = _besselj01(k * np.asarray(r_s, dtype=float))
    return float(j0 ** 2 + j1 ** 2) if np.isscalar(r_s) else (j0 ** 2 + j1 ** 2)


def effective_volume(cavity, r_s, h):
    """
    Effective volume V'_s of a rod of radius r_s inserted axially to depth h.

        V'_s(h) = (1/|E0|^2) Int_Vs |E_c|^2 dV
                = pi r_s^2 h [ J0(k r_s)^2 + J1(k r_s)^2 ]

    TM010 has no z dependence, so V'_s is EXACTLY linear in h.

    Parameters
    ----------
    cavity : Cavity (mode must be TM010)
    r_s : float          rod radius, same unit as cavity.a
    h : float or array   insertion depth, same unit as cavity.h

    Returns
    -------
    V'_s in (length unit)^3.
    """
    _require_tm010(cavity, "effective_volume")
    h = np.asarray(h, dtype=float)
    if np.any(h < 0) or np.any(h > cavity.h * (1.0 + 1e-9)):
        raise ValueError("insertion depth h must lie between 0 and the cavity height.")
    return np.pi * r_s ** 2 * h * radial_form_factor(cavity, r_s)


def effective_volume_annulus(cavity, r_inner, r_outer, h):
    """
    Effective volume of an annular shell (the quartz tube wall) inserted
    to depth h:

        V'_s = pi h [ r_o^2 (J0(k r_o)^2 + J1(k r_o)^2)
                      - r_i^2 (J0(k r_i)^2 + J1(k r_i)^2) ]

    Used to PREDICT the holder's own depth series, as a sanity check on
    the measured one. Reduces to effective_volume() when r_inner -> 0.
    """
    _require_tm010(cavity, "effective_volume_annulus")
    if not 0.0 <= r_inner < r_outer:
        raise ValueError("require 0 <= r_inner < r_outer.")
    h = np.asarray(h, dtype=float)
    k = _mode_wavenumber(cavity)
    j0o, j1o = _besselj01(k * r_outer)
    j0i, j1i = _besselj01(k * r_inner)
    return np.pi * h * (r_outer ** 2 * (j0o ** 2 + j1o ** 2)
                        - r_inner ** 2 * (j0i ** 2 + j1i ** 2))


def effective_filling(cavity, r_s, h):
    """Effective volume fraction v = V'_s / V_cavity (dimensionless)."""
    return effective_volume(cavity, r_s, h) / cavity.get_volume()


# ------------------------------------------------------------
# HOLDER CORRECTION  (Orloff eqs 12-13)
# ------------------------------------------------------------

def correct_for_holder(on_holder, holder, empty_cavity, mode="perturbation"):
    """
    Remove a sample holder that MOVES WITH THE SAMPLE.

    When the quartz tube is withdrawn together with the sample, the tube's
    own perturbation changes with depth and cannot be removed by a single
    fixed empty-tube reference.

    Two routes are available. They agree to first order and differ at
    second order in the perturbation; for this project's geometry that
    difference is about 0.7 % in eps' (see below), so the choice is not
    purely cosmetic.

    mode = "perturbation"  (DEFAULT)
        Subtract the holder's own shift at the same depth:

            x_sample = (f_ch - f_chs)/f_c
            y_sample = 1/Q_chs - 1/Q_ch

        This is exact to first order in Bethe-Schwinger perturbation
        theory, because the numerator integral is additive over disjoint
        regions and the denominator is common to both. It is therefore
        CONSISTENT WITH THE REST OF THIS MODULE, every formula in which is
        first-order perturbation theory. Default for that reason.

    mode = "lumped"
        Orloff's eqs (12) and (13), from modelling cavity, holder and
        sample as parallel admittances on one resonance (his App. B/C):

            1/f_cs^2 = 1/f_chs^2 - 1/f_ch^2 + 1/f_c^2
            1/Q_cs   = 1/Q_chs   - 1/Q_ch   + 1/Q_c

        The 2*pi cancels, so eq. 12 holds for f as written.

    Why they differ. The lumped model makes 1/f^2 additive; perturbation
    theory makes the fractional shift df/f additive. Those agree only to
    first order, the mismatch scaling as x_holder * x_sample with an
    O(1) coefficient (about 3 in practice). On a
    noise-free simulation of this cavity with a quartz tube and a
    tan d = 0.1 powder, the recovered eps' was

        perturbation : 3.40000   (true 3.40000, error 0.000 %)
        lumped       : 3.42388   (error +0.702 %)

    with x_holder = 2.8e-3 and x_sample = 6.2e-3, whose product is 0.28 %
    of x_sample - exactly the observed order. eps'' is unaffected
    (0.000 % either way), because 1/Q is additive in both models.

    Orloff's samples perturbed far less than ours, so the distinction did
    not arise for him. Both are offered here; compare them on real data.

    Assumption (A-H1): the holder and sample contributions are additive.
    In this coaxial geometry the quartz/powder interface is radial while
    E_z is tangential to it and therefore continuous, so both regions see
    the same unperturbed field and their contributions add by
    construction.

    Parameters
    ----------
    on_holder : Measurement      cavity + sample on holder, at this depth
    holder : Measurement         cavity + holder alone, at this SAME depth
    empty_cavity : Measurement   empty cavity (depth-independent)
    mode : {"perturbation", "lumped"}

    Returns
    -------
    Measurement for the cavity with the sample and no holder.
    """
    if mode not in ("perturbation", "lumped"):
        raise ValueError("mode must be 'perturbation' or 'lumped'.")

    f_c = empty_cavity.f_res
    inv_q_c = 1.0 / empty_cavity.quality_factor()

    inv_q = (1.0 / on_holder.quality_factor()
             - 1.0 / holder.quality_factor()
             + inv_q_c)
    if inv_q <= 0.0:
        raise ValueError(
            "Holder correction gave a non-physical Q (1/Q <= 0). This means "
            "the sample-on-holder measurement is LESS lossy than the holder "
            "alone - check that the two come from the same insertion depth.")

    if mode == "perturbation":
        x_sample = (holder.f_res - on_holder.f_res) / f_c
        f_cs = f_c * (1.0 - x_sample)
    else:
        inv_f2 = (1.0 / on_holder.f_res ** 2
                  - 1.0 / holder.f_res ** 2
                  + 1.0 / f_c ** 2)
        if inv_f2 <= 0.0:
            raise ValueError(
                "Holder correction gave a non-physical frequency (1/f^2 <= 0). "
                "Check that the holder and sample measurements are from the "
                "same insertion depth and are not swapped.")
        f_cs = 1.0 / np.sqrt(inv_f2)

    if f_cs <= 0.0:
        raise ValueError("Holder correction gave a non-physical frequency.")

    out = Measurement(f_cs, f_cs * inv_q, s11_min_db=on_holder.s11_min_db)
    for attr in ("Q0_unloaded", "beta_coupling", "fit_rms"):
        if hasattr(on_holder, attr):
            setattr(out, attr, getattr(on_holder, attr))
    return out


# ------------------------------------------------------------
# DEPTH SERIES
# ------------------------------------------------------------

class InsertionPoint:
    """
    One insertion depth of a partial-insertion series.

    Parameters
    ----------
    depth : float
        Insertion depth h, same unit as cavity.h.
    sample : Measurement
        Cavity with the sample (on the holder, if there is one) at this depth.
    holder : Measurement or None
        Cavity with the holder ALONE at this same depth. None means there is
        no holder, or the holder does not move (then the fixed reference in
        InsertionSeries is used instead).
    """

    def __init__(self, depth, sample, holder=None):
        self.depth = float(depth)
        self.sample = sample
        self.holder = holder


class PartialInsertionResult:
    """
    Outcome of a linear regression over a partial-insertion depth series.

    Attributes
    ----------
    slope_r, slope_i : float        fitted slopes  C(eps'-1)  and  C eps''
    slope_r_err, slope_i_err : float   1-sigma standard errors on the slopes
    beta_r, beta_i : float          non-uniform-field intercepts (= -intercept)
    beta_r_err, beta_i_err : float
    eps_real, eps_imag : float      eps' and eps''  (eps'' positive for loss)
    tan_delta : float
    material : Material             eps = eps' - j eps''
    v, x, y : arrays                every point of the series
    used : boolean array            which points fell inside the fit window
    residual_r, residual_i : arrays residuals of the two fits, for the used points
    r2_r, r2_i : float
    single_point_eps_real, single_point_eps_imag : arrays
        The conventional result computed separately at each depth. If these
        drift with depth, beta is non-zero and this correction is needed; if
        they are flat, the simple formula was already adequate.
    """

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def summary(self):
        """Human-readable one-block summary."""
        lines = [
            f"  points used        : {int(self.used.sum())} of {len(self.v)}"
            f"   (window {self.window[0] * 100:.0f}-{self.window[1] * 100:.0f} % fill)",
            f"  slope_r            : {self.slope_r:.6g} +/- {self.slope_r_err:.2g}",
            f"  slope_i            : {self.slope_i:.6g} +/- {self.slope_i_err:.2g}",
            f"  beta_r (intercept) : {self.beta_r:.4g} +/- {self.beta_r_err:.2g}",
            f"  beta_i (intercept) : {self.beta_i:.4g} +/- {self.beta_i_err:.2g}",
            f"  R^2                : {self.r2_r:.6f} (freq), {self.r2_i:.6f} (loss)",
            f"  eps'               : {self.eps_real:.5g} +/- {self.eps_real_err:.2g}",
            f"  eps''              : {self.eps_imag:.5g} +/- {self.eps_imag_err:.2g}",
            f"  tan delta          : {self.tan_delta:.5g}",
        ]
        return "\n".join(lines)

    def to_dict(self):
        """JSON-serialisable form."""
        return {
            "window_fill_fraction": list(self.window),
            "n_points_total": int(len(self.v)),
            "n_points_used": int(self.used.sum()),
            "depth": [float(d) for d in self.depth],
            "v_effective": [float(t) for t in self.v],
            "x_freq_shift": [float(t) for t in self.x],
            "y_inv_q_shift": [float(t) for t in self.y],
            "used": [bool(t) for t in self.used],
            "slope_r": float(self.slope_r), "slope_r_err": float(self.slope_r_err),
            "slope_i": float(self.slope_i), "slope_i_err": float(self.slope_i_err),
            "beta_r": float(self.beta_r), "beta_r_err": float(self.beta_r_err),
            "beta_i": float(self.beta_i), "beta_i_err": float(self.beta_i_err),
            "r2_r": float(self.r2_r), "r2_i": float(self.r2_i),
            "eps_real": float(self.eps_real), "eps_real_err": float(self.eps_real_err),
            "eps_imag": float(self.eps_imag), "eps_imag_err": float(self.eps_imag_err),
            "tan_delta": float(self.tan_delta),
            "single_point_eps_real": [float(t) for t in self.single_point_eps_real],
            "single_point_eps_imag": [float(t) for t in self.single_point_eps_imag],
        }


def _linear_fit(xv, yv):
    """
    Ordinary least squares y = m x + c.

    Returns (m, c, m_err, c_err, r2, residuals). Standard errors come from
    the residual scatter, so they describe the spread of the points about
    the line and NOT any systematic error in v.
    """
    xv = np.asarray(xv, dtype=float)
    yv = np.asarray(yv, dtype=float)
    n = len(xv)
    if n < 3:
        raise ValueError("need at least 3 points inside the fit window.")

    x_mean, y_mean = xv.mean(), yv.mean()
    sxx = np.sum((xv - x_mean) ** 2)
    if sxx == 0.0:
        raise ValueError("all insertion depths are identical - cannot fit a slope.")
    m = np.sum((xv - x_mean) * (yv - y_mean)) / sxx
    c = y_mean - m * x_mean

    residuals = yv - (m * xv + c)
    dof = n - 2
    s_sq = np.sum(residuals ** 2) / dof if dof > 0 else 0.0
    m_err = np.sqrt(s_sq / sxx)
    c_err = np.sqrt(s_sq * (1.0 / n + x_mean ** 2 / sxx))

    ss_tot = np.sum((yv - y_mean) ** 2)
    r2 = 1.0 - np.sum(residuals ** 2) / ss_tot if ss_tot > 0 else float("nan")
    return m, c, m_err, c_err, r2, residuals


class InsertionSeries:
    """
    A partial-insertion depth series for one sample.

    Parameters
    ----------
    cavity : Cavity (TM010)
    r_sample : float
        Rod / powder-column radius, same unit as cavity.a.
    reference : Measurement
        The EMPTY CAVITY (f_c, Q_c). With a moving holder this is the
        depth-independent third measurement needed by Orloff eq. 12-13.
        With no holder it is simply the unperturbed reference.
    points : list of InsertionPoint
    holder_mode : {"perturbation", "lumped"}
        How a per-depth holder measurement is removed; see
        correct_for_holder(). Default "perturbation", which is the one
        consistent with the first-order theory used everywhere else here.
    """

    def __init__(self, cavity, r_sample, reference, points=None,
                 holder_mode="perturbation"):
        _require_tm010(cavity, "InsertionSeries")
        self.cavity = cavity
        self.r_sample = float(r_sample)
        self.reference = reference
        self.points = list(points) if points else []
        self.holder_mode = holder_mode

    def add(self, depth, sample, holder=None):
        """Append one insertion depth."""
        self.points.append(InsertionPoint(depth, sample, holder))
        return self

    def sorted_points(self):
        return sorted(self.points, key=lambda p: p.depth)

    def corrected_measurements(self):
        """
        The sample's own (f_cs, Q_cs) at each depth, holder removed where a
        per-depth holder measurement was supplied.
        """
        out = []
        for p in self.sorted_points():
            if p.holder is None:
                out.append(p.sample)
            else:
                out.append(correct_for_holder(p.sample, p.holder, self.reference,
                                              mode=self.holder_mode))
        return out

    def observables(self):
        """
        Return (depth, v, x, y) arrays.

            v = V'_s/V_cavity          effective volume fraction
            x = (f_c - f_cs)/f_c       normalised frequency shift
            y = 1/Q_cs - 1/Q_c         change in inverse Q
        """
        pts = self.sorted_points()
        depth = np.array([p.depth for p in pts], dtype=float)
        meas = self.corrected_measurements()

        v = effective_filling(self.cavity, self.r_sample, depth)
        f_c = self.reference.f_res
        inv_q_c = 1.0 / self.reference.quality_factor()
        x = np.array([(f_c - m.f_res) / f_c for m in meas])
        y = np.array([1.0 / m.quality_factor() - inv_q_c for m in meas])
        return depth, v, x, y

    def fit(self, window=(0.30, 0.70)):
        """
        Regress x and y/2 against v and convert the slopes to permittivity.

        Parameters
        ----------
        window : (lo, hi)
            Fit window as a fraction of the cavity height. Orloff's default
            is 30-70 %: the data bend away from the line near 0 % and 100 %.
            This is a STARTING POINT, not a result - shrink it until the
            residuals are structureless. Pass (0.0, 1.0) to use everything.

        Returns
        -------
        PartialInsertionResult
        """
        depth, v, x, y = self.observables()
        frac = depth / self.cavity.h
        used = (frac >= window[0]) & (frac <= window[1])
        if used.sum() < 3:
            raise ValueError(
                f"only {int(used.sum())} of {len(depth)} depths fall inside the "
                f"{window[0] * 100:.0f}-{window[1] * 100:.0f} % window; widen it "
                "or take more points."
            )

        C = self.cavity.get_coefficient()
        m_r, c_r, m_r_err, c_r_err, r2_r, res_r = _linear_fit(v[used], x[used])
        m_i, c_i, m_i_err, c_i_err, r2_i, res_i = _linear_fit(v[used], y[used] / 2.0)

        eps_real = m_r / C + 1.0
        eps_imag = m_i / C

        # Conventional single-point result at every depth, for the diagnostic.
        with np.errstate(divide="ignore", invalid="ignore"):
            sp_real = x / (C * v) + 1.0
            sp_imag = y / (2.0 * C * v)

        return PartialInsertionResult(
            window=tuple(window),
            depth=depth, v=v, x=x, y=y, used=used,
            slope_r=m_r, slope_i=m_i,
            slope_r_err=m_r_err, slope_i_err=m_i_err,
            beta_r=-c_r, beta_i=-c_i,
            beta_r_err=c_r_err, beta_i_err=c_i_err,
            r2_r=r2_r, r2_i=r2_i,
            residual_r=res_r, residual_i=res_i,
            eps_real=eps_real, eps_imag=eps_imag,
            eps_real_err=m_r_err / C, eps_imag_err=m_i_err / C,
            tan_delta=eps_imag / eps_real,
            material=Material(eps=complex(eps_real, -eps_imag),
                              loss_tangent=eps_imag / eps_real),
            single_point_eps_real=sp_real, single_point_eps_imag=sp_imag,
        )


def fit_partial_insertion(cavity, r_sample, reference, points,
                          window=(0.30, 0.70), holder_mode="perturbation"):
    """Convenience wrapper: build an InsertionSeries and fit it in one call."""
    return InsertionSeries(cavity, r_sample, reference, points,
                           holder_mode=holder_mode).fit(window)


# ------------------------------------------------------------
# COMPLEX ONE-PORT RESONATOR FIT
# ------------------------------------------------------------

def baseline_level(freqs, s11, edge_fraction=0.01):
    """
    Mean |S11| over the off-resonance edges of the sweep.

    An ideal one-port resonator reflects everything off resonance, so
    |S11| -> 1 there. A real measurement does not: cable loss, an
    imperfect or drifting calibration and an imperfect short all leave
    the baseline below (or above) unity.

    Use this as a CALIBRATION CHECK. find_resonance() references the dip
    depth to an implicit baseline of exactly 1, which a good calibration
    delivers; a value materially away from 1.00 means the calibration is
    not holding at the reference plane. For scale: a baseline of 0.98
    biases the half-power Q low by about 4 %, and 0.90 by about 22 %.
    See find_resonance(normalise=True) if it cannot be re-calibrated.

    Parameters
    ----------
    edge_fraction : float
        Fraction of the sweep taken from EACH end as "off resonance".
        The resonance must be comfortably inside the sweep for this to
        mean anything - analyse_trace() in data_analysis.py already warns
        when it is not.
    """
    s11 = np.asarray(s11)
    edge = max(3, int(len(s11) * edge_fraction))
    return float(np.mean(np.abs(np.concatenate([s11[:edge], s11[-edge:]]))))


def detuning(freqs, f0):
    """
    Exact fractional detuning  delta = (f^2 - f0^2)/(f*f0) = f/f0 - f0/f.

    Near resonance, with f = f0(1+u), this expands as

        delta = 2u - u^2 + u^3 - ...

    so the familiar 2*(f - f0)/f0 is only its leading term. The u^2
    correction is negligible for a high-Q resonance but NOT for the low-Q
    ones a lossy sample produces, where the sweep has to span many
    bandwidths. The exact form is used throughout.
    """
    freqs = np.asarray(freqs, dtype=float)
    return (freqs ** 2 - f0 ** 2) / (freqs * f0)


def s11_model(freqs, f0, Q0, beta):
    """
    One-port reflection from a resonant cavity fed by a coupling loop:

        S11 = (beta - 1 - j Q0 delta) / (beta + 1 + j Q0 delta)

    This follows from Gamma = (Z - Z0)/(Z + Z0) with the cavity presenting
    the normalised impedance Z = beta*Z0/(1 + j Q0 delta), which is the
    form derived in Derivation_towards_coupling_coeff_wrt_Lloop.

    Parameters
    ----------
    f0 : float      resonant frequency, Hz
    Q0 : float      UNLOADED quality factor
    beta : float    coupling coefficient, Q0/Q_external

    Useful identities (all exact for this model, verified symbolically):
        S11(f0)        = (beta - 1)/(beta + 1)
        |S11|_min      = |beta - 1|/(beta + 1)
        1 - |S11|^2    = 4 beta / ((1+beta)^2 + (Q0 delta)^2)
        loaded Q       QL = Q0/(1 + beta)
        half-absorbed-power points at QL*delta = +-1

    The last two mean find_resonance() measures QL with no model error:
    its level 10log10((1+10^(Smin/10))/2) is exactly where the absorbed
    power above halves. beta = 1 is critical coupling (deepest dip);
    beta < 1 undercoupled, beta > 1 overcoupled. A magnitude measurement
    cannot tell those apart, since beta and 1/beta give the same |S11|.
    """
    jQd = 1j * Q0 * detuning(freqs, f0)
    return (beta - 1.0 - jQd) / (beta + 1.0 + jQd)


def fit_resonance(freqs, s11, initial=None, scale=False):
    """
    Fit the whole S11 trace to the one-port cavity model above.

    Orloff fits a damped driven harmonic oscillator to complex S21 - a
    transmission PEAK, two-port. This setup measures S11, a reflection
    DIP, for which that form does not apply; the model used here is the
    one-port equivalent from the project's own loop-antenna derivation.

    Only THREE parameters are fitted: f0, Q0 and beta. Cable delay and
    reference-plane rotation are not modelled, because the VNA
    calibration already places the reference plane at the end of the
    cables. (Pass scale=True to add a complex scale factor if a
    calibration is suspect - see below.)

    Why fit at all, when find_resonance() is already exact for this
    model? Two reasons, and the second is the important one.

    1. Noise. The fit uses every point rather than the few near the
       minimum. Measured on synthetic traces from this model (801
       points, sigma = 1e-3 on linear S11, 40 draws), the fractional
       error in D(1/Q) - which eps'' is proportional to - is:

           reference -> sample      dip      half-power      fit
           Q0 8000 -> 6000        -17 dB       1.1 %        0.10 %
           Q0 8000 -> 3000        -6.8 dB      0.4 %        0.04 %
           Q0 8000 -> 1200        -2.6 dB      0.6 %        0.06 %
           Q0 8000 ->  600        -1.3 dB      1.2 %        0.09 %

       So the half-power method is perfectly usable on calibrated data;
       the fit is about 10x better, from an already adequate baseline.

    2. The coupling assumption. find_resonance() gives the LOADED Q, and
       using it in D(1/Q) assumes the external Q is identical for the
       reference and the sample. In partial insertion the sample slides
       past the coupling loop, so that is the assumption most at risk.
       The fit separates Q0 from beta, so it does not need it at all:

           beta drift, ref -> sample    half-power eps'' error    fit
                    0 %                      0.0 %              0.02 %
                    2 %                      1.2 %              0.02 %
                    5 %                      3.0 %              0.02 %
                   10 %                      6.0 %              0.02 %

       The error tracks the drift roughly linearly. Recording
       .beta_coupling against insertion depth turns an assumption into
       a measurement.

    The coupling branch is ambiguous from magnitude alone (beta and
    1/beta give identical |S11|), so both branches are fitted from the
    measured dip depth and the lower-residual one is returned.

    Parameters
    ----------
    freqs : 1-D array, Hz
    s11 : 1-D complex array (linear)
    initial : Measurement or None
        Starting guess for f0 and the loaded Q. Defaults to find_resonance().
    scale : bool, default False
        Fit an extra complex scale factor A*exp(j*phi) multiplying the
        model. Leave False for calibrated data. Turn it on only if the
        residuals show a systematic offset or rotation, which would mean
        the calibration is not holding at the reference plane.

    Returns
    -------
    Measurement, with
        .f_res, .BW, .quality_factor()   the LOADED Q, QL = Q0/(1+beta),
                                         so it is interchangeable with
                                         find_resonance() everywhere else
        .Q0_unloaded                     the UNLOADED Q
        .beta_coupling                   the coupling coefficient
        .fit_rms                         RMS residual in linear S11 units
        .fit_scale                       the complex scale factor (1 if scale=False)

    Q0 is the quantity perturbation theory actually wants, so having it
    removes the need to assume the coupling is unchanged between the
    reference and the sample (assumption 4 in the module docstring).
    Tracking .beta_coupling across insertion depths tests that assumption
    directly.

    Raises
    ------
    ImportError if SciPy is not installed (it is needed only here).
    RuntimeError if neither branch converges or the result is non-physical.
    """
    try:
        from scipy.optimize import least_squares
    except ImportError as err:                          # pragma: no cover
        raise ImportError(
            "fit_resonance() needs SciPy (pip install scipy). The rest of "
            "CPM.py is NumPy-only; install SciPy only if you want this fit."
        ) from err

    freqs = np.asarray(freqs, dtype=float)
    s11 = np.asarray(s11, dtype=complex)
    if len(freqs) < 10:
        raise RuntimeError("need at least 10 points to fit a resonance.")

    if initial is None:
        initial = find_resonance(freqs, s11)
    f0_0 = initial.f_res
    ql_0 = initial.quality_factor()

    # beta from the measured dip depth: |S11|min = |beta-1|/(beta+1).
    # Both branches satisfy that, so try each.
    gamma_min = float(np.min(np.abs(s11)))
    gamma_min = min(gamma_min, 0.99)
    beta_under = (1.0 - gamma_min) / (1.0 + gamma_min)      # beta < 1
    beta_over = (1.0 + gamma_min) / (1.0 - gamma_min)       # beta > 1

    f_mid = 0.5 * (freqs[0] + freqs[-1])
    span = freqs[-1] - freqs[0]

    def unpack(p):
        f0 = f_mid + p[0] * span
        return f0, p[1], p[2]

    def residual(p):
        f0, Q0, beta = unpack(p)
        model = s11_model(freqs, f0, Q0, beta)
        if scale:
            model = model * (p[3] + 1j * p[4])
        r = model - s11
        return np.concatenate([r.real, r.imag])

    best = None
    for beta_0 in (beta_under, beta_over):
        p0 = [(f0_0 - f_mid) / span, ql_0 * (1.0 + beta_0), beta_0]
        lo = [-0.5, 1.0, 1e-6]
        hi = [0.5, 1e9, 1e6]
        if scale:
            p0 += [1.0, 0.0]
            lo += [-10.0, -10.0]
            hi += [10.0, 10.0]
        p0 = np.clip(np.array(p0, dtype=float),
                     np.array(lo) + 1e-12, np.array(hi) - 1e-12)
        try:
            out = least_squares(residual, p0, bounds=(lo, hi), method="trf",
                                x_scale="jac", max_nfev=20000,
                                xtol=1e-15, ftol=1e-15)
        except ValueError:                               # pragma: no cover
            continue
        if not out.success:
            continue
        if best is None or out.cost < best.cost:
            best = out

    if best is None:
        raise RuntimeError("one-port cavity fit did not converge on either "
                           "coupling branch.")

    f0_fit, q0_fit, beta_fit = unpack(best.x)
    if not (freqs[0] < f0_fit < freqs[-1]):
        raise RuntimeError("fitted resonance lies outside the sweep - widen the span.")
    if q0_fit <= 0 or beta_fit <= 0:
        raise RuntimeError("fit returned a non-physical Q0 or beta.")

    ql_fit = q0_fit / (1.0 + beta_fit)
    meas = Measurement(f0_fit, f0_fit / ql_fit,
                       s11_min_db=float(20.0 * np.log10(gamma_min)))
    meas.Q0_unloaded = q0_fit
    meas.beta_coupling = beta_fit
    meas.fit_rms = float(np.sqrt(np.mean(best.fun ** 2)))
    meas.fit_scale = complex(best.x[3], best.x[4]) if scale else 1.0 + 0j
    return meas


def compare_resonance_methods(freqs, s11):
    """
    Run every resonance-extraction method on one trace and report them all.

    Written to its own section of the results file so the methods can be
    compared on real data rather than assumed equivalent (Orloff does the
    same in his Appendix A / Table II, for his two-port case).

    Returns a dict with
        half_absorbed_power            the method in the handwritten notes
        half_absorbed_power_normalised the same, referenced to the measured
                                       off-resonance baseline instead of to
                                       an assumed |S11| = 1
        one_port_fit                   the complex resonator fit
        baseline                       the measured off-resonance |S11|
        difference                     fit vs each half-power variant

    If the fit fails (no SciPy, no convergence) its entry carries the
    error message instead of numbers and the half-power results stand.
    """
    def pack(m):
        return {"f0_Hz": m.f_res, "BW_Hz": m.BW, "Q": m.quality_factor(),
                "s11_min_dB": m.s11_min_db}

    hp = find_resonance(freqs, s11)
    out = {"baseline_abs_s11": baseline_level(freqs, s11),
           "half_absorbed_power": pack(hp)}

    try:
        out["half_absorbed_power_normalised"] = pack(
            find_resonance(freqs, s11, normalise=True))
    except ValueError as err:
        out["half_absorbed_power_normalised"] = {"error": str(err)}

    try:
        ft = fit_resonance(freqs, s11, initial=hp)
    except (ImportError, RuntimeError, ValueError) as err:
        out["one_port_fit"] = {"error": str(err)}
        out["difference"] = None
        return out

    out["one_port_fit"] = pack(ft)
    out["one_port_fit"]["Q0_unloaded"] = ft.Q0_unloaded
    out["one_port_fit"]["beta_coupling"] = ft.beta_coupling
    out["one_port_fit"]["fit_rms"] = ft.fit_rms
    out["difference"] = {}
    for key in ("half_absorbed_power", "half_absorbed_power_normalised"):
        ref = out[key]
        if "error" in ref:
            continue
        out["difference"][f"fit_vs_{key}"] = {
            "f0_ppm": (ft.f_res - ref["f0_Hz"]) / ref["f0_Hz"] * 1e6,
            "Q_percent": (ft.quality_factor() - ref["Q"]) / ref["Q"] * 100.0,
        }
    return out
