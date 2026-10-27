"""Analyse the three supplied simulation images using the user's CPM module.

Run: python simulation_analysis.py
Put CPM.py beside this script, or run:
    python simulation_analysis.py --cpm "path/to/CPM(1).py"
Dependencies: NumPy and the supplied CPM module. No S11 traces are fabricated.
Lengths are mm; image inputs below are GHz and dB. Complex values use e^(jwt).

IMPORTANT: cavity dimensions are NOT in the images. Defaults below use ideal
2.45 GHz air-cavity cross-sections and an ASSUMED TM210 height of 35 mm.
Replace these with the actual simulation dimensions before reporting results.
All annotated dielectric bandwidths are used AS SUPPLIED, assuming they are
half-absorbed-power widths as required by CPM, not fixed-dB widths.
Loss retrieval also assumes unchanged external coupling; no holder is removed.

Polder convention and linewidth:
https://2025.help.altair.com/2025/feko/topics/feko/user_guide/solver_solution_methods/media_anisotropic_feko_c.htm
FEKO's 255 G means 4*pi*Ms, not Ms. DeltaH=250 Oe is the field FWHM.
"""

import argparse
import importlib.util
from pathlib import Path
import numpy as np


# EDITABLE INPUTS ------------------------------------------------------------
C0 = 299792458.0
DESIGN_F = 2.45e9
TM010_RADIUS = 46.833
TM010_HEIGHT = 35.0
TM210_SIDE = 136.8
TM210_HEIGHT = 20.0                 # ASSUMED: set actual simulation height
TEFLON_RADIUS, TEFLON_HEIGHT = 5.0, 35.0
FERRITE_RADIUS, FERRITE_HEIGHT = 2.35, 20.0
IPA_RADIUS = 2.35

# (resonance GHz, annotated bandwidth GHz, S11 minimum dB)
# CavityS11TM010Pic.png: blue baseline; green is a different probe configuration.
EMPTY_SHORT = (2.45875, 0.00116242, -11.9313)
EMPTY_LONG = (2.46094, 0.00232951, -16.8727)
TEFLON = (2.4072, 0.0011398, -12.4297)
TEFLON_BASELINE = EMPTY_SHORT       # confirm Teflon uses the blue probe geometry

# IsopropanolPic.png: baseline is the green/long-probe configuration above.
# (insertion fraction, resonance GHz, annotated bandwidth GHz, minimum dB)
IPA = [
    (1/8, 2.4556, 0.00614393, -4.25707),
    (2/8, 2.4508, 0.0134623, -1.86255),
    (3/8, 2.4472, 0.0218218, -1.20829),
    (4/8, 2.4424, 0.0305418, -0.896031),
]

# CavityS11TM210_Polder.png: no bandwidth annotations are supplied.
# (resonance GHz, bandwidth GHz or None, S11 minimum dB)
MAG_EMPTY = (2.4496, 0.0014009, -12.4939)
MAG = {
    5000: ((2.4484, 0.0013735, -10.8964), (2.4485, 0.00110181, -12.2705)),
    2500: ((2.4484, 0.00137283, -10.3795), (2.44825, 0.00114795, -9.99721)),
}
# First trace is labelled '+' purely by convention; the image does not establish
# physical handedness. Swapping traces reverses the inferred off-diagonal term.
SATURATION_4PI_MS_G = 255.0
LINEWIDTH_OE = 250.0
GAMMA_HZ_PER_OE = 2.8e6             # approximate electron gyromagnetic ratio


def load_cpm(path):
    spec = importlib.util.spec_from_file_location("CPM", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def polder(f_hz, bias_oe):
    """Saturated uniform-bias relative tensor, bias along +z.

    mu_r = [[mu, j*kappa, 0], [-j*kappa, mu, 0], [0, 0, 1]].
    H is treated as FEKO's specified bias; no extra demagnetisation correction.
    w0 -> w0+j*gamma*DeltaH/2 gives passive mu=mu'-j*mu''.
    Frequency units cancel the common factors of 2*pi.
    """
    f0 = GAMMA_HZ_PER_OE * (bias_oe + 1j * LINEWIDTH_OE / 2)
    fm = GAMMA_HZ_PER_OE * SATURATION_4PI_MS_G
    denominator = f0**2 - f_hz**2
    return 1 + f0 * fm / denominator, f_hz * fm / denominator


def main(CPM):
    def measurement(row):
        f, bw, dip = row
        if bw is None:
            raise ValueError("A measured loss term requires a bandwidth.")
        return CPM.Measurement(f * 1e9, bw * 1e9, dip)

    def show_complex(z):
        return f"{z.real:.6g} {z.imag:+.6g}j"

    electric = CPM.Cavity(CPM.Cavity.Shape.CYLINDRICAL,
                          CPM.Cavity.Mode.TM010, TM010_RADIUS, TM010_HEIGHT)
    magnetic = CPM.Cavity(CPM.Cavity.Shape.RECTANGULAR,
                          CPM.Cavity.Mode.TM210, TM210_SIDE, TM210_HEIGHT)
    print("PROVISIONAL GEOMETRY -- replace with actual simulation dimensions")
    print(f"TM010 radius={electric.a:.5f}, height={electric.h:g} mm")
    print(f"TM210 side={magnetic.a:.5f}, height={magnetic.h:g} mm (assumed)")
    print("Annotated dielectric BWs assumed to be half-absorbed-power widths.")

    print("\nTEFLON: full insertion, geometric sample volume")
    teflon = CPM.Sample(np.pi * TEFLON_RADIUS**2 * TEFLON_HEIGHT)
    ref, loaded = measurement(TEFLON_BASELINE), measurement(TEFLON)
    mat = CPM.TM010_measure(loaded, ref, teflon, electric)
    print(f"Q reference={ref.quality_factor():.3f}, sample={loaded.quality_factor():.3f}")
    print(f"eps_r = {show_complex(mat.eps)}, tan(delta)={mat.loss_tangent:.6g}")
    # Optional finite-radius correction using CPM's existing effective volume.
    effective = CPM.Sample(CPM.effective_volume(electric, TEFLON_RADIUS, TEFLON_HEIGHT))
    corrected = CPM.TM010_measure(loaded, ref, effective, electric)
    print(f"With radial field correction: eps_r = {show_complex(corrected.eps)}")

    print("\nISOPROPANOL: partial insertion, long-probe baseline")
    series = CPM.InsertionSeries(electric, IPA_RADIUS, measurement(EMPTY_LONG))
    print(" h/height    h (mm)     f (GHz)       Q_loaded")
    for fraction, f, bw, dip in IPA:
        m = measurement((f, bw, dip))
        series.add(fraction * electric.h, m)
        print(f" {fraction:7.3f}   {fraction*electric.h:7.3f}    {f:8.5f}    {m.quality_factor():9.3f}")
    # Three points minimum: the default CPM 30-70% window contains only two.
    # Show all four and a three-point sensitivity check, without forcing zero intercept.
    for window in ((1/8, 4/8), (2/8, 4/8)):
        fit = series.fit(window=window)
        print(f"\nFit over {window[0]:.3f}-{window[1]:.3f} of height:")
        print(fit.summary())
    print("Fit errors describe scatter only; four rounded points cannot establish")
    print("a constant-depolarisation window or validate the bandwidth definition.")

    print("\nNiZn: TM210, exact complex CPM inversion if bandwidths are entered")
    ferrite = CPM.Sample(np.pi * FERRITE_RADIUS**2 * FERRITE_HEIGHT)
    # BW=0 is used ONLY as a frequency-reference container, never as measured Q.
    frequency_ref = CPM.Measurement(MAG_EMPTY[0] * 1e9, 0.0, MAG_EMPTY[2])
    pert = CPM.Perturbation(magnetic, ferrite, frequency_ref)
    for bias, rows in MAG.items():
        mu, kappa = polder(frequency_ref.f_res, bias)
        print(f"\nH0={bias} Oe, theory evaluated at {MAG_EMPTY[0]:.5f} GHz:")
        print(f"  mu={show_complex(mu)}, kappa={show_complex(kappa)}")
        tensor = np.array([[mu, 1j*kappa, 0], [-1j*kappa, mu, 0], [0, 0, 1]])
        print("  Relative Polder tensor (+z bias):")
        print(np.array2string(tensor, precision=6, suppress_small=True))
        apparent = []
        have_bw = MAG_EMPTY[1] is not None and all(row[1] is not None for row in rows)
        for sense, row in zip((+1, -1), rows):
            mu_eff = mu + sense * kappa
            response = (mu_eff - 1) / (mu_eff + 1)
            predicted_f = frequency_ref.f_res * (1 - pert.get_scale() * response.real)
            if have_bw:
                value = CPM.TM210_measure(measurement(row), measurement(MAG_EMPTY),
                                          ferrite, magnetic)["exact"]
            else:
                # Frequency-only (small-loss) inversion, not a complex measurement.
                point = CPM.Measurement(row[0] * 1e9, 0.0)
                F_real = pert.frequency_shift(point) / pert.get_scale()
                value = (1 + F_real) / (1 - F_real)
            apparent.append(value)
            print(f"  sense {sense:+d}: observed f={row[0]:.6f}, "
                  f"predicted f={predicted_f/1e9:.6f} GHz; "
                  f"inferred mu_eff={value:.6g}")
        separated = CPM.separate_circular(*apparent)
        label = "Complex retrieval" if have_bw else "Frequency-only, small-loss estimates"
        print(f"  {label}: mu={separated.mu:.6g}, kappa={separated.alpha:.6g}")
        if not have_bw:
            print("  Measured tensor loss terms: unavailable (no bandwidths).")
    print("\nChecks: '+' is arbitrary; compare both handedness assignments.")
    print("CPM's TM210 coefficient 16/5 is retained; its source flags it for verification.")
    print("Theory assumes saturated, uniform magnetisation at BOTH biases;")
    print("an unsaturated material needs additional magnetisation information.")
    print("A discrepancy may also reflect sample position, noncircular local fields,")
    print("electric loading or changed coupling, rather than the intrinsic tensor.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpm", type=Path, default=Path(__file__).with_name("CPM.py"))
    args = parser.parse_args()
    main(load_cpm(args.cpm))
