"""Loss required for a specified Q decrease, using CPM.measure().

Place beside CPM.py and run: python q_shift_loss.py
All dimensions are metres. Complex properties use real - j*loss.
Reference Q values may be loaded or unloaded, but use one convention
consistently; loaded-Q differencing assumes unchanged external coupling.
"""

from math import pi
import CPM

# Adjustable inputs ---------------------------------------------------------
F_REFERENCE = 2.45e9
Q_REFERENCE_TM010 = 1640.0
Q_REFERENCE_TM110 = 2490.0
Q_REFERENCE_TM210 = 1800.0
Q_DROP = 0.05                     # (Q_shifted-Q_reference)/Q_reference = -0.05
SAMPLE_RADIUS = 5e-3
FORM_FACTOR = 0.984               # retained from your calculation, for each mode

# (mode, cavity shape, radius/side [m], full insertion height [m], reference Q)
CONFIGURATIONS = [
    (CPM.Cavity.Mode.TM010, CPM.Cavity.Shape.CYLINDRICAL,
     46.8331e-3, 35e-3, Q_REFERENCE_TM010),
    (CPM.Cavity.Mode.TM110, CPM.Cavity.Shape.CYLINDRICAL,
     74.622e-3, 30e-3, Q_REFERENCE_TM110),
    (CPM.Cavity.Mode.TM210, CPM.Cavity.Shape.RECTANGULAR,
     136.802e-3, 20e-3, Q_REFERENCE_TM210),
]


def main():
    if not 0 < Q_DROP < 1:
        raise ValueError("Q_DROP must be between 0 and 1.")
    print(f"Target Q change: {-100 * Q_DROP:.3f}% at fixed frequency")
    print(f"Corresponding BW change: {100 * Q_DROP / (1-Q_DROP):.3f}%")
    for mode, shape, size, height, q_ref in CONFIGURATIONS:
        if q_ref <= 0:
            raise ValueError("Reference Q values must be positive.")
        cavity = CPM.Cavity(shape, mode, size, height)
        sample = CPM.Sample(FORM_FACTOR * pi * SAMPLE_RADIUS**2 * height)
        q_shifted = q_ref * (1 - Q_DROP)
        reference = CPM.Measurement(F_REFERENCE, F_REFERENCE / q_ref)
        shifted = CPM.Measurement(F_REFERENCE, F_REFERENCE / q_shifted)
        result = CPM.measure(shifted, reference, sample, cavity)

        print(f"\n{mode.value}: Q {q_ref:g} -> {q_shifted:g}")
        print(f"  BW {reference.BW:.3f} -> {shifted.BW:.3f} Hz"
              f" (increase {shifted.BW-reference.BW:.3f} Hz)")
        if mode == CPM.Cavity.Mode.TM010:
            print(f"  eps' = {result.eps.real:.9g}; eps'' = {-result.eps.imag:.9g}")
        else:
            # 'notes' enforces mu'=1 at zero frequency shift to first order
            # in loss. Exact complex inversion introduces a second-order
            # departure of mu' from 1; both are supplied by CPM.measure().
            print(f"  Unit-real, small-loss estimate: mu'' = {-result['notes'].imag:.9g}")
            print(f"  Exact inversion: mu' = {result['exact'].real:.9g};"
                  f" mu'' = {-result['exact'].imag:.9g}")


if __name__ == "__main__":
    main()
