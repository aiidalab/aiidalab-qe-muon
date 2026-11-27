"""Quantities plotted in the convergence analysis of the polarization (no widget import, so it can be tested alone)."""

from __future__ import annotations

import numpy as np

# Below this |P_r| the relative difference is meaningless (P_r crosses zero): the point is not plotted.
REFERENCE_TOLERANCE = 1e-12

# The toggle buttons of the convergence view: value -> y-axis label.
CONVERGENCE_QUANTITIES = {
    "P": "P(t)",
    "deltaP": "ΔP(t)",
    "deltaP_rel": "100·ΔP(t)/Pᵣ(t) (%)",
}


def format_hdim(value) -> str:
    """max_hdim as HTML: ``10<sup>6</sup>`` for a power of ten, ``5&middot;10<sup>5</sup>`` for 5e5, else the integer."""
    value = int(value)
    exponent = int(np.floor(np.log10(value)))
    mantissa = value / 10**exponent
    if np.isclose(mantissa, 1):
        return f"10<sup>{exponent}</sup>"
    if np.isclose(mantissa, round(mantissa)):
        return f"{int(round(mantissa))}&middot;10<sup>{exponent}</sup>"
    return str(value)


def convergence_quantity(polarization, reference, kind: str = "P") -> np.ndarray:
    """Return the quantity to plot to check the convergence of ``polarization`` against ``reference``.

    :param polarization: P(t), computed with the max_hdim under test.
    :param reference: Pᵣ(t), computed with the largest max_hdim.
    :param kind: ``"P"``: P(t); ``"deltaP"``: ΔP(t) = P(t) - Pᵣ(t);
        ``"deltaP_rel"``: 100·ΔP(t)/Pᵣ(t), in percent. Where Pᵣ(t) is (numerically) zero this is ``nan``,
        so that the point is left out of the plot.
    :raises ValueError: for an unknown ``kind``.
    """
    p = np.asarray(polarization, dtype=float)
    p_ref = np.asarray(reference, dtype=float)

    if kind == "P":
        return p
    if kind == "deltaP":
        return p - p_ref
    if kind == "deltaP_rel":
        safe = np.abs(p_ref) > REFERENCE_TOLERANCE
        return np.where(safe, 100.0 * (p - p_ref) / np.where(safe, p_ref, 1.0), np.nan)
    raise ValueError(f"Unknown convergence quantity {kind!r}, choose among {sorted(CONVERGENCE_QUANTITIES)}")
