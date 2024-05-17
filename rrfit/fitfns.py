""" """

import numpy as np


def asymmetric_lorentzian(x, ofs, height, phi, fr, fwhm):
    """
    Lorentzian fit fn for resonator magnitude response in logarithmic scale
    x: array of probe frequencies (independent variables)
    ofs: y offset
    height: Lorentzian peak/dip to floor distance
    phi: phase factor to account for peak asymmetry
    fr: resonant frequency (peak/dip location)
    fwhm: full width half maxmimum of peak/dip
    """
    numerator = height * np.exp(1j * phi)
    denominator = 1 + 2j * ((x - fr) / fwhm)
    return 20 * np.log10(np.abs(1 - numerator / denominator)) + ofs


def cable_delay_linear(x, tau, theta):
    """
    linear model which gives an initial guess for correcting cable delay tau
    fit this with the unwrapped phase of complex resonator signal
    x: array of probe frequencies (independent variables)
    tau: cable delay
    theta: arbitrary phase offset
    """
    return 2 * np.pi * x * tau + theta
