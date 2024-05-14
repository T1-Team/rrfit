""" """

import numpy as np

def cable_delay_linear(x, tau, theta):
    """
    linear model which gives an initial guess for correcting cable delay tau
    fit this with the unwrapped phase of complex resonator signal
    x: array of probe frequencies (independent variables)
    tau: cable delay
    theta: arbitrary phase offset
    """
    return 2 * np.pi * x * tau + theta
