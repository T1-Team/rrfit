""" """

from collections import defaultdict
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from rrfit.circlefit import fit_circle
from rrfit.dataio import Trace, Device
from rrfit.fitfns import rr_s21_hanger, centered_phase

# from rrfit.waterfall import QIntVsTemp_consistent
from rrfit.hangerfit import fit_s21


def plot_delayfit(x, data, best_fit, residuals, tau):
    """ """
    fig, (res_ax, data_ax) = plt.subplots(2, 1, sharex=True, height_ratios=(1, 4))
    fig.suptitle(f"Fitted cable delay: {tau:.3e}s")

    res_ax.scatter(x, residuals, s=2, c="k")
    res_ax.set(xlabel="Frequency (Hz)", ylabel="residuals")

    data_ax.scatter(x, data, s=2, c="k", label="data")
    data_ax.plot(x, best_fit, c="r", label="best fit")
    data_ax.set(xlabel="Frequency (Hz)", ylabel="arg(S21) (rad)")
    data_ax.legend()

    fig.tight_layout()
    plt.show()


def plot_hangerfit(trace: Trace, do_fit: bool = False, plot_title="", figsize=(12, 12)):
    """do_fit = True will refit Trace and overwrite any existing fit param values"""

    # get raw s21
    s21raw = trace.s21real + 1j * trace.s21imag

    # do cable delay correction
    tau = getattr(trace, "tau", 0)
    s21nodelay = s21raw * np.exp(-1j * 2 * np.pi * trace.frequency * tau)

    if do_fit:
        fit_params = fit_s21(s21nodelay, trace.frequency, plot=False)
        for param, value in fit_params.items():
            setattr(trace, param, value)

    # find s21 in the canonical position
    bamp = getattr(trace, "background_amp", 1)
    bphase = getattr(trace, "background_phase", 0)
    orp = bamp * np.exp(1j * bphase)
    s21canonical = s21nodelay / orp

    # circle fit to find radius and center
    # TODO save radius, center, rp, orp, theta in hdf5 metadata
    radius, center = fit_circle(s21canonical)

    # generate plot figure
    empty = h5py._hl.base.Empty
    temp = trace.temperature
    temp_fmt = "" if type(temp) is empty else f"{temp * 1e3:.2f}mK"
    # figtitle = (
    #    f"S21 hanger fit for device {trace.device_name} trace #{trace.id} at "
    #    f"{trace.power:.1f}dBm {temp_fmt}\n"
    # )

    # extract s21 best fit from Trace, if available
    s21_args = (trace.frequency, trace.fr, trace.Ql, trace.absQc, trace.phi)
    if any(isinstance(obj, empty) for obj in s21_args):
        s21_fit = None
    else:
        s21_fit = rr_s21_hanger(*s21_args)
        # figtitle += (
        #    f"fr = {trace.fr:.3g} ± {trace.fr_err:.3g}\n"
        #    f"Qi = {trace.Qi:.2g} ± {trace.Qi_err:.2g}\n"
        #    f"Ql = {trace.Ql:.2g} ± {trace.Ql_err:.2g}\n"
        #    f"|Qc| = {trace.absQc:.2g} ± {trace.absQc_err:.2g}\n"
        #    f"phi = {trace.phi:.2f} ± {trace.phi_err:.2f}\n"
        # )

    # extract centered phase best fit from Trace
    # rp = center - (orp - center)
    # beta = np.arccos(np.abs(rp.real - center.real) / radius)
    # theta = (beta + np.pi) % (2 * np.pi) - np.pi
    # cp_fit = centered_phase(trace.frequency, trace.fr, trace.Ql, theta)

    fig = plt.figure(tight_layout=True, figsize=figsize)
    fig.suptitle(plot_title)
    gs = GridSpec(6, 6, figure=fig)

    s21_ax = fig.add_subplot(gs[:3, :3])
    s21_ax.set(xlabel="Re(S21)", ylabel="Im(S21)")
    s21_ax.set(title="Circle fit")
    s21_ax.set_aspect("equal", "datalim")
    s21_ax.locator_params(axis="both", nbins=6)
    s21_ax.grid(visible=True, alpha=0.5)
    s21_ax.scatter(s21canonical.real, s21canonical.imag, s=3, c="k", label="data")
    s21_ax.scatter(s21canonical.real[0], s21canonical.imag[0], s=8, c="g")

    s21_cp_ax = fig.add_subplot(gs[:3, 3:])
    s21_cp_ax.set(xlabel="Frequency (Hz)", ylabel="arg(S21c) (rad)")
    s21_cp_ax.set(title="Centered phase fit")
    s21_cp_ax.locator_params(axis="both", nbins=6)
    cphase = np.unwrap(np.angle(s21canonical - center))
    s21_cp_ax.scatter(trace.frequency, cphase, s=3, c="k", label="data")

    s21_mag_ax = fig.add_subplot(gs[3:, :3])
    s21_mag_ax.set(xlabel="Frequency (Hz)", ylabel="|S21| (dB)")
    s21_mag_ax.set(title="Fitted S21 Magnitude")
    s21_mag_ax.locator_params(axis="both", nbins=6)
    s21_mag = 20 * np.log10(np.abs(s21canonical))
    s21_mag_ax.scatter(trace.frequency, s21_mag, s=3, c="k", label="data")

    s21_phase_ax = fig.add_subplot(gs[3:, 3:])
    s21_phase_ax.set(xlabel="Frequency (Hz)", ylabel="arg(S21) (rad)")
    s21_phase_ax.set(title="Fitted S21 Phase")
    s21_phase_ax.locator_params(axis="both", nbins=6)
    s21_phase = np.unwrap(np.angle(s21canonical))
    s21_phase_ax.scatter(trace.frequency, s21_phase, s=3, c="k", label="data")

    if s21_fit is not None:
        s21_ax.plot(s21_fit.real, s21_fit.imag, ls="--", c="r", label="fit")
        # s21_cp_ax.plot(trace.frequency, cp_fit, ls="--", c="r", label="fit")
        s21_mag_fit = 20 * np.log10(np.abs(s21_fit))
        s21_mag_ax.plot(trace.frequency, s21_mag_fit, ls="--", c="r", label="fit")
        s21_phase_fit = np.unwrap(np.angle(s21_fit))
        s21_phase_ax.plot(trace.frequency, s21_phase_fit, ls="--", c="r", label="fit")

    s21_ax.legend()
    s21_cp_ax.legend()
    s21_mag_ax.legend()
    s21_phase_ax.legend()
    plt.show()


def plot_Qs_vs_power(device: Device, figsize=(6, 8)):
    """ """
    fig, (Qi_ax, Ql_ax, absQc_ax) = plt.subplots(3, 1, sharex=True, figsize=figsize)
    fig.suptitle(f"Device {device.name}: Qs vs input power")
    args = {"ecolor": "m", "ms": 4, "marker": "o", "mfc": "m", "mec": "m"}
    for trace in device.traces:
        if not trace.is_excluded:
            Qi_ax.errorbar(trace.power, trace.Qi, yerr=trace.Qi_err, **args)
            Ql_ax.errorbar(trace.power, trace.Ql, yerr=trace.Ql_err, **args)
            absQc_ax.errorbar(trace.power, trace.absQc, yerr=trace.absQc_err, **args)
    Qi_ax.set(ylabel="Qi")
    Ql_ax.set(ylabel="Ql")
    absQc_ax.set(xlabel="Power (dBm)", ylabel="|Qc|")
    fig.tight_layout()
    return fig
