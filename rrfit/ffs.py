""" """

import random
from typing import Dict, Tuple, Optional

import corner
import emcee
import numpy as np
import matplotlib.pyplot as plt

from lmfit import Parameters, minimize, fit_report
from scipy.constants import k, hbar
from scipy.special import digamma, kn, iv, k0, i0

from rrfit.dataio import Device

kb = k  # Boltzmann constant


def FFS_TLS_func(temp, Q_TLS0, omega):
    """ """
    if Q_TLS0 == 0.0:
        return 0.0

    prefactor = 1 / (Q_TLS0 * np.pi)
    xi = (hbar * omega) / (2 * np.pi * kb * temp)
    digamma_arg = 0.5 + 1j * xi

    return prefactor * np.real(digamma(digamma_arg) - np.log(xi))


def sigma_1_fn(temp, omega, gap):
    """ """
    if temp is None:
        return 0

    xi = (hbar * omega) / (2 * kb * temp)
    prefactor = (4 * gap) / (hbar * omega)

    return prefactor * np.exp(-gap / (kb * temp)) * np.sinh(xi) * k0(xi)


def sigma_2_fn(temp, omega, gap):
    """ """
    if temp is None:
        return (np.pi * gap) / (hbar * omega)

    xi = (hbar * omega) / (2 * kb * temp)
    prefactor = (np.pi * gap) / (hbar * omega)
    sqrt_term = np.sqrt((2 * np.pi * kb * temp) / gap)
    exp_term = np.exp(-gap / (kb * temp))

    term_1 = sqrt_term * exp_term
    term_2 = 2 * exp_term * np.exp(-xi) * i0(xi)

    return prefactor * (1 - term_1 - term_2)


def FFS_QP_func(temp, omega, gap, alpha, gamma):
    """ """

    sigma_1_T = sigma_1_fn(temp, omega, gap)
    sigma_2_T = sigma_2_fn(temp, omega, gap)

    sigma_1_0K = sigma_1_fn(None, omega, gap)
    sigma_2_0K = sigma_2_fn(None, omega, gap)

    phi_T = np.arctan(sigma_2_T / sigma_1_T)
    sin_ratio = np.sin(gamma * phi_T) / np.sin(gamma * np.pi / 2)

    sigma_norm_T = np.sqrt(sigma_1_T**2 + sigma_2_T**2)
    sigma_norm_0K = np.sqrt(sigma_1_0K**2 + sigma_2_0K**2)
    sigma_norm_ratio = (sigma_norm_T / sigma_norm_0K) ** gamma

    prefactor = alpha / 2

    return prefactor * (1 - sin_ratio * sigma_norm_ratio)


GAMMA_DICT = {
    "thickFilmLocal": -0.5,
    "thickFilmExtremeAnomalous": -1.0 / 3.0,
    "thinFilmLocal": -1.0,
}


def ffs_vs_temp_model(temp, params, limit="thinFilmLocal"):
    """ """

    Q_TLS0 = params["Q_TLS0"].value
    Tc = params["tc"].value
    alpha = params["alpha"].value
    f0 = params["f0"].value

    gamma = GAMMA_DICT[limit]
    omega = 2 * np.pi * f0
    gap = 1.764 * kb * Tc

    FFS_TLS = FFS_TLS_func(temp, Q_TLS0, omega)
    FFS_QP = FFS_QP_func(temp, omega, gap, alpha, gamma)
    FFS = FFS_TLS + FFS_QP

    return (f0 * FFS) + f0


def ffs_error_function(params, temp, data, data_err, limit):
    """ """
    model = ffs_vs_temp_model(temp, params, limit=limit)
    return (data - model) / data_err


def get_ffs_err(fr0, fr0_err, fr, fr_err):
    f0, fT, df0, dfT = fr0, fr, fr0_err, fr_err
    partial_f0 = 1 / f0
    partial_fT = -fT / (f0**2)
    return np.sqrt((partial_f0 * df0) ** 2.0 + (partial_fT * dfT) ** 2)


def fit_FFS_vs_temp(
    device: Device,
    init_params: Parameters,
    limit: str = "thinFilmLocal",
    makePlot: bool = True,
    powers=None,
    verbose=False,
    fit_QTLS0=True,
    fit_QP=True,
    fit_f0=True,
    method="least_squares",
):
    """ """

    traces = []
    for trace in device.traces:
        if trace.is_excluded:
            continue
        if powers is not None and trace.power not in powers:
            continue
        traces.append(trace)

    traces.sort(key=lambda tr: tr.temperature)
    temp = np.array([tr.temperature for tr in traces])
    fr = np.array([tr.fr for tr in traces])
    fr_err = np.array([tr.fr_err for tr in traces])
    fr_ref = traces[0].fr
    fr_ref_err = traces[0].fr_err

    params = init_params.copy()

    if not fit_QTLS0:
        params["Q_TLS0"].value = 0.0
        params["Q_TLS0"].vary = False

    if not fit_QP:
        params["alpha"].value = 0.0
        params["alpha"].vary = False
        params["tc"].vary = False

    params["f0"].value = fr_ref
    if not fit_f0:
        params["f0"].stderr = fr_ref_err
        params["f0"].vary = False

    result = minimize(
        ffs_error_function,
        params=params,
        args=(temp, fr, fr_err, limit),
        method=method,
    )

    if verbose:
        print(fit_report(result))

    red_chi2 = result.redchi

    initFig = None
    fittedFig = None

    if makePlot:
        temp_interp = np.linspace(np.min(temp), np.max(temp), 251)
        model_init_fr = ffs_vs_temp_model(temp_interp, params, limit=limit)
        model_fit_fr = ffs_vs_temp_model(temp_interp, result.params, limit=limit)

        ffs_initial = np.array([(tr.fr - fr_ref) / fr_ref for tr in traces])
        ffs_initial_err = get_ffs_err(fr_ref, fr_ref_err, fr, fr_err)

        f0_final = result.params["f0"].value
        f0_final_err = result.params["f0"].stderr
        f0_final_err = 0 if f0_final_err is None else f0_final_err
        ffs_final = np.array([(tr.fr - f0_final) / f0_final for tr in traces])
        ffs_final_err = get_ffs_err(f0_final, f0_final_err, fr, fr_err)

        model_init_ffs = (model_init_fr - fr_ref) / fr_ref
        model_fit_ffs = (model_fit_fr - f0_final) / f0_final

        # Initial guess plot (white background)
        initFig, ax = plt.subplots(figsize=(8, 6), dpi=150, facecolor="white")
        ax.set_facecolor("white")
        ax.errorbar(
            temp * 1e3,  # mK
            ffs_initial * 1e6,  # ppm
            yerr=ffs_initial_err * 1e6,  # ppm
            fmt="o",
            label="data",
        )
        ax.plot(temp_interp * 1e3, model_init_ffs * 1e6, "r", label="initial guess")
        ax.set_xlabel("Temperature (mK)")
        ax.set_ylabel(r"$\delta f/f$ (ppm)")
        ax.set_title(f"FFS init params, {device.name}")
        ax.legend()

        # Final fit plot (white background)
        fittedFig, ax2 = plt.subplots(figsize=(8, 6), dpi=150, facecolor="white")
        ax2.set_facecolor("white")
        ax2.errorbar(
            temp * 1e3,  # mK
            ffs_final * 1e6,  # ppm
            yerr=ffs_final_err * 1e6,  # ppm
            fmt="o",
            label="data",
        )
        ax2.plot(temp_interp * 1e3, model_fit_ffs * 1e6, "g", label="fit")
        ax2.set_xlabel("Temperature (mK)")
        ax2.set_ylabel(r"$\delta f/f$ (ppm)")
        ax2.set_title(f"FFS fit, {device.name}")
        ax2.legend()

        initFig.tight_layout()
        fittedFig.tight_layout()

    return result.params, initFig, fittedFig, red_chi2


def _create_default_init_params() -> Parameters:
    """ """
    params = Parameters()
    params.add("Q_TLS0", value=1e6, min=0.0)
    params.add("alpha", value=0.5, min=0.0)
    params.add("tc", value=0.7, min=0.0, max=2.0)
    params.add("f0", value=5e9, min=1e9, max=9e9)
    return params


def createFitHistograms_FFS(
    initDict: Dict[str, np.ndarray],
    finalDict: Dict[str, np.ndarray],
    boundsDict: Dict[str, Tuple[float, float]],
    red_chi2: np.ndarray,
):
    """ """

    # χ² histogram (white background)
    chi2Fig, chi_ax = plt.subplots(1, 1, figsize=(8.0, 6.0), dpi=150, facecolor="white")
    chi_ax.set_facecolor("white")
    chi_ax.hist(red_chi2, bins=30, alpha=0.7)
    chi_ax.set_xlabel(r"$\chi^2_\nu$")
    chi_ax.set_ylabel("Count")
    chi_ax.set_title("FFS multi-start χ²")

    # Init vs final scatter per parameter (white background)
    countFig, axs = plt.subplots(
        len(boundsDict),
        1,
        figsize=(8.0, 6.0 * len(boundsDict)),
        dpi=150,
        sharex=False,
        facecolor="white",
    )
    if len(boundsDict) == 1:
        axs = [axs]

    for ax, (pname, _) in zip(axs, boundsDict.items()):
        ax.set_facecolor("white")
        ax.scatter(
            initDict[pname],
            finalDict[pname],
            c=red_chi2,
            cmap="plasma",
        )
        ax.set_xlabel(f"{pname} init")
        ax.set_ylabel(f"{pname} final")
        ax.set_title(pname)

    probFig = None  # placeholder for compatibility with older API
    chi2Fig.tight_layout()
    countFig.tight_layout()
    return chi2Fig, countFig, probFig


def fitIterated_FFS_gammaFixed(
    device: Device,
    boundsDict: Dict[str, Tuple[float, float]],
    numIter: int,
    retries: int = 25,
    init_params: Optional[Parameters] = None,
    limit: str = "thinFilmLocal",
    makePlot: bool = True,
    powers=None,
    fit_QTLS0=True,
    fit_QP=True,
    fit_f0=True,
    method="least_squares",
):
    """ """
    if init_params is None:
        init_params = _create_default_init_params()

    initDict: Dict[str, np.ndarray] = {}
    finalDict: Dict[str, np.ndarray] = {}
    red_chi2_arr = np.zeros(numIter)

    for pname in boundsDict.keys():
        initDict[pname] = np.zeros(numIter)
        finalDict[pname] = np.zeros(numIter)

    # multi-start loop
    for i in range(numIter):
        check = True
        retry_count = 0

        while check:
            try:
                # Randomize starting values within bounds
                for pname, (low, high) in boundsDict.items():
                    val = random.uniform(low, high)
                    initDict[pname][i] = val
                    init_params[pname].value = val

                params_fit, _, _, red_chi2 = fit_FFS_vs_temp(
                    device,
                    init_params,
                    limit=limit,
                    makePlot=False,
                    powers=powers,
                    fit_QTLS0=fit_QTLS0,
                    fit_QP=fit_QP,
                    fit_f0=fit_f0,
                    method=method,
                )

                for pname in boundsDict.keys():
                    finalDict[pname][i] = params_fit[pname].value

                red_chi2_arr[i] = red_chi2
                check = False

            except ValueError as err:
                retry_count += 1
                if retry_count >= retries:
                    raise ValueError(
                        f"FFS fit aborted after {retries} retries in iteration {i}: {err}"
                    )

    # summary plots for all iterations
    if makePlot:
        chi2Fig, countFig, probFig = createFitHistograms_FFS(
            initDict, finalDict, boundsDict, red_chi2_arr
        )
    else:
        chi2Fig = countFig = probFig = None

    # choose best iteration and re-fit with plotting
    bestInd = int(np.argmin(red_chi2_arr))
    for pname in boundsDict.keys():
        init_params[pname].value = initDict[pname][bestInd]

    params_best, initFig, fittedFig, _ = fit_FFS_vs_temp(
        device,
        init_params,
        limit=limit,
        makePlot=True,
        powers=powers,
        verbose=True,
        fit_QTLS0=fit_QTLS0,
        fit_QP=fit_QP,
        fit_f0=fit_f0,
        method=method,
    )

    # stash best params on the device for convenience
    setattr(device, "ffs_fit_params", params_best)

    print(f"\n[FFS] Best-fit parameters at {powers = }")
    for name, par in params_best.items():
        print(f"  {name:8s} = {par.value:.6g}")

    return (
        initDict,
        finalDict,
        red_chi2_arr,
        [chi2Fig, countFig, probFig, initFig, fittedFig],
    )


def plot_ffs_vs_temp(devices, figsize=(8, 6)):
    """ """
    fig, ax = plt.subplots(figsize=figsize)

    for idx, device in enumerate(devices):
        traces = [trace for trace in device.traces if not trace.is_excluded]
        temperatures = np.array([trace.temperature for trace in traces])
        sorted_temp_idx = np.argsort(temperatures)
        sorted_temperatures_mK = temperatures[sorted_temp_idx] * 1e3
        frs = np.array([trace.fr for trace in traces])[sorted_temp_idx]
        ffrs_ppm = ((frs - frs[0]) / frs[0]) * 1e6

        ax.scatter(sorted_temperatures_mK, ffrs_ppm, c=f"C{idx}", label=device.pitch)

    ax.set_xlabel("Temperature (mK)", fontsize=20)
    ax.set_ylabel(r"$\delta f/f$ (ppm)", fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=20, width=2)
    ax.legend()


def emcee_ffs(
    device,
    limit="thinFilmLocal",
    **emcee_kws,
):
    """Device must have an attribute 'ffs_fit_params' from a previous fit"""

    traces = [tr for tr in device.traces if not tr.is_excluded]
    traces.sort(key=lambda tr: tr.temperature)
    temp = np.array([tr.temperature for tr in traces])
    fr = np.array([tr.fr for tr in traces])
    fr_err = np.array([tr.fr_err for tr in traces])
    params = device.ffs_fit_params

    if not emcee_kws:
        emcee_kws = dict(
            steps=5000,
            burn=500,
            thin=20,
            is_weighted=True,
            progress=True,
        )

    emcee_result = minimize(
        ffs_error_function,
        params=params,
        args=(temp, fr, fr_err, limit),
        method="emcee",
        nan_policy="omit",
        **emcee_kws,
    )

    emcee_plot = corner.corner(
        emcee_result.flatchain,
        labels=emcee_result.var_names,
        truths=list(emcee_result.params.valuesdict().values()),
    )
    emcee_plot.tight_layout()

    print("Median of posterior probability distribution")
    print("--------------------------------------------")
    print(fit_report(emcee_result.params))

    highest_prob = np.argmax(emcee_result.lnprob)
    hp_loc = np.unravel_index(highest_prob, emcee_result.lnprob.shape)
    mle_soln = emcee_result.chain[hp_loc]
    for i, par in enumerate(params):
        params[par].value = mle_soln[i]

    print("\nMaximum Likelihood Estimation from emcee       ")
    print("-------------------------------------------------")
    print("Parameter  MLE Value   Median Value   Uncertainty")
    fmt = "  {:5s}  {:11.5f} {:11.5f}   {:11.5f}".format
    for name, param in params.items():
        print(
            fmt(
                name,
                param.value,
                emcee_result.params[name].value,
                emcee_result.params[name].stderr,
            )
        )

    print("\nError estimates from emcee:")
    print("------------------------------------------------------")
    print("Parameter  -2sigma  -1sigma   median  +1sigma  +2sigma")

    for name in params.keys():
        quantiles = np.percentile(
            emcee_result.flatchain[name], [2.275, 15.865, 50, 84.135, 97.275]
        )
        median = quantiles[2]
        err_m2 = quantiles[0] - median
        err_m1 = quantiles[1] - median
        err_p1 = quantiles[3] - median
        err_p2 = quantiles[4] - median
        fmt = "  {:5s}   {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f}".format
        print(fmt(name, err_m2, err_m1, median, err_p1, err_p2))
