""" """

import random
from typing import Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

from lmfit import Parameters, minimize
from scipy.constants import k, hbar
from scipy.special import digamma, kn, iv

from rrfit.dataio import Device

kb = k  # Boltzmann constant [J/K]

# ---------------------------------------------------------------------------
# TLS + QP helper functions
# ---------------------------------------------------------------------------

def FFS_TLS_func(tempK: np.ndarray, Q_TLS0: float, omega: float) -> np.ndarray:
    """
    TLS-induced fractional frequency shift vs temperature.

    Based on the standard digamma-log expression (cf. Gao thesis).
    Returns Δf/f0 (dimensionless).
    """
    tempK = np.asarray(tempK, dtype=float)
    # Avoid division by zero at T=0
    tempK = np.where(tempK <= 0, 1e-6, tempK)

    arg = 0.5 - hbar * omega / (2j * np.pi * kb * tempK)
    FFS_TLS = (1.0 / (Q_TLS0 * np.pi)) * (
        np.real(digamma(arg)) - np.log(hbar * omega / (2 * np.pi * kb * tempK))
    )
    return FFS_TLS

def sigma1_func(tempK: np.ndarray, omega: float, gap: float) -> np.ndarray:
    """
    Real part of complex conductivity σ1 in the local limit, normalized so σ_n = 1.
    """
    tempK = np.asarray(tempK, dtype=float)
    tempK = np.where(tempK <= 0, 1e-6, tempK)

    xi = hbar * omega / (2 * kb * tempK)
    sigma1 = (4 * gap / (hbar * omega)) * np.exp(-gap / (kb * tempK)) * np.sinh(xi) * kn(0, xi)
    return sigma1

def sigma2_func(tempK: np.ndarray, omega: float, gap: float) -> np.ndarray:
    """
    Imaginary part of complex conductivity σ2 in the local limit, normalized so σ_n = 1.
    """
    tempK = np.asarray(tempK, dtype=float)
    tempK = np.where(tempK <= 0, 1e-6, tempK)

    xi = hbar * omega / (2 * kb * tempK)
    sigma2 = (np.pi * gap / (hbar * omega)) * (
        1
        - np.sqrt(2 * np.pi * kb * tempK / gap) * np.exp(-gap / (kb * tempK))
        - 2 * np.exp(-gap / (kb * tempK)) * np.exp(-xi) * iv(0, xi)
    )
    return sigma2


def sigma2_0K_func(omega: float, gap: float) -> float:
    """σ2 evaluated at T = 0 (normalized)."""
    return np.pi * gap / (hbar * omega)

def FFS_QP_func(
    tempK: np.ndarray,
    omega: float,
    gap: float,
    alpha: float,
    Tc: float,
    gamma: float,
) -> np.ndarray:
    """
    Quasiparticle-induced fractional frequency shift vs temperature.

    Simpler, Gao-style model:
        Δf/f0(T) = (α/2) * (σ2(T)/σ2(0) - 1)

    where σ2 is the imaginary part of the complex conductivity in the
    local limit (Mattis–Bardeen approximation).

    This is monotonic in the low-T regime (T << Tc) and avoids the
    high-temperature "resurgence" seen with more complicated gamma
    formulas. The argument `gamma` is kept for API compatibility but
    is not used here.
    """
    tempK = np.asarray(tempK, dtype=float)
    tempK = np.where(tempK <= 0, 1e-6, tempK)

    # Imaginary conductivity at T and at 0 K
    sigma2 = sigma2_func(tempK, omega, gap)
    sigma2_0 = sigma2_0K_func(omega, gap)

    # Normalized ratio (1 at T=0, decreasing with T)
    ratio = sigma2 / sigma2_0

    # Gao-style QP frequency shift
    FFS_QP = (alpha / 2.0) * (ratio - 1.0)

    return FFS_QP

# ---------------------------------------------------------------------------
# FFS vs T model and residuals
# ---------------------------------------------------------------------------

GAMMA_DICT = {
    "thickFilmLocal": -0.5,
    "thickFilmExtremeAnomalous": -1.0 / 3.0,
    "thinFilmLocal": -1.0,
}


def ffs_vs_temp_model(
    tempK: np.ndarray,
    params: Parameters,
    freq0: float,
    limit: str = "thinFilmLocal",
    fitQP: bool = True,
) -> np.ndarray:
    """
    Model:

        Δf/f0(T) = FFS_TLS(T; Q_TLS0) + FFS_QP(T; alpha, Tc, gamma(limit)).
    """
    Q_TLS0 = params["Q_TLS0"].value
    Tc = params["tc"].value
    alpha = params["alpha"].value if fitQP else 0.0

    gamma = GAMMA_DICT.get(limit, GAMMA_DICT["thinFilmLocal"])

    tempK = np.asarray(tempK, dtype=float)
    omega = 2.0 * np.pi * freq0
    gap = 1.764 * kb * Tc

    FFS_TLS = FFS_TLS_func(tempK, Q_TLS0, omega)
    if fitQP:
        FFS_QP = FFS_QP_func(tempK, omega, gap, alpha, Tc, gamma)
    else:
        FFS_QP = 0.0

    return FFS_TLS + FFS_QP


def ffs_error_function(
    params: Parameters,
    tempsK: np.ndarray,
    data: np.ndarray,
    dataErr: np.ndarray,
    freq0: float,
    limit: str,
    fitQP: bool,
) -> np.ndarray:
    """Residuals for lmfit.minimize."""
    model = ffs_vs_temp_model(tempsK, params, freq0, limit=limit, fitQP=fitQP)
    return (data - model) / dataErr


# ---------------------------------------------------------------------------
# Data collection from a Device at fixed power
# ---------------------------------------------------------------------------

def _collect_ffs_data_for_device(
    device: Device,
    powerContour: Optional[float] = None,
):
    """
    Collect (tempsK, Δf/f0, Δf/f0_err, f0_ref) for a Device at one power.

    - Uses only traces with trace.is_excluded != True.
    - If powerContour is not None, only traces with that power are used.
    - Uses lowest-temperature trace as reference frequency f0_ref.
    """
    traces = []
    for tr in device.traces:
        if getattr(tr, "is_excluded", False):
            continue
        if powerContour is not None and tr.power != powerContour:
            continue
        if tr.fr is None:
            continue
        traces.append(tr)

    if len(traces) < 2:
        raise ValueError("Not enough traces to perform FFS fit (need at least 2 at this power).")

    traces.sort(key=lambda tr: tr.temperature)

    tr0 = traces[0]
    f0_ref = tr0.fr
    f0_ref_err = tr0.fr_err if tr0.fr_err is not None else 0.0

    tempsK = np.array([tr.temperature for tr in traces], dtype=float)
    freqs = np.array([tr.fr for tr in traces], dtype=float)
    freq_errs = np.array(
        [tr.fr_err if tr.fr_err is not None else 0.0 for tr in traces],
        dtype=float,
    )

    frac_shift = (freqs - f0_ref) / f0_ref
    frac_shift_err = np.sqrt(freq_errs**2 + f0_ref_err**2) / max(f0_ref, 1.0)

    # Avoid zero uncertainties
    if np.any(frac_shift_err <= 0):
        positive = frac_shift_err[frac_shift_err > 0]
        if positive.size == 0:
            frac_shift_err[:] = 1e-8
        else:
            frac_shift_err = np.where(
                frac_shift_err <= 0, 0.5 * np.min(positive), frac_shift_err
            )

    return tempsK, frac_shift, frac_shift_err, f0_ref


# ---------------------------------------------------------------------------
# Single FFS vs T fit on a Device at fixed power
# ---------------------------------------------------------------------------

def Fit_FFSVsTemp_gammaFixed(
    device: Device,
    init_params: Parameters,
    limit: str = "thinFilmLocal",
    makePlot: bool = True,
    powerContour: Optional[float] = None,
    fitQP: bool = True,
    fixTc: Optional[float] = None,
    fixalpha: Optional[float] = None,
):
    """
    Perform a single FFS vs T fit for a given Device at a fixed power.

    Returns
    -------
    params_fit : lmfit.Parameters
    initFig : matplotlib Figure or None
    fittedFig : matplotlib Figure or None
    red_chi2 : float
    """
    tempsK, frac_shift, frac_shift_err, f0_ref = _collect_ffs_data_for_device(
        device, powerContour=powerContour
    )

    params = init_params.copy()

    if fixTc is not None:
        params["tc"].value = fixTc
        params["tc"].vary = False

    if fixalpha is not None:
        params["alpha"].value = fixalpha
        params["alpha"].vary = False

    if not fitQP:
        params["alpha"].value = 0.0
        params["alpha"].vary = False

    result = minimize(
        ffs_error_function,
        params=params,
        args=(tempsK, frac_shift, frac_shift_err, f0_ref, limit, fitQP),
        method="least_squares",
    )

    red_chi2 = result.redchi

    initFig = None
    fittedFig = None

    if makePlot:
        temp_interp = np.linspace(np.min(tempsK), np.max(tempsK), 300)
        model_init = ffs_vs_temp_model(temp_interp, params, f0_ref, limit=limit, fitQP=fitQP)
        model_fit = ffs_vs_temp_model(
            temp_interp, result.params, f0_ref, limit=limit, fitQP=fitQP
        )

        # Initial guess plot (white background)
        initFig, ax = plt.subplots(figsize=(5, 3), dpi=150, facecolor="white")
        ax.set_facecolor("white")
        ax.errorbar(
            tempsK * 1e3,
            frac_shift,
            yerr=frac_shift_err,
            fmt="o",
            capsize=2,
            markersize=3,
            label="data",
        )
        ax.plot(temp_interp * 1e3, model_init, "r", label="initial guess")
        ax.set_xlabel("Temperature (mK)")
        ax.set_ylabel("Fractional frequency shift Δf/f0")
        ax.set_title(f"FFS init params, {device.name}")
        ax.legend()

        # Final fit plot (white background)
        fittedFig, ax2 = plt.subplots(figsize=(5, 3), dpi=150, facecolor="white")
        ax2.set_facecolor("white")
        ax2.errorbar(
            tempsK * 1e3,
            frac_shift,
            yerr=frac_shift_err,
            fmt="o",
            capsize=2,
            markersize=3,
            label="data",
        )
        ax2.plot(temp_interp * 1e3, model_fit, "g", label="fit")
        ax2.set_xlabel("Temperature (mK)")
        ax2.set_ylabel("Fractional frequency shift Δf/f0")
        ax2.set_title(f"FFS fit, {device.name}")
        ax2.legend()

    return result.params, initFig, fittedFig, red_chi2


# ---------------------------------------------------------------------------
# Multi-start wrapper helpers
# ---------------------------------------------------------------------------

def _create_default_init_params() -> Parameters:
    """Default starting values for Q_TLS0, alpha, tc."""
    p = Parameters()
    p.add("Q_TLS0", value=1e6, min=0.0)
    p.add("alpha", value=0.1, min=0.0)
    p.add("tc", value=4.4, min=0.0, max=10.0)  # overwritten by boundsDict
    return p


def _apply_bounds_to_params(
    init_params: Parameters,
    boundsDict: Dict[str, Tuple[float, float]],
) -> Parameters:
    """
    Copy bounds from boundsDict onto the lmfit Parameters object.

    - For each pname in boundsDict:
        * set .min and .max
        * if low == high: fix the parameter (vary = False, value = low)
    """
    for pname, (low, high) in boundsDict.items():
        if pname not in init_params:
            init_params.add(pname, value=0.5 * (low + high), min=low, max=high)
        else:
            init_params[pname].min = low
            init_params[pname].max = high

        if np.isclose(low, high):
            init_params[pname].value = low
            init_params[pname].vary = False

    return init_params


def createFitHistograms_FFS(
    initDict: Dict[str, np.ndarray],
    finalDict: Dict[str, np.ndarray],
    boundsDict: Dict[str, Tuple[float, float]],
    red_chi2: np.ndarray,
):
    """
    Lightweight visualization of the multi-start results:
    - χ² histogram
    - init vs final scatter per parameter
    """
    # χ² histogram (white background)
    chi2Fig, chi_ax = plt.subplots(1, 1, figsize=(4.0, 3.0), dpi=150, facecolor="white")
    chi_ax.set_facecolor("white")
    chi_ax.hist(red_chi2, bins=30, alpha=0.7)
    chi_ax.set_xlabel(r"$\chi^2_\nu$")
    chi_ax.set_ylabel("Count")
    chi_ax.set_title("FFS multi-start χ²")

    # Init vs final scatter per parameter (white background)
    countFig, axs = plt.subplots(
        len(boundsDict),
        1,
        figsize=(4.0, 3.0 * len(boundsDict)),
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
            cmap="viridis",
            s=15,
        )
        ax.set_xlabel(f"{pname} init")
        ax.set_ylabel(f"{pname} final")
        ax.set_title(pname)

    probFig = None  # placeholder for compatibility with older API
    return chi2Fig, countFig, probFig


# ---------------------------------------------------------------------------
# Multi-start FFS vs T fitting: main public function
# ---------------------------------------------------------------------------

def fitIterated_FFS_gammaFixed(
    device: Device,
    boundsDict: Dict[str, Tuple[float, float]],
    numIter: int,
    limit: str = "thinFilmLocal",
    makePlot: bool = True,
    powerContour: Optional[float] = None,
    fitQP: bool = True,
    retries: int = 10,
    init_params: Optional[Parameters] = None,
    fixTc: Optional[float] = None,
    fixalpha: Optional[float] = None,
):
    """
    Multi-start FFS vs T fit, analogous to waterfall.fitIterated and the old
    hangerDevice.fitIterated_FFS_gammaFixed.

    Returns
    -------
    initDict : dict
        Random initial values used in each iteration, per parameter.
    finalDict : dict
        Fitted parameter values for each iteration.
    red_chi2_arr : np.ndarray
        Reduced chi^2 for each iteration.
    figList : list
        [chi2Fig, countFig, probFig, initFig, fittedFig]
    """
    if init_params is None:
        init_params = _create_default_init_params()

    # Enforce bounds in the lmfit Parameters
    init_params = _apply_bounds_to_params(init_params, boundsDict)

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

                params_fit, _, _, red_chi2 = Fit_FFSVsTemp_gammaFixed(
                    device,
                    init_params,
                    limit=limit,
                    makePlot=False,
                    powerContour=powerContour,
                    fitQP=fitQP,
                    fixTc=fixTc,
                    fixalpha=fixalpha,
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

    params_best, initFig, fittedFig, _ = Fit_FFSVsTemp_gammaFixed(
        device,
        init_params,
        limit=limit,
        makePlot=makePlot,
        powerContour=powerContour,
        fitQP=fitQP,
        fixTc=fixTc,
        fixalpha=fixalpha,
    )

    # stash best params on the device for convenience
    setattr(device, "ffs_best_params", params_best)

    print("\n[FFS] Best-fit parameters at powerContour =", powerContour)
    for name, par in params_best.items():
        print(f"  {name:8s} = {par.value:.6g}")

    return initDict, finalDict, red_chi2_arr, [chi2Fig, countFig, probFig, initFig, fittedFig]
