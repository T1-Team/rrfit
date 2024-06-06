""" """

from lmfit.model import ModelResult
import matplotlib.pyplot as plt

from rrfit.models import S21LogMagModel


def fit_magnitude(s21_mag, f, plot=False) -> ModelResult:
    """ """
    result = S21LogMagModel().fit(s21_mag, f)
    title = ""
    params = result.params.valuesdict()
    for param in ["fr", "phi", "Ql", "absQc", "Qi"]:
        title += f"{param} = {params[param]:.2g}, "
        print(title[:-2])
    if plot:
        result.plot(
            datafmt=".",
            xlabel="Frequency (MHz)",
            ylabel="|S21| (dB)",
            data_kws={"ms": 2, "c": "k"},
            fit_kws={"lw": 1.5, "c": "r"},
            title=f"Magnitude fit: {title[:-2]}",
        )
    return result
