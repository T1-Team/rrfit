""" """

import matplotlib.pyplot as plt

from rrfit.models import S21PhaseLinearModel


def fit_cable_delay(s21_phase, f, exclude=None, plot=False) -> float:
    """
    exclude tuple(int, int): select data to exclude from cable delay fit
    """

    model = S21PhaseLinearModel()

    # fit all data to a linear model
    if exclude is None:
        result = model.fit(s21_phase, f)
        tau = result.best_values["tau"]
        if plot:
            result.plot(
                datafmt=".",
                xlabel="Frequency (MHz)",
                ylabel="arg(S21) (rad)",
                data_kws={"ms": 2, "c": "k"},
                fit_kws={"lw": 1.5, "c": "r"},
                title=f"Fitted cable delay: {tau:.3e}s",
            )
        return tau

    # fit left-most and right-most data points each to a linear model
    # return the mean tau from these two fits

    l_idx, r_idx = exclude

    # linear fit to left-most data points
    lphase, lf = s21_phase[:l_idx], f[:l_idx]
    lresult = model.fit(lphase, lf)

    # linear fit to right-most data points
    rphase, rf = s21_phase[r_idx:], f[r_idx:]
    rresult = model.fit(rphase, rf)

    ltau, rtau = lresult.best_values["tau"], rresult.best_values["tau"]
    tau = (ltau + rtau) / 2

    if plot:
        plt.scatter(f, s21_phase, s=2, c="k", label="data")
        plt.plot(lf, lresult.best_fit, lw=1.5, c="r", label="left fit")
        plt.plot(rf, rresult.best_fit, lw=1.5, c="r", label="right fit")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("arg(S21) (rad)")
        plt.title(f"Fitted cable delay: {tau:.3e}s")
        plt.legend()
        plt.show()

    return tau
