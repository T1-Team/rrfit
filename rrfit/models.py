""" """

from lmfit import Model
import numpy as np

from rrfit.fitfns import cable_delay_linear


class FitModel(Model):
    """thin wrapper around lmfit's Model class to simplify guessing and fitting"""

    def __init__(self, fitfn, *args, **kwargs):
        """ """
        name = self.__class__.__name__
        super().__init__(func=fitfn, name=name, *args, **kwargs)

    def fit(self, data, x, params=None, verbose=False, **kwargs):
        """ """
        if params is None:
            params = self.guess(data, x)
        result = super().fit(data, params=params, x=x, **kwargs)
        if verbose:
            print(result.fit_report())
        return result

    def make_params(self, guesses: dict = None, **kwargs):
        """ """
        if guesses is not None:
            for param, hint in guesses.items():
                self.set_param_hint(param, **hint)
        return super().make_params(**kwargs)


class S21PhaseLinearModel(FitModel):
    """Fit unwrapped s21 phase to a line to extract cable delay"""

    def __init__(self, *args, **kwargs):
        """ """
        fitfn = cable_delay_linear
        super().__init__(fitfn, *args, **kwargs)

    def guess(self, data, f):
        """ """
        tau_guess = ((data[-1] - data[0]) / (f[-1] - f[0])) / (2 * np.pi)
        theta_guess = np.average(data - 2 * np.pi * f * tau_guess)
        guesses = {
            "theta": {"value": theta_guess},
            "tau": {"value": tau_guess},
        }
        return self.make_params(guesses=guesses)
