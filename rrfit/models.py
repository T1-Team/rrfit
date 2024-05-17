""" """

from lmfit import Model
import numpy as np

from rrfit.fitfns import asymmetric_lorentzian, cable_delay_linear


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


class S21LogMagModel(FitModel):
    """ """

    def __init__(self, *args, **kwargs):
        """ """
        fitfn = asymmetric_lorentzian
        super().__init__(fitfn, *args, **kwargs)

    def guess(self, data, f):
        """ """
        # guess resonance frequency to lie at the center frequency of the acquired sweep
        fr_i = len(f) // 2
        fr_guess = f[fr_i]

        # guess y-offset based on extreme left and right off-resonant data
        xp = len(f) // 6
        l_or, r_or = np.average(data[:xp]), np.average(data[-xp:])
        ofs_guess = (l_or + r_or) / 2

        # guess lorentzian height to be the data range in linear scale
        max_, min_ = np.max(data), np.min(data)
        height_guess = 10 ** ((max_ - ofs_guess) / 20) - 10 ** ((min_ - ofs_guess) / 20)

        # guess fwhm as the frequency difference between the steepest data points
        data_lin = 10 ** (data / 20)
        dl, dr = np.diff(data_lin[:fr_i]), np.diff(data_lin[fr_i:])
        lw_i, rw_i = np.argmax(np.abs(dl)), fr_i + np.argmax(np.abs(dr))
        fwhm_guess = np.abs(f[rw_i] - f[lw_i])

        # guess phi as pi * the ratio of the prominence and range, with a sign
        l_mag, r_mag = data[:fr_i], data[fr_i:]
        prominence, range_ = max_ - ofs_guess, max_ - min_
        sign = 1 if np.average(l_mag) > np.average(r_mag) else -1
        phi_guess = sign * np.pi * prominence / range_

        # set bounds on the guesses and set them as the model's parameter hints
        guesses = {
            "ofs": {"value": ofs_guess},
            "height": {"value": height_guess, "min": 0},
            "phi": {"value": phi_guess, "min": -np.pi, "max": np.pi},
            "fr": {"value": fr_guess, "min": f[0], "max": f[-1]},
            "fwhm": {"value": fwhm_guess, "min": f[1] - f[0], "max": f[-1] - f[0]},
        }
        return self.make_params(guesses=guesses)

    def post_fit(self, result):
        """Calculate Ql, absQc, and Qi from fit result best values and show plot"""
        result.params.add("Ql", expr="fr / fwhm")
        result.params.add("absQc", expr="abs(Ql / height)")
        result.params.add("Qi", expr="1 / ((1 / Ql) - (cos(phi) / absQc))")
