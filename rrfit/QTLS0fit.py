""" functions to fit QTLS0 from Qint vs nbar or FFS vs temp data"""

import numpy as np
import lmfit
from scipy.constants import k, hbar
import matplotlib.pyplot as plt

from rrfit.dataio import Device
from rrfit.fitfns import dBmtoW, nbarvsPin, Qivsnbar
from rrfit.models import QivsnbarModel


def fit_qtls0(device: Device):
    traces = [tr for tr in device.traces if not tr.is_excluded]
    traces.sort(key=lambda x: x.power)
    line_attenuation = getattr(device, "attenuation", 0)

    pow_W = np.array([dBmtoW(tr.power - line_attenuation) for tr in traces])
    fr = np.array([tr.fr for tr in traces])
    fr_avg = np.mean(fr)

    temp = np.array([tr.temperature for tr in traces])
    temp_avg = np.mean(temp)

    Qi = np.array([tr.Qi for tr in traces])
    Qi_err = np.array([tr.Qi_err for tr in traces])
    Ql = np.array([tr.Ql for tr in traces])
    avgQc = np.mean(np.array([tr.absQc for tr in traces]))
    avgphi = np.mean(np.array([tr.phi for tr in traces]))
    
    nbar = nbarvsPin(pow_W, fr_avg, Ql, avgQc)

    result = QivsnbarModel(fr=fr_avg, temp=temp_avg).fit(Qi, nbar)
    print(result.fit_report())

    device.qtls0 = result.best_values["qtls0"]
    device.qtls0_err = result.params["qtls0"].stderr

    fig, (res_ax, data_ax) = plt.subplots(2, 1, sharex=True, height_ratios=(1, 4))
    #fig.suptitle(f"Device '{device.name}' (pitch {device.pitch}um) QTLS0 from Qi vs nbar")
    nbar_interp = np.linspace(min(nbar), max(nbar), 101)
    qint_interp = Qivsnbar(nbar_interp,
                           result.best_values["qtls0"],
                           result.best_values["nc"],
                           result.best_values["beta"],
                           result.best_values["Qother"],
                           fr_avg, temp_avg)
    res_ax.scatter(nbar, result.residual, s=8, c="k")
    res_ax.set(xlabel="nbar", ylabel="residuals")
    data_ax.errorbar(nbar, Qi, yerr=Qi_err, fmt="ko", label="data")
    #data_ax.errorbar(nbar, Qi, yerr=0, fmt="ko", label="data")
    data_ax.plot(nbar_interp, qint_interp, c="r", ls="--", label="best fit")
    data_ax.set(xlabel="nbar", ylabel="Qi", yscale="log", xscale="log")
    data_ax.legend()
    fig.tight_layout()
