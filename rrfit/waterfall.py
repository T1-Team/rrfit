from scipy.constants import h, k
from scipy.special import kn
from lmfit import Parameters, minimize, report_fit
from rrfit.dataio import Device
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from rrfit.fitfns import nbarvsPin, dBmtoW
import random
import matplotlib.cm as cm

def QTLSFunc(nbar, qtls0, beta_1, beta_2, D, fr, temp):
    tanh_term = np.tanh((h * fr) / (2 * k * temp))
    sqrt_term = np.sqrt(1 + (np.power(nbar, beta_2) / (D * np.power(temp, beta_1))) * tanh_term)
    return qtls0 * sqrt_term / tanh_term

def QPQFunc(tempK, delta_QP0, tc, fr):
    gap = tc * 1.764 * k
    oneOverQPQ = (delta_QP0) * np.exp(-gap/(k*tempK)) * np.sinh(h*fr / (2*k*tempK)) * kn(0, h*fr / (2*k*tempK))
    return 1/oneOverQPQ

# error function to solve the transcendental equation for Q_int
def consistentQintError(params, outerParams, temp, freq0, power, Qc, fitQP = True):
    delta_QP0 = outerParams['delta_QP0']
    Q_TLS0    = outerParams['Q_TLS0']
    D         = outerParams['D_0']
    tc        = outerParams['tc']
    Q_other   = outerParams['Q_other']
    beta      = outerParams['beta']
    beta2     = outerParams['beta2']

    Qint = params['Qint']

    Ql = 1 / (1/Qint + 1/Qc)
    nbar = nbarvsPin(power, freq0, Ql, Qc)

    QPQ = QPQFunc(temp, delta_QP0, tc, freq0)
    QTLS = QTLSFunc(nbar, Q_TLS0, beta, beta2, D, freq0, temp)
    if fitQP:
        oneOverQ = 1/QPQ + 1/QTLS + 1/Q_other
    else:
        oneOverQ = 1/QTLS + 1/Q_other
    Q = 1/oneOverQ
    #print(f"1: {(Q-Qint) ** 2}")
    return (Q-Qint)**2


def QIntVsTemp_consistent(temp, params, freq0, power, Qc, Qint_init):

    QInt = np.zeros(np.size(power))
    for i, currentPower in enumerate(power):
        QintParam = Parameters()
        QintParam.add('Qint', value=Qint_init[i], min=Qint_init[i]/50)

        out = minimize(consistentQintError, params=QintParam, args=(params, temp[i], freq0[i], currentPower, Qc), method="least_squares")

        QInt[i] = out.params['Qint'].value
    return QInt


def QIntVsTemp_consistent_error_function(params, temps, freq0, power, Qc, Qint_init, data, errors):
    resid = []

    for i in range(len(temps)):
        #res = (data[i] - QIntVsTemp_consistent([temps[i]], params, [freq0[i]], [power[i]], Qc, [Qint_init[i]])[0]) / errors[i]
        res = (data[i] - QIntVsTemp_consistent([temps[i]], params, [freq0[i]], [power[i]], Qc, [Qint_init[i]])[0])
        #print(f"{res}")
        resid.append(res)
    return np.hstack(resid)

# fit function left by Alex. Note that it does not calculate nbar as a function of Qint, but assumes that nbar is a
# fixed number. We were calculating nbar using the Qint measured from the data
def QIntVsTemp_TLS_QP_Beta_fit_usingParams(temp, params, freq0, nbar, powerID, zResonator=None):
    delta_QP0 = params['delta_QP0']
    Q_TLS0    = params['Q_TLS0']
    D         = params['D_%i'%powerID]
    tc       = params['tc']
    Q_other   = params['Q_other']
    beta      = params['beta']
    try:
        beta2     = params['beta2']
    except:
        beta2 = 1.0
    #omega     = 2 * np.pi * freq0
    #temp = tempMK * 1e-3

    QPQ = QPQFunc(temp, delta_QP0, tc, freq0)
    QTLS = QTLSFunc(nbar, Q_TLS0, beta, beta2, D, freq0, temp)
    oneOverQ = 1/QPQ + 1/QTLS + 1/Q_other
    Q = 1/oneOverQ
    
    #if np.isnan(Q):
    #    print([(p.name, p.value) for p in params.values()])
    #    print(freq0)
    #    print(nbar)
    #    print(QPQ)
    #    print(QTLS)
    #    print(temp)
    return Q

def QIntVsTemp_TLS_QP_Beta_error_function_usingParams(params, temps, data, freq0, nbarLis, powerIDs, errors):
    resid = []
    for i in range(len(temps)):
        #val = (data[i] - QIntVsTemp_TLS_QP_Beta_fit_usingParams(temps[i], params, freq0[i], nbarLis[i], powerIDs)) / errors[i]
        val = (data[i] - QIntVsTemp_TLS_QP_Beta_fit_usingParams(temps[i], params, freq0[i], nbarLis[i], powerIDs))
        resid.append(val)
    return np.hstack(resid)

def plot_Qi_vs_temp(device: Device, figsize =(12, 8), plotParams = None, fitFunc=QIntVsTemp_consistent):
    """ """
    
    traces = [tr for tr in device.traces if not tr.is_excluded]
    min_temp = min([tr.temperature for tr in traces])
    max_temp = max([tr.temperature for tr in traces])
    data = defaultdict(list)
    for trace in traces:
        data[trace.power].append(trace)

    fig, ax = plt.subplots(figsize=figsize)    
    #fig.suptitle(f"Device {device.name} (pitch = {device.pitch}um): Qi vs temp")

    uniquePowers = {tr.power for tr in device.traces if not tr.is_excluded}
    C = cm.rainbow(np.linspace(1, 0, len(uniquePowers)))

    for idx, (power, traces) in enumerate(sorted(data.items(), reverse=True)):
        traces.sort(key=lambda x: x.temperature)
        temp = np.array([tr.temperature for tr in traces])
        Qi = np.array([tr.Qi for tr in traces])
        Qi_err = np.array([tr.Qi_err for tr in traces])
        Qc = np.mean(np.array([tr.absQc for tr in traces]))
        Ql = np.array([tr.Ql for tr in traces])
        freq0List = np.array([tr.fr for tr in traces])
        devPowerArray_W = np.array([dBmtoW(tr.power - device.attenuation) for tr in traces])

       #ax.errorbar(temp, Qi, yerr=Qi_err, mec=f"C{idx}", ls="", mfc=f"C{idx}", marker="o", ms=6, label=f"{power:.1f} dBm")

        color = C[idx] #f"C{idx}"

        if not plotParams is None:
            if fitFunc == QIntVsTemp_consistent:
                tempAxis = np.linspace(min_temp, max_temp, 100)
                ##################################################
                ### interpolation functions for freq0 and Qint ###
                # Qint interpolation is just for an initial guess to the consistent fitting algorithm
                # the interpolated freq0 data is actually used in the final curve
                

                #tempList = tempArray[matchingInds]
                #freq0List = freq0Array[matchingInds]
                #QIntList = QIntArray[matchingInds]

                    #sortInds = np.argsort(tempList)

                    #tempList = tempList[sortInds]
                    #freq0List = freq0List[sortInds]
                    #QIntList = QIntList[sortInds]

                freq0Interp = np.interp(tempAxis, temp, freq0List)
                QIntInterp = np.interp(tempAxis, temp, Qi)
                    ##################################################

                ys = fitFunc(tempAxis, plotParams, freq0Interp,
                            np.ones(np.size(tempAxis)) * devPowerArray_W[0],
                            Qc, QIntInterp)
                                      
                # if the last data point is very similar to the second last data point, we may have a case where the
                # initial guess from interp was way off. Run a few more iterations to see if it gets better
                count = 0
                while abs(ys[-1]/ys[-2]-1) < 0.001:
                    if count == 5:
                        break
                    ys = fitFunc(tempAxis, plotParams, freq0Interp,
                                 np.ones(np.size(tempAxis)) * devPowerArray_W[0],
                                 Qc, ys)
                        
                # plot the final data
                ax.plot(tempAxis, ys, label='{0:.1f} dBm'.format(power),
                        color=color)
            else:
                # calculate the nbar array from data
                nbarArray = nbarvsPin(devPowerArray_W, freq0List, Ql, Qc)
                ys = fitFunc(temp, plotParams, freq0List,
                             nbarArray, 0)
                ax.plot(temp, ys, label='{0:.1f} dBm'.format(power),
                        color=color)
            
            ax.errorbar(temp, Qi, Qi_err,
                        marker='o', markersize=8, alpha=0.8,
                        linewidth=0,
                        markeredgecolor=color, markeredgewidth=1,
                        markerfacecolor=color, zorder=1, elinewidth=1,
                        capsize=4, ecolor='k')

        else:
            ax.errorbar(temp, Qi, Qi_err,
                        marker='o', markersize=8, alpha=0.8,
                        linewidth=0,
                        markeredgecolor=color, markeredgewidth=1,
                        markerfacecolor=color, zorder=1, elinewidth=1,
                        capsize=4, ecolor='k',
                        label='{0:.1f} dBm'.format(power))

    ax.tick_params(axis='both', which='major', labelsize=16, size=8, width=2)
    ax.tick_params(which='minor', size=4, width=2)
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    ax.set_xlabel(r"Temperature ($K$)", fontsize=16)
    ax.set_ylabel(r"$Q_{int}$", fontsize=16)
    ax.set_yscale("log")
    ax.legend(frameon=False)
    fig.tight_layout()
    #plt.show()
    return fig, ax

####################################################################################################################
# Functions written by Russell McLellan, Sept 11, 2022
# fitIterated will try to repeatedly fit a dataset with random starting values. The best fit is stored in the self
# object for later use. The purpose of the function is to be more robust to local minima - rather than rely on a
# hand picked starting value, we randomly seed the parameter space and find the lowest spot
# createFitHistograms will display the results of fitIterated
# these two functions were tested for the Q vs T data, but I believe the structure is general enough to use with any
# fit function if we turn the 'Fit_QIntVsTemp' function call to different functions and adjust createFitHistograms
# to properly display different numbers of parameters in the histogram arrays
# I assume that we use the lmfit module to fit the data.
# inputs:
#   boundsDict - dictionary with the same names as the lmfit Parameters object. boundsDict[<param name>][0] is the
#                lower bound of the parameter range the algorithm will guess, and boundsDict[<param name>][1] is the
#                upper bound
#   numIter - number of guesses you want to try
#   fitFunc, errorFunc - functions for the fit. fitFunc returns the data, errorFunc the residuals
#   makePlot - boolean - set to True to plot the results, False to only store the data
# outputs:
#   initDict - dictionary with same names as the lmfit Parameters object. Each entry is a numIter length numpy array
#              corresponding to an initial guess of the parameter. initDict[<parameter1>][i],
#              initDict[<parameter2>][i], initDict[<parameter3>][i], etc correspond to a single guess of the set of
#              parameters.
#   finalDict - same as initDict, but saves the result of the fit
#   red_chi2 - numpy array, length numIter - reduced chi2 values corresponding to the fits
#   and a list of figures created if makePlot=True
def fitIterated(device, boundsDict, numIter, consistent=False, makePlot=True, fitQP=True, retries = 10, init_params=None):

    if init_params is None:
        init_params = Parameters()
        init_params.add('delta_QP0', value=2e-4, min=0)
        init_params.add('Q_TLS0', value=1e6, min=0)
        init_params.add('tc', value=2.0, min=0.0, max=4.5)
        init_params.add('Q_other', value=1e7, min=0)
        init_params.add('beta', value=1, min=0, max=2.0)
        init_params.add('beta2', value=1, min=0, max=2.0)
        init_params.add('D_0', value=100, min=0)
    setattr(device, "best_params", init_params.copy())

    # initialize output variables
    initDict = {}
    finalDict = {}
    for param in boundsDict.keys():
        initDict[param] = np.zeros(numIter)
        finalDict[param] = np.zeros(numIter)
    red_chi2_arr = np.zeros(numIter)
    # run the fits in a loop
    for i in range(numIter):
        print(f"Running iteration {i+1}/{numIter}...")
        # we sometimes encounter a ValueError if the fit is seeded with initial guesses that cause NaN. I set up a
        # while loop with a try/except block to redo any guess that throws a ValueError.
        # Note that there is no breaking from this While - if boundsDict is set too poorly, the program will hang
        # AJ - added a while loop break if exceed `retries``
        check = True
        retry_count = 0
        while check:
            try:
                for param in boundsDict.keys():
                    initDict[param][i] = random.uniform(boundsDict[param][0], boundsDict[param][1])
                    init_params[param].value = initDict[param][i]
                    # AJ - added to try and avoid nan values, need to eventually remove this because it does not make sense to bound params based on initial guess values
                    # param bounds and initial guess range should be independent knobs
                    #init_params[param].min = boundsDict[param][0]
                    #init_params[param].max = boundsDict[param][1]
                #print(f"Outside minimize: {[(p.name, p.value) for p in init_params.values()]}")
                params, red_chi2 = Fit_QIntVsTemp(device, init_params, consistent=consistent, makePlot=False)  # run the fit!
                for param in boundsDict.keys():
                    finalDict[param][i] = params[param].value
                red_chi2_arr[i] = red_chi2
                check = False
            except ValueError as err:
                retry_count += 1
                if retry_count >= retries:
                    print(f"Fit aborted! Exceeded {retries = } for {err = }")
                    return
            #    pass
    # create summary plots, if requested
    if makePlot:
        chi2Fig, countFig, probFig = createFitHistograms(device, initDict, finalDict, boundsDict, red_chi2_arr)
    else:
        chi2Fig = None
        countFig = None
        probFig = None
    # save the best fit in the hanger object
    bestInd = red_chi2_arr.argmin()
    for param in init_params.keys():
        device.best_params[param].value = initDict[param][bestInd]
    final_fit_params, initFig, fittedFig = Fit_QIntVsTemp(device, device.best_params, consistent=consistent, makePlot=makePlot)
    device.best_params = final_fit_params
    return initDict, finalDict, red_chi2_arr, [chi2Fig, countFig, probFig, initFig, fittedFig]

# inputs are taken from the fitIterated function except for:
#   probCutoff - probability cutoff for chi2 values. If a fit has a probability lower than probCutoff of being
#                correct (relative to the best fit), then it is excluded from the plot. Without probCutoff you end
#                up plotting some terrible fits that make the visualization break. We only want to look at fits that
#                are reasonably good
def createFitHistograms(device, initDict, finalDict, boundsDict, red_chi2, probCutoff=0.1):
    plt.rcParams['font.size'] = 12
    import scipy.stats as stats
    nfree = device.waterfall_fit_result.nfree
    prob = stats.chi2.pdf(red_chi2 / min(red_chi2) * nfree, nfree) / \
           stats.chi2.pdf(nfree, nfree)  # normalized probability to best result
    indsToKeep = np.where(prob > probCutoff)[0]  # indices where the probability is greater than probCutoff
    # histogram of chi2 values
    chi2Fig, hax = plt.subplots(1, 1, figsize=(4.5, 4.5), dpi=150)
    hax.hist(red_chi2[indsToKeep], 30, alpha=0.7)
    hax.set_xlabel(r'$\chi^2$', labelpad=20)
    hax.set_ylabel('Counts')
    # histograms of raw counts
    countFig, hax = plt.subplots(2, 4, figsize=(10, 6), dpi=150)
    hax = np.concatenate((hax[0], hax[1]))
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    for i, param in enumerate(initDict.keys()):
        myBins = np.linspace(min([boundsDict[param][0], min(finalDict[param][indsToKeep])]),
                             max([boundsDict[param][1], max(finalDict[param][indsToKeep])]),
                             30)
        hax[i].hist(initDict[param][indsToKeep], myBins, label='initial', alpha=0.7)
        hax[i].hist(finalDict[param][indsToKeep], myBins, label='final', alpha=0.7)
        hax[i].axvline(x=boundsDict[param][0], color='k', linestyle='--', label='initial guess bounds')
        hax[i].axvline(x=boundsDict[param][1], color='k', linestyle='--')
        hax[i].set_xlabel(param)
        hax[i].set_ylabel('Counts')
        if i == 6:  # add legend to last plot only
            hax[i].legend(bbox_to_anchor=(1, 1))
    hax[-1].set_axis_off()
    plt.suptitle('Counts histograms')
    # histograms of probability weighted data
    # The weighting will suppress any local minima results with a low probability of being correct
    probFig, hax = plt.subplots(2, 4, figsize=(10, 6), dpi=150)
    hax = np.concatenate((hax[0], hax[1]))
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    for i, param in enumerate(initDict.keys()):
        myBins = np.linspace(min([boundsDict[param][0], min(finalDict[param][indsToKeep])]),
                             max([boundsDict[param][1], max(finalDict[param][indsToKeep])]),
                             30)
        hax[i].hist(finalDict[param][indsToKeep], myBins, weights=prob[indsToKeep], label='final', alpha=0.7,
                    color='C1')
        hax[i].set_xlabel(param)
        hax[i].legend()
        hax[i].set_ylabel('Counts*prob')
    hax[-1].set_axis_off()
    plt.suptitle('Counts*prob histograms')
    return chi2Fig, countFig, probFig

def Fit_QIntVsTemp(device, init_params, consistent=False, makePlot=True):
    # rewritten from scratch by Russell, Sept 7, 2022
    # there is a separate set of calls to deal with the case when the QInt calculation is done in a self-consistent
    # manner using fitFunc=QIntVsTemp_consistent, errorFunc=QIntVsTemp_consistent_error_function. That's the only
    # way to get a smoothly varying curve.
    # TODO: add analytic way of varying freq0 for the plot instead of interpolations?
    # inputs:
    #   consistent - boolean. Set to False to calculate nbar from data, True to calculate nbar and Q simultaneously
    #   makePlot - boolean that sets whether plots are generated
    # find arrays of things for later
    
    traces = [tr for tr in device.traces if not tr.is_excluded]

    line_attenuation = getattr(device, "attenuation", 0)

    devPowerArray_W = np.array([dBmtoW(tr.power - line_attenuation) for tr in traces])
    tempArray = np.array([tr.temperature for tr in traces])
    freq0Array = np.array([tr.fr for tr in traces])

    QIntArray = np.array([tr.Qi for tr in traces])
    QIntErrArray = np.array([tr.Qi_err for tr in traces])

    Qc = np.mean(np.array([tr.absQc for tr in traces]))
    avgphi = np.mean(np.array([tr.phi for tr in traces]))
    Ql = np.array([tr.Ql for tr in traces])

    # calculate the nbar array for use in the fit
    nbarArray = nbarvsPin(devPowerArray_W, freq0Array, Ql, Qc)

    #### Fit
    if consistent:
        print("Starting consistent fit...")
        out_main = minimize(QIntVsTemp_consistent_error_function, init_params, \
                args=(tempArray, freq0Array, devPowerArray_W, Qc, QIntArray, QIntArray, QIntErrArray),
                method="least_squares")
        print("Done consistent fit")
        #if out_main.params['Q_other'].stderr is None:
        #    self.initParams_QIntVsTemp.pop('Q_other')
        #    init_params = self.initParams_QIntVsTemp
        #    out_main = minimize(QIntVsTemp_consistent_error_function_noQother, init_params, \
        #                        args=(
        #                        tempArray, freq0Array, devPowerArray_W, Qc, QIntArray, QIntArray, QIntErrArray,
        #                        self.zResonator))
        #    self.initParams_QIntVsTemp.add('Q_other', value=np.Inf, min=0, vary=False)
        #    out_main.params.add('Q_other', value=np.Inf, min=0, vary=False)
    else:
        out_main = minimize(QIntVsTemp_TLS_QP_Beta_error_function_usingParams, init_params, \
                args=(tempArray, QIntArray, freq0Array, nbarArray, 0, QIntErrArray),
                method="least_squares")
    fit_params = out_main.params
    final_red_chi2 = out_main.redchi
    device.waterfall_fit_result = out_main

    #### plot
    if makePlot:
        if consistent:
            fitFunc = QIntVsTemp_consistent
        else:
            fitFunc = QIntVsTemp_TLS_QP_Beta_fit_usingParams
        initFig = None #initFig, ax = plot_Qi_vs_temp(device, fitFunc=fitFunc, plotParams=init_params)
        #ax.set_title('Init params, ' + device.name)
        fittedFig, ax = plot_Qi_vs_temp(device, fitFunc=fitFunc, plotParams=out_main.params)
        #ax.set_title('Fitted params, ' + device.name)
        # Print out fit parameters:
        report_fit(out_main)
        print('Reduced Chi Squared: {0:.2f}'.format(out_main.redchi))
        return fit_params, initFig, fittedFig
    else:
        return fit_params, final_red_chi2