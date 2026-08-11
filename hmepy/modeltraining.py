import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import explained_variance_score
from sklearn.feature_selection import SequentialFeatureSelector
from scipy.optimize import minimize, Bounds
## Testing
from hmepy.correlations import Correlator
from hmepy.emulator import Emulator
from hmepy.utils import *
from hmepy.fullwave import *
import numpy as np
import math
import re
import warnings

__all__ = ['emulatorFromData']

## Helper function to obtain a design matrix from model
def designMatrix(data, model, paramNames):
    mNames = getFeatureNames(model)
    mFuncs = [nameToFunction(mf, paramNames) for mf in mNames]
    return evalFuncs(mFuncs, data).T

## Helper function to obtain the parameter covariance matrix
def vcov(data, model, paramNames, outputName):
    X = designMatrix(data, model, paramNames)
    Y = data[outputName]
    Xinv = np.linalg.inv(np.matmul(X.T, X))
    resid = Y - np.matmul(np.matmul(X, Xinv), np.matmul(X.T, Y))
    dof = np.shape(X)[0] - np.shape(X)[1]
    return Xinv * np.sum(resid**2/dof)

def getCoefModel(data, ranges, outputName, add = False,
                  order = 2, verbose = False):
    '''
    Model generation

    Creates a best fit of coefficients for a given data set.

    There are two ways to generate the model: either starting with all possible
    terms (including interaction terms) up to order n, and then stepwise removing;
    or starting with an intercept term only and stepwise adding terms up to order n.
    In each case, only if the fit criterion is improved is a deletion or addition
    upheld. Which method is chosen depends on the value of add: if add = True then
    stepwise addition is performed, else stepwise deletion is performed. If there is
    not enough data to allow for a full model to be fitted, and add = False, then a
    warning will be provided and stepwise addition will be performed.

    Parameters
    ----------
    data: DataFrame
        The input and output values, with each row corresponding to one observation.
    ranges: {'name': [float, float], ...}
        The ranges of the input parameters.
    outputName: str
        The name of the output to create the model for.
    add: bool
        If False, stepwise-deletion is performed; else stepwise addition is performed.
    order: int
        The maximum order of the regression terms (including interactions).
    verbose: bool
        Whether to print progress messages during model fit.
    
    Returns
    -------
    A LinearRegression object.
    '''

    if verbose: print(outputName)
    Xdat = data.loc[:,ranges.keys()]
    Ydat = data.loc[:,outputName]
    if not(add) and math.comb(len(ranges)+order, len(ranges)) > 2/3 * np.shape(data)[0]:
        if verbose:
            print("""Maximum number of regression terms is greater
                  than the available degress of freedom. Changing to add = True.""")
        add = True
    if add:
        dirStr = "forward"
    else:
        dirStr = "backward"
    model = Pipeline([
        ('poly', PolynomialFeatures(order, include_bias = False)),
        ('SFS', SequentialFeatureSelector(LinearRegression(),
                                          direction = dirStr,
                                          scoring = 'explained_variance',
                                          cv = None)),
        ('linear', LinearRegression())
    ])
    model = model.fit(Xdat, Ydat)
    if explained_variance_score(model.predict(Xdat), Ydat) > 0.9999:
        if verbose:
            print('Model selection suggests perfect fit for output ', outputName, 
                  ' - possible overfit. Removing this output from emulation')
        return None
    return model

def hyperparameterEstimate(inputs, outputs, model, hpRange,
                           corrName = "expSq", beta = None, delta = None,
                           nSteps = 30, verbose = False, logLik = True,
                           performOpt = True):
    '''
    Hyperparameter Estimation

    Performs hyperparameter fitting for a linear model.

    The 'optimal' regression coefficients beta and standard deviation sigma
    can be found in closed form, given the hyperparameters of the correlation structure.
    Those hyperparameters are estimated by first evaluating the likelihood on a coarse
    grid and finding the value(s) that maximise this likelihood (by default, the log-
    likelihood is used). This initial guess can be used as a 'seed' for a traditional
    optimisation procedure, giving the 'best' choice for the hyperparameters and the
    nugget term for the weakly stationary process. Once these are found, the final MLE
    for the regression coefficients beta and global variance sigma^2 are calculated.

    If beta is not None, then the regression coefficients are assumed to be fixed (often
    by a process such as that in getCoefModel); otherwise, they are updated with each change
    in hyperparameter during the optimisation procedure. If performOpt is False, then the
    result of the grid search is used as the final result; due to the often subdominant effect of
    the correlation structure within emulation, an 'optimal' choice of the hyperparameters is
    frequently unnecessary.

    Parameters
    ----------
    inputs: DataFrame
        The input parameter values for each output value.
    outputs: [float] | DataFrame
        The output values associated with the inputs.
    model: LinearRegression
        The fitted linear regression model.
    hpRange: {'hp': [float, float], ...}
        The allowed ranges of the hyperparameters, as a named dict.
    corrName: str
        Which correlation function to use - the string should correspond to a valid function.
    beta: [float] | None
        If relevant, the pre-fitted regression coefficients.
    delta: float | None
        The value of the nugget term, in the range [0,1].
    nSteps: int
        The coarseness of the grid: higher nSteps gives a finer grid.
    verbose: bool
        Whether to print out progress messages.
    logLik: bool
        If True, log-likelihood is used to assess quality of fit.
    performOpt: bool
        If True, optimisation is performed to find the best hyperparameters.
    
    Returns
    -------
    A named dict consisting of hyperparameters, nugget term, coefficients, and sigma.
    '''
    
    if verbose:
        print(outputs.name)
    hpStartDict = {}
    for key in hpRange.keys():
        hpStartDict[key] = hpRange[key][0]
    corr = Correlator(corrName, hpStartDict)
    H = designMatrix(inputs, model, list(inputs.columns))
    avTest = np.diag(np.ones(np.shape(inputs)[1]))
    avH = designMatrix(pd.DataFrame(data = avTest, columns = list(inputs.columns)), model, list(inputs.columns))
    isActive = [sum(avH[i,:]) > 1 for i in range(np.shape(avH)[0])]
    if sum(isActive) == 0:
        isActive = [True for i in range(len(isActive))]
    def corrMat(points, hp, delt):
        thisCorr = corr.setHyperp(hp, delt)
        return thisCorr.getCorr(points, actives = isActive)
    def funcToOpt(params, logLik = logLik, returnStats = False):
        hp = dict(zip(hpRange.keys(), params[0:(len(params))]))
        if corrName == "matern":
            if (hp['nu'] > 0.5 and hp['nu'] <= 1): hp['nu'] = 0.5
            if (hp['nu'] > 1 and hp['nu'] <= 2): hp['nu'] = 1.5
            if (hp['nu'] > 2): hp['nu'] = 2.5
        try:
            delta = params[-1]
        except:
            delta = 0
        if delta is None:
            delta = 0
        A = corrMat(inputs, hp, delta)
        aDet = np.linalg.det(A)
        if aDet <= 0:
            aDet = 1e-20
        Ainv = np.linalg.inv(A)
        if beta is None:
            invMat = np.linalg.inv(np.matmul(np.matmul(H.T, Ainv), H))
            bML = np.matmul(np.matmul(invMat, H.T), np.matmul(Ainv, outputs))
        else:
            bML = beta
        modDiff = outputs - np.matmul(H, bML)
        sigSqML = np.matmul(modDiff.T, np.matmul(Ainv, modDiff))/len(outputs)
        if sigSqML <= 0:
            sigSqML = 1e-20
        if logLik:
            lik = -len(outputs) * math.log(sigSqML)/2 - math.log(aDet)/2
        else:
            lik = 1/math.sqrt(sigSqML**len(outputs)) * 1/math.sqrt(aDet)
        if math.isinf(lik):
            lik = -math.inf
        if returnStats:
            return {'beta': bML, 'sigma': float(math.sqrt(sigSqML))}
        return lik
    ## Possibly some gradient stuff here.
    if hasattr(hpRange, 'theta') and len(hpRange['theta']) == 1:
        if delta is None:
            delta = 0.05
        bestPoint = hpRange
        bestDelta = delta
        bestParams = bestPoint.append(delta)
    else:
        if corrName == "matern":
            theta_step = np.diff(hpRange['theta'])[0] / (nSteps * 4)
            gridSearch = np.array([(x,y) for x in np.arange(hpRange['theta'][0], hpRange['theta'][1], theta_step) for y in [0.5, 1.5, 2.5, 3.5]])
        else:
            if len(hpRange) == 1:
                first_range = list(hpRange.values())[0]
                first_step = np.diff(first_range)[0] / (nSteps * 4)
                gridSearch = np.array(np.arange(first_range[0], first_range[1], first_step))
            if len(hpRange) == 2:
                first_range = list(hpRange.values())[0]
                second_range = list(hpRange.values())[1]
                firstRange = np.arange(first_range[0], first_range[1], np.diff(first_range)[0] / nSteps)
                secondRange = np.arange(second_range[0], second_range[1], np.diff(second_range)[0] / nSteps)
                gridSearch = np.array([(x,y) for x in firstRange for y in secondRange])
        if delta is None:
            dval = 0.01
        else:
            dval = delta
        if len(np.shape(gridSearch)) == 1:
            gridLiks = [funcToOpt([x, delta]) for x in gridSearch]
        else:
            gridLiks = np.apply_along_axis(lambda x: funcToOpt(np.append(x,delta)), 1, gridSearch)
        bestLik = np.argmax(gridLiks)
        if len(np.shape(gridSearch)) == 1:
            bestPoint = {list(hpRange.keys())[0]: gridSearch[bestLik]}
        else:
            bestPoint = dict(zip(hpRange.keys(), gridSearch[bestLik,:]))
        if sum(isActive) == np.shape(inputs)[1]:
            delta = 0
        if delta is None:
            delta = 0.05
        if delta == 0:
            bestDelta = 1e-10
        else:
            bestDelta = delta
        if performOpt:
            initialParams = np.append(list(bestPoint.values()), bestDelta)
            lower = np.append([hp[0] for hp in list(hpRange.values())], 0)
            upper = np.append([hp[1] for hp in list(hpRange.values())], 0.5)
            bds = Bounds(lower, upper)
            optimed = minimize(lambda x: -1*funcToOpt(x), initialParams, bounds = bds)
            bestParams = optimed.x
        else:
            bestParams = np.append(list(bestPoint.values()), bestDelta)
        otherPars = funcToOpt(bestParams, returnStats = True)
        allPoints = {'hp': dict(zip(hpRange.keys(), bestParams[0:len(bestParams)])),
                     'delta': bestParams[-1], 'sigma': otherPars['sigma'],
                     'beta': otherPars['beta']}
    return allPoints

def emulatorFromData(inputData, outputNames, ranges = None,
                     inputNames = None, emulatorType = None,
                     specifiedPriors = None, order = 2, betaVar = False,
                     corrName = "expSq", adjusted = True, discrepancies = None,
                     verbose = False, naRm = False, checkRanges = True,
                     targets = None, hasHierarchy = False, covOpts = None,
                     **kwargs):
    '''
    Emulator construction

    Given data, create an Emulator object for each output.

    Many of the parameters passed to this function are optional: the minimal operating
    example requires inputData, outputNames, and one of ranges or inputNames. If ranges
    is provided, the names are intuited from that; otherwise, the names are taken from
    inputNames and each parameter range is assumed to be [-1, 1].

    If the minimal collection of arguments is provided, then the emulators are created as
    follows:
    - The basis functions and associated regression coefficients are generated up to quadratic
      order, allowing for interactions terms. These regression parameters are henceforth assumed
      'known' (so that Var[beta] = 0).
    - The correlation function, c(x, x'), is chosen to be expSq, and a corresponding Correlator
      object is created. The hyperparameters of the correlation structure are determined using a
      constrained optimisation via maximum likelihood. This determines global variance sigma^2,
      correlation length, and size of the nugget term.
    - Emulators are created with these prior specifications, then updated with respect to the data
      using the Bayes linear update formulae, resulting in trained Emulator objects.

    In the presence of prior beliefs about the structure of the emulators, information can be supplied
    directly using the specifiedPriors argument. This may contain specific regression coefficient
    values beta and regression functions func, correlation structures u, hyperparameter values hyperP,
    and nugget term values delta, and need not include all.

    Some rudimentary data handling functionality exists (for instance, removing NA rows and checking
    for data suitability with respect to observational data, if provided), but is not a
    substitute for sense-checking data directly. If naRM = True, then rows containing at least
    one None or NA value are removed before training; if targets are provided, then output values are
    checked against these to ensure that at least the targets can be attained somewhere within the
    volume defined by the training points. If checkRanges = True, then the input ranges are potentially
    modified to be a (conservative) minimum enclosing hyperrectangle containing at least the
    original ranges; this is common in later waves of emulation and history matching when the acceptable
    space has been reduced, but should only be used if it is believed that the collection of
    training points adequately describes the full space of interest.

    Parameters
    ----------
    inputData: DataFrame
        Required. A DataFrame containing parameter and output values, one observation to a row.
    outputNames: [str]
        Required. The names of the outputs to train emulators to.
    ranges: {'name': [float, float], ...}
        Required if inputNames = None. A named dict of input parameter ranges, as an upper and lower bound.
    inputNames: [str]
        Required if ranges = None. The names of the input parameters.
    emulatorType: str | None
        What type of emulator to train (deterministic, stochastic, multistate).
    specifiedPriors: dict
        A named list of priors, as described above.
    order: int
        The maximal order of the regression models fitted. Default is 2 for quadratic fits.
    betaVar: bool
        If True, then an estimate for Var[beta] is determined; else the regression coefficients are deemed known.
    corrName: str
        The correlation type to use (for example expSq, matern, ratQuad).
    adjusted: bool
        If True, Emulator objects undergo Bayes linear adjustment before return.
    discrepancies: {'outname': {'internal': float, 'external': float}} | None
        The internal and external model discrepancy associated with each output.
    verbose: bool
        Whether to print progress messages during the fitting process.
    naRm: bool
        Whether to remove NA-containing rows (where they exist).
    checkRanges: bool
        If True, ranges are modified based on prior ranges and the extent of the training data.
    targets: {'name': {'val': float, 'sigma': float}, ...} | {'name': [lower, upper]} | None
        The observational data, in the usual form.
    hasHierarchy: bool
        Mostly internal; determines whether hierarchical emulation has been performed.
    covOpts: dict | None
        If covariance emulation is performed, any prior choices for the covariance hyperparameters.
    **kwargs:
        Any other arguments to pass to functions getCoefModel and hyperparameterEstimate.

    Returns
    -------
    A list of Emulator objects, one for each of the outputs desired.
    '''

    if not(all([outname in inputData.columns for outname in outputNames])):
        raise ValueError("outputNames do not match data.")
    if not(targets is None) and len(np.intersect1d(list(targets.keys()), outputNames)) == len(outputNames):
        doPreflight = preflight(inputData, targets, verbose = verbose, naRm = naRm)
        if doPreflight and verbose:
            print("Some outputs may not be adequately emulated due to consistent over/underestimation of outputs in training data.")
            print("Consider looking at the outputs (e.g. using behaviourPlot); outputs may require extra runs and/or transformation of data.")
    if ranges is None:
        if inputNames is None:
            raise ValueError("One of ranges or inputNames must be provided.")
        else:
            warnings.warn("No ranges provided: inputs assumed to be in range [-1, 1] for all parameters.")
            ranges = {}
            for name in inputNames:
                ranges[name] = [-1,1]
    if any([np.diff(ran)<=0 for ran in ranges.values()]):
        raise ValueError("Ranges either not specified, or misspecified.")
    if emulatorType is None:
        emulatorType = "default"
    ## Stuff about variance emulation and repetitions at points
    if naRm:
        inputData.dropna(inplace = True)
    if checkRanges:
        newRanges = {}
        for name in ranges.keys():
            coldat = inputData.loc[:,name]
            maxmin = ranges[name]
            coldiff = np.ptp(coldat)
            lowBd = max(maxmin[0], np.min(coldat) - 0.05*coldiff)
            upBd = min(maxmin[1], np.max(coldat) + 0.05*coldiff)
            newRanges[name] = [lowBd, upBd]
        ranges = newRanges
    if np.shape(inputData)[0] < 10*len(ranges):
        warnings.warn("Fewer than " + 10*len(ranges) + " valid points in " + len(ranges) + " dimensions - treat emulated outputs with caution or include more data points (minimum 10 times number of input parameters).")
    inputScaled = evalFuncs(scaleInput, inputData.loc[:,ranges.keys()], ranges)
    for name in outputNames:
        inputScaled[name] = inputData[name]
    if emulatorType == "default":
        if not(hasattr(specifiedPriors, "func")):
            if verbose:
                print("Fitting regression surfaces...")
            models = [getCoefModel(inputScaled, ranges, oName, order = order, verbose = verbose) for oName in outputNames]
            isValidModel = [not(mod is None) for mod in models]
            outputNames = [o for (o,v) in zip(outputNames, isValidModel) if v]
            models = [m for (m, v) in zip(models, isValidModel) if v]
            if len(models) == 0:
                raise ArithmeticError("No outputs can be robustly represented by regression surfaces at this order.")
            modelBetaMus = [np.insert(m.named_steps['linear'].coef_, 0, m.named_steps['linear'].intercept_) for m in models]
            modelFeatNames = [getFeatureNames(m) for m in models]
            modelBasisF = [[nameToFunction(mfi, list(ranges.keys())) for mfi in mf] for mf in modelFeatNames]
            if not(betaVar):
                modelBetaSigmas = [np.zeros((len(mbm), len(mbm))) for mbm in modelBetaMus]
            else:
                modelBetaSigmas = [vcov(inputScaled, models[ind], list(ranges.keys()), outputNames[ind]) for ind in range(len(models))]
        else:
            if ('beta' in specifiedPriors) and all(['mu' in specifiedPriors['beta'][key].keys() for key in list(specifiedPriors['beta'].keys())]):
                betaPriors = specifiedPriors['beta']
                if any([len(betaPriors[key]['mu']) != len(betaPriors[key]['func'])]):
                    raise KeyError("Provided regression functions and coefficients do not have the same length.")
                modelBetaMus = {key: betaPriors[key]['mu'] for key in betaPriors.keys()}
                if all(['sigma' in betaPriors[key].keys() for key in betaPriors.keys()]):
                    modelBetaSigmas = {key: betaPriors[key]['sigma'] for key in betaPriors.keys()}
                else:
                    modelBetaSigmas = {}
                    for key in list(betaPriors.keys()):
                        thisOutput = betaPriors[key]
                        if not 'sigma' in thisOutput or np.shape(thisOutput['sigma'])[0] != len(thisOutput['mu'] or np.shape(thisOutput['sigma'])[1] != len(thisOutput['mu'])):
                            modelBetaSigmas[key] = np.diag(np.zeros(len(thisOutput['mu'])))
                        else:
                            modelBetaSigmas[key] = thisOutput['sigma']  
                modelBasisF = {key: betaPriors[key]['func'] for key in betaPriors.keys()}
            else:
                modelBasisF = specifiedPriors['func']
                modelBetaSigmas = {key: np.zeros(len(modelBasisF[key])) for key in modelBasisF.keys()}
        if not specifiedPriors is None and hasattr(specifiedPriors, 'delta'):
            modelDeltas = specifiedPriors['delta']
        else:
            modelDeltas = None
        modelUSigmas = None
        modelUCorrs = None
        if not specifiedPriors is None and hasattr(specifiedPriors, 'u'):
            if all([hasattr(specifiedPriors['u'][key], 'sigma') for key in list(specifiedPriors['u'].keys())]):
                modelUSigmas = {key: specifiedPriors['u'][key]['sigma'] for key in specifiedPriors['u'].keys()}
            if all([hasattr(specifiedPriors['u'][key], 'corr') for key in list(specifiedPriors['u'].keys())]):
                modelUCorrs = [specifiedPriors['u'][key]['corr'] for key in list(specifiedPriors['u'].keys())]
        if modelUCorrs is None:
            modelUCorrs = [corrName for i in range(len(models))]
        if verbose:
            print("Building correlation structures...")
        if not(specifiedPriors is None) and hasattr(specifiedPriors, 'hyperP'):
            thran = specifiedPriors['hyperP']
        else:
            if 'thetaRanges' in kwargs:
                thran = kwargs.get('thetaRanges')
            else:
                def getThran(cName):
                    tRange = None
                    if cName == 'expSq' or corrName == 'ornUhl':
                        tRange = {'theta': [min(2/(order+1), 1/3), 2/order]}
                    if cName == 'matern':
                        tRange = {'theta': [min(2/(order+1), 1/3), 2/order], 'nu': [0.5, 3.5]}
                    if cName == 'ratQuad':
                        tRange = {'theta': [min(2/(order+1), 1/3), 2/order], 'alpha': [0.01,5]}
                    if tRange is None:
                        warnings.warn("User-defined correlation function" + cName + "chosen but no corresponding hyperparameter ranges. Reverting to expSq.")
                        return getThran("expSq")
                    return tRange
                thran = [getThran(cn) for cn in modelUCorrs]
        specs = [hyperparameterEstimate(inputScaled.loc[:,ranges.keys()],
                                        inputScaled.loc[:,outputNames[i]],
                                        models[i],
                                        thran[i],
                                        modelUCorrs[i],
                                        modelBetaMus[i],
                                        delta = modelDeltas,
                                        verbose = verbose, performOpt = False) for i in range(len(models))]
        if modelUSigmas is None:
            modelUSigmas = [specs[i]['sigma'] for i in range(len(specs))]
        if verbose:
            print("Creating emulators...")
        emlist = [Emulator(modelBasisF[i], beta = {'mu': modelBetaMus[i], 'sigma': modelBetaSigmas[i]},
                        u = {'sigma': modelUSigmas[i], 'corr': Correlator(modelUCorrs[i], specs[i]['hp'], specs[i]['delta'])},
                        ranges = ranges, model = models[i], outName = outputNames[i]) for i in range(len(models))]
        if (adjusted):
            for i in range(len(emlist)):
                emlist[i] = emlist[i].adjust(inputData, emlist[i].outputName)
        return emlist
