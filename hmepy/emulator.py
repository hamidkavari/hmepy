'''
Defines the Emulator object: the core class definition for 
the package.
'''

import numbers
import pandas as pd
import numpy as np
import copy
from hmepy.correlations import Correlator
from hmepy.utils import *

__all__ = ["Emulator"]

class Emulator:
    """
    Bayes Linear Emulator Object.

    This creates a univariate emulator. The general structure is
    f(x) = g(x) * beta + u(x), for regression functions g(x),
    regression coefficients beta, and correlation structure u(x).
    The first term can be considered to be a global output response
    to the input x, while the second term encodes the local variation
    around this global response. An emulator can be created with or
    without data; the practical method is to create prior specifications
    for beta and u(x) in the absence of data, then use the Bayes linear
    update equations to generate an emulator capable of posterior
    predictions.

    Options for Emulators
    ---------------------
    There are four core components that must be specified for an
    emulator: the basis functions (basisF), regression coefficients
    (beta), the correlation structure u, a Correlator object), and the
    input ranges (ranges). Other arguments to the constructor are
    optional, and many are used internally (for example, model is used
    by emulatorFromData).

    Attributes
    ----------
    bF: list(func)
        A collection of functions that operate on vectors of input parameters.
        Typically, the first is the constant function lambda x: 1.
    beta:
        The elements of beta must have consistent dimensions in themselves
        and with bF; if bF has length 5 then bMu is a 5-vector and bSig a
        5x5 matrix.
        bMu: [float]
            The regression coefficients, E[beta].
        bSig: [[float]]
            The covariance matrix corresponding to the regression, Var[beta].
    u:
        The correlation function, consisting of a global uncertainty sigma and a
        correlation structure. The expectation is given by E[u(x)] and the variance
        is Var[u(x)] = sigma^2*corr(x,x), Cov[u(x), u(x')]=sigma^2*corr(x, x').
        uMu: func
            The expectation of the correlation structure. By construction,
            defaults to E[u(x)] = 0 for all x.
        uSig: float
            The global sigma value.
        corr: Correlator
            The correlation structure, defined as a Correlator object.
    betaUCov: [func]
        The (x-dependent) covariance between beta and u: Cov[beta, u(x)]. This is
        an n-vector, where n is the length of E[beta]. By default no covariance is
        assumed between global (beta) and local (u) pieces.
    ranges: dict
        A dictionary of ranges with keys equal to the parameter names and where
        each value is a 2-vector consisting of a lower and upper bound. Internally, all
        parameters are scaled to reside in the region [-1,1] so the ranges specification
        is mandatory.
    data: DataFrame
        Typically provided as a DataFrame consisting of input points and output
        values (one row corresponding to one run), internally separated for ease of use.
        Seldom used directly, but called in the process of Bayes linear adjustment.
        inData: DataFrame
            The input values for the data provided
        outData: [float]
            The corresponding output values.
    outputName: str
        The name of the output that the emulator represents. Default: None
    model: Object
        If an emulator's priors have been determined using regression (for example,
        in emulatorFromData), then the corresponding model is stored here. Default: None
    modelTerms: Object
        The corresponding model terms themselves, for internal use.
    oEm: Emulator
        For an adjusted emulator, an Emulator object corresponding to the *unadjusted*
        predictions. Used internally when modifying hyperparameters of an adjusted
        emulator (for instance, changing the hyperparameters of the correlation structure u).
        Default: None
    activeVars: [bool]
        A boolean vector with length equal to the number of dimensions in the input
        space: True indicates that the relevant dimension is active in this output, while
        False indicates that it is inactive. Default: None (corresponding to all inputs
        being active).
    disc: dict
        If provided, the internal and external discrepancy as a dictionary
        {'internal': float, 'external': float}. If only one is provided then the other is
        assumed 0. Default: None (internal = 0, external = 0).
    multiplier: float
        A multiplier on the global uncertainty sigma. Default: 1
    Internal Attributes
    -------------------
    Not designed to be accessed or modified directly; all pertain to the process
    of Bayes linear adjustment.
    dCorrs: [[float]]
        The correlation matrix for the input data, corr.getCorr(inData).
    designMat: [[float]]
        The design matrix given by application of bF to inData: the (i,j)th element
        is bf[i](inData[j,:]).
    uExpMod: [float]
        The modified expectation of the correlation structure.
    uVarMod: [[float]]
        The modified variance of the correlation structure.
    betaUCorrMod: [[float]]
        The modified covariance between beta and u.

    Methods
    -------
    For more details on these methods, see the associated help files.
    getExp(x, includeC = True, cData = None)
        Calculates the emulator expectation of the points provided in x.
    getCov(x, xp = None, full = False, includeC = True, cX = None, cXP = None)
        Calculates the emulator covariance between points x and xp.
    implausibility(x, z, cutoff = None)
        Calculates the emulator implausibility for points x relative to targetz.
    adjust(data, outName)
        Performs Bayes linear adjustment of an untrained emulator.
    setSigma(sigma)
        Modifies the global variance value sigma.
    multSigma(m)
        Multiplies the global variance sigma by some constant value m.
    setHyperparams(hp, nug = None)
        Modifies the correlation structure hyperparameters.
    """

    def __init__(self, basisF, beta, u, ranges, data = None,
                 model = None, origEm = None, outName = None,
                 aVars = None, discs = None, multiplier = 1):
        '''
        Parameters
        ----------
        basisF: [func]
            The basis functions for the (global) regression surface.
        beta: {'mu': [float], 'sigma': [[float]]}
            A dictionary consisting of expectation mu and variance sigma
            for the regression coefficients.
        u: {'corr': Correlator, 'sigma': float}
            A dictionary with the specifications for the correlation structure:
            a global variance sigma and correlation structure corr.
        ranges: dict
            A dictionary of ranges, whose names match the input variable names
            and where each element is a 2-vector [lower, upper] with lower<upper.
        data: DataFrame
            If adjusted, the data used to adjust. Default: None
        model: Object
            If prior specifications were determined using, eg, linear regression,
            the regression model object. Default: None
        origEm: Emulator
            If this emulator was Bayes linear adjusted, then this is the pre-adjusted
            emulator. Default: None
        outName: str
            The name of the output emulated. Default: None
        aVars: [bool]
            A boolean vector where True at position i indicates that the input variable
            in position i is active. Default: None (all active).
        discs: dict
            Discrepancies in the form {'internal': float, 'external': float}.
            Default: None (internal = external = 0).
        multiplier: float
            The multiplicative term to apply to sigma. Modified when multSigma
            is called. Default: 1
        '''

        self.model = model
        ## Change this line
        self.modelTerms = None
        self.oEm = origEm
        self.bF = basisF
        self.bMu = beta['mu']
        self.bSig = beta['sigma']
        self.uMu = lambda x: 0
        self.multiplier = multiplier
        self.betaUCov = [lambda x: 0 for i in range(len(self.bMu))]
        if isinstance(u['sigma'], numbers.Number):
            self.uSig = self.multiplier * u['sigma']
        else:
            self.uSig = lambda x: self.multiplier * u['sigma'](x)
        self.corr = u['corr']
        if not(outName is None):
            self.outputName = outName
        if ranges is None:
            raise ValueError("No ranges specified for input parameters.")
        self.ranges = ranges
        if aVars is None:
            testDF = pd.DataFrame(np.diag(np.ones(len(ranges))),
                                  columns = ranges.keys())
            tvec = np.zeros((len(self.bF), len(ranges)))
            for i in range(len(self.bF)):
                for j in range(len(ranges)):
                    tvec[i, j] = self.bF[i](testDF.iloc[j,:])
            tvecsum = tvec.sum(axis = 0)
            self.activeVars = tvecsum > 1
        else:
            self.activeVars = aVars
        if not(self.activeVars).any():
            self.activeVars = np.full(len(ranges), True, dtype = bool)
        if not(discs is None):
            try:
                intdisc = discs['internal']
            except:
                intdisc = 0
            try:
                extdisc = discs['external']
            except:
                extdisc = 0
            self.disc = {'internal': intdisc, 'external': extdisc}
        else:
            self.disc = {'internal': 0, 'external': 0}
        if not(data is None):
            self.inData = evalFuncs(scaleInput, data.loc[:,ranges.keys()], self.ranges)
            self.outData = data.loc[:,self.outputName]
        if hasattr(self, 'inData'):
            if isinstance(self.uSig, numbers.Number):
                dCorr = self.uSig**2 * self.corr.getCorr(self.inData, actives = self.activeVars)
            else:
                sCorr = self.corr.getCorr(self.inData, actives = self.activeVars)
                dCorr = np.vstack([sCorr[i,:] * evalFuncs(self.uSig, self.inData)**2 for i in range(np.shape(sCorr)[0])])
            self.dCorrs = np.linalg.inv(dCorr)
            self.designMat = evalFuncs(self.bF, self.inData).T
            self.uVarMod = np.matmul(np.matmul(
                np.matmul(self.dCorrs, self.designMat), 
                np.matmul(self.bSig, self.designMat.T)),
                self.dCorrs)
            self.uExpMod = np.matmul(self.dCorrs,
                                     self.outData - np.matmul(self.designMat, self.bMu))
            self.betaUCovMod = np.matmul(
                np.matmul(self.bSig, self.designMat.T),
                self.dCorrs
            )
    
    def getExp(self, x, includeC = True, cData = None):
        """
        Returns the emulator expectation of points x.
         
        One can choose to separate the effects of the regression surface and the
        correlation structure by setting includeC = False, in which case
        only the regression surface (updated or otherwise) contributes.

        Parameters
        ----------
        x: DataFrame
            A dataframe of input points; one point per row.
        includeC: bool
            If True, full (adjusted) prediction is returned (the default).
            If False, only the (adjusted) regression prediction is returned.
        cData: [[float]]
            If the correlation matrix between x and the emulator input data
            has been pre-calculated, then this is used rather than calculating
            it again (used primarily in implausibility calculations). Default: None
        
        Returns
        -------
        A vector corresponding to the emulator predictions, with number of elements
        equal to the number of rows of x.
        """

        x = evalFuncs(scaleInput, x.loc[:,self.ranges.keys()], self.ranges)
        if np.count_nonzero(self.bSig) != 0 or self.model is None:
            g = evalFuncs(self.bF, x).T
            betaPart = [np.matmul(g[i], self.bMu) for i in range(np.shape(g)[0])]
        else:
            betaPart = self.model.predict(x)
        bU = evalFuncs(self.betaUCov, x)
        uPart = x.apply(self.uMu, axis = 1)
        if hasattr(self, 'inData'):
            if cData is None:
                cData = self.corr.getCorr(self.inData, x, self.activeVars)
            if isinstance(self.uSig, numbers.Number):
                added = np.matmul(
                    np.matmul(bU.T, self.designMat.T) + self.uSig**2 * cData,
                    self.uExpMod)
                uPart = uPart + added
            else:
                sigPreds = np.vstack([cData[i,:] * evalFuncs(self.uSig, x)**2 for i in range(np.shape(cData)[0])])
                uPart = uPart + np.matmul(
                    np.matmul(bU.T, self.designMat.T) + sigPreds,
                    self.uExpMod
                )
        if includeC:
            return betaPart + uPart     
        return betaPart
    def getCov(self, x, xp = None, full = False, includeC = True, cX = None, cXP = None):
        """
        Returns the emulator (co)variance between points x and xp.

        As with getExp, only the effect of the regression surface can be included
        via the includeC option. The output of this function varies dependent on the
        input: if x and xp have the same size (and full is not True), then the output is
        the individual uncertainties for each point (i.e. the diagonal of the emulator covariance
        matrix) as an n-vector (where n is the number of rows of x). If x and xp have
        different sizes, then the full matrix is provided: if x has n rows and xp has m, then
        the output is an nxm matrix with (i,j)th element Cov[f(x[i,:]), f(x'[j,:])]. Even if
        x and xp have the same size, if full = True then the full nxn matrix is returned.

        Parameters
        ----------
        x: DataFrame
            The first collection of points
        xp: DataFrame | None
            The second collection of points. If xp=None, then it is assumed that
            the desired output is getCov(x,x,...). Default: None
        full: bool
            If True, the full covariance matrix is output; if False, only the diagonal.
            Note that getCov(..., full = False, ...) is faster than np.diag(getCov(..., full = True, ...))
            since computational efficiencies can be applied during the calculation.
            Default: False
        includeC: bool
            If True, full (adjusted) prediction is returned (the default).
            If False, only the (adjusted) regression prediction is returned.
        cX, cXP: [[float]]
            If correlation matrices between x (xp) and the training data have
            already been calculated, they are provided here.

        Returns
        -------
        If full = False, an n-vector of emulator uncertainties; if full = True
        an nxm covariance matrix.

        """

        betaPart = np.zeros(np.shape(x)[0])
        x = evalFuncs(scaleInput, x.loc[:,self.ranges.keys()], self.ranges)
        if np.count_nonzero(self.bSig) != 0:
            gX = evalFuncs(self.bF, x).T
        else:
            gX = None
        bUX = evalFuncs(self.betaUCov, x)
        noneFlag = False
        if xp is None:
            noneFlag = True
            xp = x
            gXP = gX
            bUXP = bUX
        else:
            xp = evalFuncs(scaleInput, xp.loc[:,self.ranges.keys()], self.ranges)
            if np.count_nonzero(self.bSig) != 0:
                gXP = evalFuncs(self.bF, xp).T
            else:
                gXP = None
            bUXP = evalFuncs(self.betaUCov, xp)
        if full or np.shape(x)[0] != np.shape(xp)[0]:
            xXpC = self.corr.getCorr(xp, x, self.activeVars)
            if not(gX is None):
                betaPart = np.matmul(np.matmul(gX, self.bSig), gXP.T)
            if isinstance(self.uSig, numbers.Number):
                uPart = self.uSig**2 * xXpC
            else:
                uPart = np.outer(evalFuncs(self.uSig, x), evalFuncs(self.uSig, xp)) * xXpC
            if hasattr(self, 'inData'):
                if cX is None:
                    cX = self.corr.getCorr(self.inData, x, self.activeVars)
                if cXP is None:
                    if noneFlag:
                        cXP = cX
                    else:
                        cXP = self.corr.getCorr(self.inData, xp, self.activeVars)
                if isinstance(self.uSig, numbers.Number):
                    uPart = uPart - self.uSig**4 * np.matmul(
                        np.matmul(cX, self.dCorrs - self.uVarMod),
                        cXP.T
                    )
                    if np.count_nonzero(self.betaUCovMod) != 0:
                        bUX = bUX - np.matmul(self.betaUCovMod,
                                              np.matmul(self.designMat, bUX) + self.uSig**2 * cX.T)
                        if noneFlag:
                            bUXP = bUX
                        else:
                            bUXP = bUXP - np.matmul(self.betaUCovMod,
                                                    np.matmul(self.designMat, bUXP) + self.uSig**2 * cXP.T)
                else:
                    cX = np.outer(evalFuncs(self.uSig, x), evalFuncs(self.uSig, self.inData)) * cX
                    cXP = np.outer(evalFuncs(self.uSig, xp), evalFuncs(self.uSig, self.inData)) * cXP
                    uPart = uPart - np.matmul(cX,
                                              np.matmul(self.dCorrs - self.uVarMod), cXP.T)
                    if np.count_nonzero(self.betaUCovMod) != 0:
                        bUX = bUX - np.matmul(self.betaUCovMod,
                                              np.matmul(self.designMat, bUX) + cX.T)
                        bUXP = bUXP - np.matmul(self.betaUCov,
                                                np.matmul(self.designMat, bUXP) + cXP.T)
            if np.count_nonzero(bUX) != 0:
                bU = np.matmul(gX, bUXP) + np.matmul(bUX.T, gXP.T) 
            else:
                bU = np.zeros(np.shape(x)[0])
        else:
            indexAlong = range(np.shape(x)[0])
            if not(gX is None):
                betaPart = [np.matmul(gX[i,:], np.matmul(self.bSig, gXP[i,:])) for i in indexAlong]
            if np.array_equal(x, xp):
                if isinstance(self.uSig, numbers.Number):
                    uPart = [self.uSig**2 for i in indexAlong]
                else:
                    uPart = evalFuncs(self.uSig, x)**2
            else:
                if isinstance(self.uSig, numbers.Number):
                    uPartS = self.uSig**2
                else:
                    uPartS = np.diag([self.uSig(x.loc[i,:])*self.uSig(xp.loc[i,:]) for i in indexAlong])
                uPart = uPartS * np.diag(self.corr.getCorr(x, xp, self.activeVars))
            if hasattr(self, 'inData'):
                if cX is None:
                    cX = self.corr.getCorr(self.inData, x, self.activeVars)
                if cXP is None:
                    if noneFlag:
                        cXP = cX
                    else:
                        cXP = self.corr.getCorr(self.inData, xp, self.activeVars)
                if isinstance(self.uSig, numbers.Number):
                    cX = cX * self.uSig**2
                    cXP = cXP * self.uSig**2
                else:
                    cX = np.outer(evalFuncs(self.uSig, x), evalFuncs(self.uSig, self.inData)) * cX
                    cXP = np.outer(evalFuncs(self.uSig, xp), evalFuncs(self.uSig, self.inData)) * cXP
                uPart = uPart - np.sum(np.matmul(cX, self.dCorrs - self.uVarMod) * cXP, axis = 1)
                if np.count_nonzero(self.betaUCovMod) != 0:
                    bUX = bUX - np.matmul(self.betaUCovMod,
                                          np.matmul(self.designMat, bUX) + cX.T)
                    if noneFlag:
                        bUXP = bUX
                    else:
                        bUXP = bUXP - np.matmul(self.betaUCovMod,
                                                np.matmul(self.designMat, bUXP) + cXP.T)
            if np.count_nonzero(bUX) != 0:
                gValX = np.array([evalFuncs(self.bF, x.iloc[[i],:]) for i in indexAlong])[:,:,0]
                gValXP = np.array([evalFuncs(self.bF, xp.iloc[[i],:]) for i in indexAlong])[:,:,0]
                bU = [np.dot(gValX[i,:], bUXP[:,i]) + np.dot(gValXP[i,:], bUX[:,i]) for i in indexAlong]
            else:
                bU = np.zeros(np.shape(x)[0])
        if includeC:
            result = betaPart + uPart + bU
        else:
            result = betaPart
        result = np.round(result, 10)
        result[result < 0] = 1e-10
        return np.abs(result)
    def implausibility(self, x, z, cutoff = None):
        """
        Emulator Implausibility

        Given an output target value, z we can define an implausibility for a given point
        as I(x) = (E[f(x)] - z)/Var[f(x)-z]. High values of I(x) indicates that the point
        x is unlikely (according to the emulator) to give rise to a match to the observed
        value z.

        z can be supplied in two ways: either an observation value and corresponding
        uncertainty {'val': float, 'sigma': float} or as a pair of values [low, upp]
        corresponding to a lower and upper bound on the observed value. If a cutoff is provided,
        then comparisons against this cutoff are made and the output is [bool], where
        a value of True indicates that I(x) <= cutoff.

        Parameters
        ----------
        x: DataFrame
            The collection of points to calculate implausibility for.
        z: dict | [float]
            Either a (val, sigma) pair or a pair of number bounding the observed value.
        cutoff: float | None
            Whether to compare against a cutoff implausibility value.

        Returns
        -------
        If cutoff is None, then an n-vector of raw implausibility values;
        else an n-vector of bools where each element corresponds to I(x) <= cutoff.
        
        """

        tempScaleX = evalFuncs(scaleInput, x.loc[:,self.ranges.keys()], self.ranges)
        corrX = self.corr.getCorr(self.inData, tempScaleX, self.activeVars)
        if not(self.disc is None):
            disc_quad = self.disc['internal']**2 + self.disc['external']**2
        else:
            disc_quad = 0
        if isinstance(z, dict) and 'val' in z.keys():
            if not('sigma' in z.keys()):
                z['sigma'] = 0
            impVar = self.getCov(x, cX = corrX, cXP = corrX) + z['sigma']**2 + disc_quad
            imp = np.sqrt((z['val'] - self.getExp(x, cData = corrX))**2/impVar)
        else:
            def inBounds(val, low, upp):
                if val <= upp and val >= low:
                    return 0
                if val < low:
                    return -1
                return 1
            pred = self.getExp(x, cData = corrX)
            try:
                bCheck = [inBounds(y, z[0], z[1]) for y in pred]
            except:
                raise TypeError("Problem with form of targets: perhaps target missing?")
            whichComp = [z[y >= 0] for y in bCheck]
            uncerts = self.getCov(x, cX = corrX, cXP = corrX) + disc_quad
            uncerts[uncerts <= 0] = 0.0001
            imp = bCheck * (pred - whichComp)/np.sqrt(uncerts)
        if cutoff is None:
            return imp
        return imp <= cutoff
    def adjust(self, data, outName):
        """
        Bayes linear adjustment

        Performs Bayes Linear adjustment on an emulator given data. Note that this does
        not modify in place; the original Emulator object remains accessible after adjustment.
        Note that due to the requirement that outName is provided, one may
        provide a larger DataFrame to adjustment and only those columns relevant to this
        emulator will be used: this can be useful for mass adjustment of many emulators.

        Parameters
        ----------
        data: DataFrame
            The data with which to perform adjustment.
        outName: str
            The name of the output the emulator aims to represent.

        Returns
        -------
        An Emulator object corresponding to the Bayes linear updated version of the
        prior emulator. The prior Emulator object is retained as an argument within the
        new Emulator (namely as oEm).

        """

        thisDataIn = evalFuncs(scaleInput, data.loc[:,self.ranges.keys()], self.ranges)
        thisDataOut = data.loc[:,outName]
        if np.count_nonzero(np.linalg.eigvals(self.bSig)) == 0:
            newBetaVar = self.bSig
            newBetaExp = self.bMu
        else:
            gMat = evalFuncs(self.bF, thisDataIn)
            oT = self.corr.getCorr(thisDataIn, actives = self.activeVars)
            oI = np.linalg.inv(oT)
            sigInv = np.linalg.inv(self.bSig)
            newBetaVar = np.linalg.inv(
                np.matmul(gMat, np.matmul(oI, gMat.T)) + sigInv
            )
            newBetaExp = np.matmul(
                newBetaVar,
                np.matmul(sigInv, self.bMu) + np.matmul(gMat, np.matmul(oI, thisDataOut))
            )
        columnNames = [*self.ranges.keys()]
        columnNames.append(outName)
        newEm = Emulator(self.bF, {'mu': newBetaExp, 'sigma': newBetaVar},
                         {'sigma': self.uSig, 'corr': self.corr},
                         self.ranges, data = data.loc[:,columnNames],
                         origEm = self, outName = outName, model = self.model,
                         discs = self.disc, aVars = self.activeVars, multiplier = self.multiplier)
        return newEm
    def setSigma(self, sigma):
        """
        Set the global variance, sigma.

        This function should be used instead of direct manipulation of self.uSig,
        since modification of sigma for a trained emulator would also modify the process of
        training itself. This function performs modification at the oEm level and retrains if
        necessary. Note that the function does not modify in place.

        Parameters
        ----------
        sigma: float | func
            The new sigma to use in the correlation structure.
        
        Returns
        -------
        A new Emulator object (appropriately adjusted) with the new prior sigma.
        
        """
        if self.oEm is None:
            nEm = copy.deepcopy(self)
            nEm.uSig = sigma
            return nEm
        nOEm = copy.deepcopy(self.oEm)
        nOEm.uSig = sigma
        datin = evalFuncs(scaleInput, self.inData, self.ranges, False)
        datin[self.outputName] = self.outData
        return nOEm.adjust(datin, self.outputName)
    def multSigma(self, m):
        """
        Apply multiplicative factor to global variance

        Similar to setSigma, this should be used rather than direct manipulation
        of Emulator.uSig and for the same reasons. This also ensures that applying a
        scalar multiplicative factor to a functional uSig behaves as we would
        expect it to. This function does not modify in place.

        Parameters
        ----------
        m: float
            The multiplicative term
        
        Returns
        -------
        A new Emulator object with the new sigma specification.

        """
        if self.oEm is None:
            nEm = copy.deepcopy(self)
            nEm.multiplier = nEm.multiplier * m
            return nEm
        nOEm = copy.deepcopy(self.oEm)
        nOEm.multiplier = nOEm.multiplier*m
        datin = evalFuncs(scaleInput, self.inData, self.ranges, False)
        datin[self.outputName] = self.outData
        return nOEm.adjust(datin, self.outputName)
    def setHyperparams(self, hp, nug = None):
        """
        Modify the hyperparameters of the Emulator correlation structure.

        This should be called rather than changing, eg, Emulator.corr.hyperp,
        or indeed instead of using Emulator.corr.setHyperp(), since modification
        of the correlation structure hyperparameters for a Bayes linear adjusted
        emulator will change the data adjustment. This method applies the changes
        to the unadjusted emulator, and performs Bayes linear updating if necessary
        afterwards on the new structure. This function does not modify in place.

        Parameters
        ----------
        hp: dict
            A dictionary of hyperparameters for the given correlation structure
            (for examples, see the Correlator documentation)
        nug: float
            If the nugget term should also be modified, the change is specified here.
            Default: None (do not change the nugget term).
        
        Returns
        -------
        A new Emulator object (appropriately adjusted) with the new correlation
        specifications.

        """
        if nug is None:
            nug = self.corr.nugget
        currentU = self.corr
        if not(set(currentU.hyperp.keys()).issubset(set(hp.keys()))):
            raise ValueError("Not all hyperparameters are specified for " + currentU.corrName + " correlation function.")
        if self.oEm is None:
            nEm = copy.deepcopy(self)
            nEm.corr = nEm.corr.setHyperp(hp, nug)
            return nEm
        nOEm = copy.deepcopy(self.oEm)
        nOEm.corr = nOEm.corr.setHyperp(hp, nug)
        datin = evalFuncs(scaleInput, self.inData, self.ranges, False)
        datin[self.outputName] = self.outData
        return nOEm.adjust(datin, self.outputName)
    def __str__(self):
        outString = "Parameters and Ranges\n"
        for key in self.ranges.keys():
            outString = outString + key + ": (" + ", ".join(str(ran) for ran in self.ranges[key]) + "); "
        outString = outString[:-2] + "\n"
        outString = outString + "Specifications:\n"
        outString = outString + "\tBasis Functions: "
        if not(self.model is None):
            fNames = getFeatureNames(self.model)
            outString = outString + ", ".join(fNames) + "\n"
        else:
            # Do another thing (maybe need to add function_to_names into utils)
            outString = outString + "Coming soon\n"
        outString = outString + "\tActive Variables: "
        outString = outString + ", ".join([i for (i, v) in zip(list(self.ranges.keys()), self.activeVars) if v]) + "\n"
        outString = outString + "\tRegression Surface Expectation: "
        outString = outString + "; ".join(str(round(b, 4)) for b in self.bMu) + "\n"
        outString = outString + "\tRegression Surface Variance (Eigenvalues): "
        eigvals = np.real_if_close(np.linalg.eigvals(self.bSig))
        outString = outString + "(" + "; ".join(str(round(float(ev), 4)) for ev in eigvals) + ")\n"
        outString = outString + "Correlation Structure:\n"
        if hasattr(self, 'dCorrs'):
            outString = outString + "Bayes-adjusted emulator - prior specifications listed.\n"
        outString = outString + "\tPrior Variance: "
        if isinstance(self.uSig, numbers.Number):
            outString = outString + str(round(self.uSig**2, 4)) + "\n"
        else:
            outString = outString + str(self.uSig([sum(self.ranges[key])/2 for key in self.ranges.keys()])) + "\n"
        outString = outString + "\tExpectation: " + str(self.uMu(np.zeros(len(self.ranges)))) + "\n"
        outString = outString + self.corr.__str__(prepend = "\t")
        outString = outString + "Mixed Covariance: "
        outString = outString + "(" + ", ".join([str(bc) for bc in evalFuncs(self.betaUCov, np.zeros(len(self.ranges)))]) + ")\n"
        return f"{outString}"
