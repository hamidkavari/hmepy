import numpy as np
import pandas as pd
import math
import warnings
from copy import deepcopy
from hmepy.emulator import *

__all__ = ['nthImplausible']

'''
    Code to sequentially check emulator implausibilities.
'''

def sequentialImp(ems, x, z, n = 1, cutoff = 3):
    outfails = np.full(x.shape[0], 0)
    for em in ems:
        reduced = x.loc[[i < n for i in outfails]]
        if np.shape(reduced)[0] == 0:
            return np.full(x.shape[0], False)
        thisTarg = z[em.outputName]
        thisImp = em.implausibility(reduced, thisTarg, cutoff)
        modOutFails = outfails[[i<n for i in outfails]]
        modOutFails[[not j for j in thisImp]] = modOutFails[[not j for j in thisImp]] + 1
        outfails[[i<n for i in outfails]] = modOutFails
    return np.array([i < n for i in outfails])

def nthImplausible(ems, x, z, n = None, maxImp = math.inf,
                   cutoff = None, sequential = False, getRaw = False):
    """
        N-th Maximum Implausibility

        Computes the nth-maximum implausibility of points given a collection of emulators.

        For a collection of emulators, we often combine the implausibility
        measures for a given set of observations. The maximum implausibility of a point,
        given a set of univariate emulators and an associated collection of target values,
        is the largest implausibility of the collected set of implausibilities. The 2nd
        maximum is the maximum of the set without the largest value, and so on. By default,
        maximum implausibility will be considered when there are fewer than 10 targets to
        match to; otherwise second-maximum implausibility is considered.

        If sequential = True and a specific cutoff has been provided, then the
        emulators' implausibility will be evaluated one emulator at a time. If a point
        is judged implausible by more than n emulators, False is returned without
        further evaluated. This can be useful when dealing with large numbers of targets
        and emulators.

        Parameters
        ----------
        ems: [Emulator]
            A set of Emulator objects
        x: DataFrame
            An input point, or DataFrame of points.
        z: [Dict] | [List]
            The target values, in {"val": ..., "sigma": ...} or [upper, lower] form.
        n: int
            The implausibility level to return.
        maxImp: float
            A maximum implausibility to return.
        cutoff: float | [float]
            A numeric value, or list of such, representing allowed implausibility for each target.
        sequential: bool
            Should the emulators be evaluated sequentially?
        getRaw: bool
            Determines whether nth-implausibility should be applied or if all-target results are returned.

        
        Returns
        -------
        Either the nth-maximum implausibilities, or booleans (if cutoff is given).
        
    """
    
    if np.shape(x)[0] == 0:
        if getRaw:
            return pd.DataFrame(index=x.index if hasattr(x, "index") else [])
        if cutoff is None:
            return np.array([], dtype=float)
        return np.array([], dtype=bool)

    if isinstance(ems, Emulator):
        if not ems.outputName in list(z.keys()):
            raise KeyError("Target not found corresponding to named emulator.")
        return ems.implausibility(x, z[ems.outputName], cutoff)
    emNames = np.unique([em.outputName for em in ems])
    if n is None:
        if len(emNames) > 10:
            n = 2
        else:
            n = 1
    for key in list(z.keys()):
        if len(z[key]) == 1:
            warnings.warn("Target " + key + " is a single value; assuming it's a val with 5 percent sigma.")
            z[key] = {'val': z[key], 'sigma': z[key]*0.05}
    if n > len(emNames):
        warnings.warn("n cannot be greater than the number of targets. Switching to minimum implausibility.")
        n = len(emNames)
    if not(cutoff is None) and isinstance(cutoff, (float, int)):
        cutoff = np.full(len(ems), cutoff, dtype = 'float')
    else: 
        cutoff = np.full(len(ems), None)
    if (len(ems) > 10 or sequential) and not(cutoff[0] is None):
        return sequentialImp(ems, x, z, n, cutoff[0])
    implausibles = np.c_[[ems[i].implausibility(x, z[ems[i].outputName], cutoff[i]) for i in range(len(ems))]].T
    dImplausibles = pd.DataFrame(implausibles, index = range(implausibles.shape[0]), columns = [em.outputName for em in ems])
    if getRaw:
        return dImplausibles
    if not(cutoff[0] is None):
        impMat = np.full((x.shape[0], len(emNames)), True)
        for i in range(len(emNames)):
            key = emNames[i]
            subCols = dImplausibles.loc[:,key]
            if len(np.shape(subCols)) == 1:
                impMat[:,i] = subCols
            else:
                impMat[:,i] = np.apply_along_axis(all, 1, subCols)
    else:
        impMat = np.full((x.shape[0], len(emNames)), 0, dtype = 'float')
        for i in range(len(emNames)):
            key = emNames[i]
            subCols = dImplausibles.loc[:,key]
            if len(np.shape(subCols)) == 1:
                impMat[:,i] = subCols
            else:
                impMat[:,i] = np.apply_along_axis(max, 1, subCols)
    if not(cutoff[0] is None):
        return np.apply_along_axis(lambda x: sum(x) > len(x) - n, 1, impMat)
    if n == 1:
        imps = np.apply_along_axis(max, 1, impMat)
    else:
        imps = np.apply_along_axis(lambda x: sorted(x, reverse = True)[n-1], 1, impMat)
    imps[imps > maxImp] = maxImp
    return imps
