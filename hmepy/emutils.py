import numpy as np
import pandas as pd
import json
import re
from hmepy.emulator import Emulator
from hmepy.correlations import Correlator
from hmepy.modeltraining import emulatorFromData
from hmepy.implausibility import nthImplausible
from hmepy.utils import *

__all__ = ['collectEmulators', 'getRanges',
'importEmulatorFromJSON', 'exportEmulatorToJSON',
'spaceReduction']

def _is_emulator_like(value):
    return isinstance(value, Emulator) or (
        hasattr(value, "outputName")
        and hasattr(value, "ranges")
        and hasattr(value, "implausibility")
    )

def collectEmulators(ems, targets = None, cutoff = 3,
                     ordering = ['p', 'i', 'v'], sampleSize = 200):
    '''
    Emulator Collation

    Collects and orders emulators for use in other functions.

    Most often used as a pre-processing stage for functions such as nthImplausible
    and generateNewDesign, this function unnests lists of lists of emulators (a commmon
    output after multiple waves of emulation), ordering them in terms of their possible
    restrictiveness for efficiency in calculating implausbility.

    Criteria for ordering and the short form string are as follows:
    - Number of Parameters ('p'): Emulators with more parameters are likely to have
      arisen from later-wave emulation, so take precedence;
    - Implausibility ('i', requires targets): A test collection of points is created,
      with size sampleSize, and the emulator implausibilities are calculated for each
      emulator. The most restrictive ones are placed first.
    - Input space volume ('v'): A hyperparameter volume is created for each emulator
      using em.ranges; those with smaller volume are likely to have come from later
      waves of emulation, and take precedence.
    The criteria can be ordered in whichever fashion is desired: the default is that
    number of parameters is considered first, using implausibility (if targets are
    provided) to break ties, and using hyperrectangle volume to break any remaining ties.
    If two emulators are indistinguishable using the metrics, they maintain the order
    possesses prior to the collectEmulators call.

    Parameters:
    -----------
    ems: [Emulator] | [[Emulator], [Emulator], ...]
        The list (or nested list) or Emulator objects.
    targets: [{'val': float, 'sigma': float}] | [[float, float]] | None
        Targets for the outputs (if 'i' is used).
    cutoff: float
        If using option 'i', the cutoff for implausibility.
    ordering: [str]
        The metrics to use, and the order in which to perform them.
    sampleSize: int
        How many points to consider implausibility for if using option 'i'.
    
    Returns:
    --------
    A flattened list of emulators, ordered with respect to the chosen metrics.
    '''
    
    if _is_emulator_like(ems):
        return [ems]
    if all(_is_emulator_like(em) for em in ems):
        emNames = [em.outputName for em in ems]
        emRangeLength = [len(em.ranges) for em in ems]
        emRangeProds = [np.prod([np.diff(ran)[0] for ran in em.ranges.values()]) for em in ems]
        if targets is None:
            whichOrder = list(set(['p', 'v']).intersection(ordering))
        else:
            whichOrder = list(set(['p', 'i', 'v']).intersection(ordering))
        if len(whichOrder) == 0:
            if targets is None: whichOrder = ['p', 'v']
            else: whichOrder = ['p', 'i', 'v']
        if not(targets is None):
            sampPoints = np.zeros((sampleSize, max(emRangeLength)), dtype = 'float')
            mostRanges = (ems[np.argmax(emRangeLength)]).ranges
            for i in range(sampPoints.shape[1]):
                sampPoints[:,i] = np.random.uniform(list(mostRanges.values())[i][0], list(mostRanges.values())[i][1], sampleSize)
            sampPoints = pd.DataFrame(sampPoints, columns = list(mostRanges.keys()))
            emImps = nthImplausible(ems, sampPoints, targets, cutoff = cutoff, getRaw = True)
            impOrder = np.apply_along_axis(sum, 0, emImps)
        emOrder = pd.DataFrame(data = {'p': [-1*erl for erl in emRangeLength], 'v': emRangeProds})
        if not(targets is None):
            emOrder['i'] = impOrder
        emOrder = emOrder.loc[:,whichOrder].sort_values(by = whichOrder, axis = 0)
        emOrderIndices = list(emOrder.index)
        return [ems[i] for i in emOrderIndices]
    return collectEmulators([em for emlist in ems for em in emlist], targets = targets,
                            cutoff = cutoff, ordering = ordering, sampleSize = sampleSize)

def getRanges(ems, minimal = True):
    '''
    Get Parameter Ranges

    Determine mimimal or maximal parameter ranges for a collection of emulators.

    This is a more complex version of a call to em.ranges, which accommodates a variety
    of inputs to the ems argument (including nested and hierarchical emulator collections).
    Either the largest (most conservative) set of ranges or the smallest (most restrictive)
    are returned, depending on the value of minimal.

    This is used in point generation in two ways: for Latin Hypercube proposals, the minimal
    hyperrectangle is used; for importance sampling, the largest set of ranges is taken.

    Parameters:
    -----------
    ems: [Emulator] | [[Emulator]]
        The emulators in question
    minimal: bool
        Whether to return the smallest (True) or largest (False) hyperrectangle.
    
    Returns:
    --------
    A named list of ranges, each consisting of an upper and lower bound.
    '''

    ems = collectEmulators(ems)
    rangeLengths = [len(list(em.ranges.values())) for em in ems]
    if not len(np.unique(rangeLengths)) == 1:
        ems = [em for em in ems if rangeLengths == max(rangeLengths)]
    rangeWidths = np.full((len(ems), max(rangeLengths)), 0, dtype = 'float')
    for i in range(len(ems)):
        for j in range(max(rangeLengths)):
            rangeWidths[i, j] = np.diff(list(ems[i].ranges.values())[j])[0]
    if minimal:
        whichChoose = np.apply_along_axis(np.argmin, 0, rangeWidths)
    else:
        whichChoose = np.apply_along_axis(np.argmax, 0, rangeWidths)
    newRanges = [list(ems[whichChoose[i]].ranges.values())[i] for i in range(len(list(ems[0].ranges.values())))]
    return dict(zip(list(ems[0].ranges.keys()), newRanges))


def exportEmulatorToJSON(ems, inputs = None, filename = None, outputType = "json"):
    if isinstance(ems, Emulator):
        if inputs is None:
            inputs = {"data": None}
        inputRanges = ems.ranges
        outName = ems.outputName
        inputUID = None
        if (hasattr(ems, "inData")):
            inputDF = evalFuncs(scaleInput, ems.inData, inputRanges, False)
            inputUID = inputDF.apply(lambda x: hash(tuple(x)), axis = 1)
            inputDF[outName] = ems.outData
            if (not inputs['data'] is None):
                if not outName in list(inputs["data"].columns):
                    inputs["data"][outName] = None
                alreadyUsed = [str(uid) in str(list(inputs["data"]["uid"])) for uid in inputUID]
                for i in range(len(alreadyUsed)):
                    if alreadyUsed[i]:
                        matchingIndex = inputs["data"][inputs["data"]["uid"] == list(inputUID)[i]].index[0]
                        inputs["data"].loc[matchingIndex, outName] = inputDF.iloc[i][outName]
                inputSubset = pd.concat([inputUID, inputDF], axis = 1)[np.logical_not(alreadyUsed)]
                inputSubset.columns = ["uid"] + list(inputDF.columns)
                if not np.shape(inputSubset)[0] == 0:
                    for nm in list(inputs["data"].columns):
                        if not nm in list(inputs["data"].columns):
                            inputSubset[nm] = None
                    inputs['data'] = pd.concat([inputs["data"], inputSubset], axis = 0)
            else:
                inputs['data'] = pd.concat([inputUID, inputDF], axis = 1)
                inputs['data'].columns = ["uid"] + list(inputDF.columns)
        else:
            thisDat = None
        inputs = {"em": {
            "in.ranges": inputRanges,
            "out.name": outName,
            "input.uid": list(inputUID),
            "basis.f": list(getFeatureNames(ems.model)),
            "basis.beta": list(ems.bMu),
            "emulator.discrepancies": ems.disc,
            "corr.name": ems.corr.corrName,
            "corr.sigma": ems.uSig,
            "corr.details": ems.corr.hyperp,
            "nugget": ems.corr.nugget
        },
        "data": inputs['data'].fillna("NULL")}
    else:
        if inputs is None:
            inputs = {
                "data": None,
                "ems": None
            }
        ems = collectEmulators(ems, ordering = "p")
        for em in ems:
            exEmDetails = exportEmulatorToJSON(em, inputs, outputType = "object")
            if inputs['ems'] is None:
                inputs['ems'] = {"emulator1": None}
                inputIndex = 1
            else:
                inputIndex = len(inputs['ems'])+1
            inputs['ems']['emulator' + str(inputIndex)] = exEmDetails['em']
            inputs['data'] = exEmDetails['data']
    if filename is None:
        if outputType == "json":
            inputs['data'] = inputs['data'].to_dict(orient = "records")
            jsonRes = json.dumps(inputs, ensure_ascii = False)
            return jsonRes
        else:
            return inputs
    else:
        inputs['data'] = inputs['data'].to_dict(orient = "records")
        with open(filename, 'w') as f:
            json.dump(inputs, f, ensure_ascii = False, indent = 4)

def importEmulatorFromJSON(filename = None, details = None):
    if not filename is None:
        with open(filename) as f:
            details = json.load(f)
    inputData = pd.DataFrame(details['data'])
    if "em" in list(details.keys()):
        inEmDetails = details['em']
        if len(np.shape(inEmDetails['basis.f'])) == 2:
            actualFuncs = [inEmDetails['basis.f'][i][1] for i in range(len(inEmDetails['basis.f']))]
            newFuncs = [re.sub(r"x\[\[(\d+)\]\]", lambda x: list(inEmDetails['in.ranges'].keys())[int(x.group(1))-1], actualFuncs[i]) for i in range(len(actualFuncs))]
            newFuncs = [re.sub(r"\*\s", "", re.sub(r"return\((.*)\)", "\\1", newFuncs[i])) for i in range(len(newFuncs))]
            functions = [nameToFunction(newFuncs[i], list(inEmDetails['in.ranges'])) for i in range(len(newFuncs))]
        else:
            if any([re.search(nm, elem) for nm in inEmDetails['in.ranges'] for elem in inEmDetails['basis.f']]):
                rangeNames = list(inEmDetails['in.ranges'].keys())
                functions = [nameToFunction(inEmDetails['basis.f'][i], rangeNames) for i in range(len(inEmDetails['basis.f']))]
        inDataRelev = [inputData['uid'][i] in inEmDetails['input.uid'] for i in range(np.shape(inputData)[0])]
        inData = inputData.iloc[inDataRelev,:]
        for key in ["corr.name", 'corr.sigma', "out.name"]:
            if not isinstance(inEmDetails[key], (str, int, float)):
                inEmDetails[key] = inEmDetails[key][0]
        inEmDetails['corr.name'] = re.sub("_([a-z])", lambda x: x.group(1).upper(), inEmDetails['corr.name'])
        em = emulatorFromData(inData, [inEmDetails['out.name']],
        inEmDetails['in.ranges'], discrepancies = inEmDetails['emulator.discrepancies'],
        specifiedPriors = {"func": functions, "beta": {"mu": inEmDetails['basis.beta']},
        "u": {"sigma": inEmDetails['corr.sigma'], 
        "corr": Correlator(inEmDetails['corr.name'], inEmDetails['corr.details'],
        nug = inEmDetails['nugget'])}}, verbose = False, moreVerbose = False, checkRanges = False)
    else:
        emList = [importEmulatorFromJSON(details = {
                "data": inputData, "em": details['ems'][key]
            })[0] for key in list(details['ems'].keys())]
        return emList
    return em

def spaceReduction(ems, prevPoints, targets, newPoints = None):
    '''
    Estimate Space Reduction

    Provide an interval for the estimated remaining space after a wave of emulation.

    The space removed at each wave can be used as an indication of how informative
    the data is, or to identify when performing further waves of history matching
    may be unnecessary.

    The implausibility of the trained emulators is evaluated on the proposed points
    from the previous wave, and a Wilson interval is generated based on the acceptance
    rate found. If this wave of emulators has also been used to propose new points, then
    the acceptance rate is combined with the ratio of space occupied by the old and new
    proposals (providing an upper bound on the volume of remaining space). 

    Parameters:
    -----------
    ems: [Emulator] | [[Emulator]]
        The emulators in question
    prevPoints: DataFrame
        The previous wave of proposed points
    targets: [{'val': float, 'sigma': float}] | [[float, float]]
        The observational targets to match to
    newPoints: DataFrame | None
        New points (if they have already been proposed)
    
    Returns:
    --------
    A pair of floats, representing lower and upper bounds for the space remaining.
    '''
    def wilsonInt(ns, n, z = 1.96):
        pup = (ns + z**2/2)/(n+z**2) + z/(n+z**2) * np.sqrt((ns*(n-ns))/n + z**2/4)
        pdown = (ns + z**2/2)/(n+z**2) - z/(n+z**2) * np.sqrt((ns*(n-ns))/n + z**2/4)
        return pup, pdown
    if not prevPoints is None:
        inputNames = list(ems[0].ranges.keys())
        prevInputs = prevPoints.loc[:,inputNames]
        newInputs = newPoints.loc[:,inputNames]
        prevVol = np.prod(prevInputs.max() - prevInputs.min())
        newVol = np.prod(newInputs.max() - newInputs.min())
        volRatio = max(newVol/prevVol,1)
    else:
        volRatio = None
    acceptedPoints = nthImplausible(ems, prevPoints, targets, cutoff = 3)
    upperPercent, lowerPercent = wilsonInt(sum(acceptedPoints), np.shape(prevPoints)[0])
    if not volRatio is None:
        upperPercent = max(upperPercent, volRatio)
    return lowerPercent, upperPercent
