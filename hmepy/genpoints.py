import numpy as np
import pandas as pd
#pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib.pyplot as plt
import random
import math
import statistics
import warnings
from scipy.stats import qmc, multivariate_normal
from hmepy.utils import *
from hmepy.emutils import *
from hmepy.implausibility import *

class OptionDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

## Helper for generating random numbers from a uniform sphere
def runifs(n, d, c = None, r = 1):
    if c is None:
        c = np.zeros(d)
    initialPoints = np.random.normal(0, 1, (n,d))
    expVals = np.random.exponential(1, n)
    denoms = np.sqrt(expVals + np.apply_along_axis(lambda x: np.sum(x**2), 1, initialPoints))
    swept = np.apply_along_axis(lambda x: x/denoms, 0, initialPoints)
    return np.apply_along_axis(lambda x: x+c, 1, swept*r)

## Helper for indicator function for points relative to uniform sphere
def punifs(x, c = None, r = 1):
    if c is None:
        c = np.zeros(len(x))
    testVal = np.sum((x-c)**2)/r**2
    if testVal <= 1: return 1
    return 0

## Function for checking that points are in range
def inRange(data, ranges):
    lowerBounds = [ranges[rkey][0] for rkey in list(ranges.keys())]
    upperBounds = [ranges[rkey][1] for rkey in list(ranges.keys())]
    isBigger = np.apply_along_axis(lambda x: all([x[i] >= lowerBounds[i] for i in range(len(x))]), 1, data)
    isSmaller = np.apply_along_axis(lambda x: all([x[i] <= upperBounds[i] for i in range(len(x))]), 1, data)
    return isBigger & isSmaller

## Function to obtain an approximate maximin sample from a set of proposed points
def maximinSample(points, n, reps = 1000):
    sampleList = np.zeros((reps, n))
    measureList = np.zeros(reps)
    for i in range(reps):
        sampleIndex = random.sample(range(np.shape(points)[0]), n)
        sampledVals = points.iloc[sampleIndex,:]
        measure = [math.dist(sampledVals.iloc[i,:], sampledVals.iloc[j,:]) for i in range(sampledVals.shape[0]) for j in np.arange(start = i+1, stop = sampledVals.shape[1])]
        sampleList[i,:] = sampleIndex
        measureList[i] = min(measure)
    bestIndices = sampleList[np.argmax(measureList),:]
    return points.iloc[[int(a) for a in bestIndices],:]

## Function to perform pca transformation of points x based on a sample of points sPoints.
def pcaTransform(x, sPoints, forward = True):
    xnames = x.columns
    spMean = np.apply_along_axis(statistics.mean, 0, sPoints)
    spSD = np.apply_along_axis(statistics.stdev, 0, sPoints)
    sTraf = np.apply_along_axis(lambda y: (y - spMean)/spSD, 1, sPoints)
    sEvals, sEvecs = np.linalg.eigh(np.cov(sTraf, rowvar = False))
    sEvals = np.real_if_close(sEvals).astype(float)
    sEvecs = np.real_if_close(sEvecs).astype(float)
    sEvals[sEvals < 1e-10] = 1e-10
    if forward:
        xMean = np.apply_along_axis(statistics.mean, 0, x)
        xSD = np.apply_along_axis(statistics.stdev, 0, x)
        xNew = np.apply_along_axis(lambda y: (y - xMean)/xSD, 1, x)
        traf = np.matmul(xNew, np.matmul(sEvecs, np.diag(1/np.sqrt(sEvals))))
        return pd.DataFrame(traf, columns = xnames)
    preTraf = np.matmul(x, np.matmul(np.diag(np.sqrt(sEvals)), sEvecs.T))
    traf = np.apply_along_axis(lambda y: y*spSD+spMean, 1, preTraf)
    return pd.DataFrame(traf, columns = xnames)

def generateNewDesign(ems, nPoints, z, method = "default", cutoff = 3,
                      plausibleSet = None, verbose = False, opts = None,
                      **kwargs):
    '''
    Generate Proposal Points

    Given a set of trained emulators, this function finds the next set of points that
    will be informative for a subsequent wave of emulation or, in the event that the
    current wave is the last desired, a set of points that 'optimally' span the parameter
    region of interest. There are a number of different methods that can be utilised,
    alone or in combination with one another, to generate the points.

    If the method argument contains 'lhs', a large Latin hypercube is generated and
    non-implausible points from this design are retained. If this rejection step generates
    a sufficiently large number of points, then further arguments to method are ignored and
    the design returned is a maximin subset of those non-implausible points.

    If the method argument contains 'line', then line sampling is performed. Given an
    already established collection of non-implausible points, rays are drawn between pairs
    of points (selected so as to maximise the distance between them) and more points are
    sampled along those rays. Points thus sampled are retained if they lie on (or close to) the
    boundary of the non-implausible space, or on the boundary of the parameter region of
    interest.

    If the method argument contains 'importance', then importance sampling is performed.
    Given a collection of non-implausible points, a mixture distribution of either
    multivariate normal or uniform ellipsoid proposals around the current non-implausible
    set are constructed. The optimal standard deviation (in the normal case) or optimal radius
    (for the ellipsoids) is determined using a burn-in phase, and points are proposed until
    the desired number of non-implausible points have been found.

    If the method argument contains 'slice', then slice sampling is performed. Given at least
    one non-implausible point, a minimum enclosing hyperrectangle (MEH), perhaps after transformation
    of the space using PCA, is determined and points are sampled for each dimension of the
    parameter space uniformly, shrinking the MEH as appropriate. This method is akin to a Gibbs
    sampler.

    For any sampling strategy, the parameters ems, nPoints, and z must be provided, corresponding
    to the emulators, number of points desired, and observations respectively. All methods rely on
    a means of assessing point suitability, which we refer to generally as an implausibility measure.
    By default, this function uses nth-maximum implausibility as provided by nthImplausible; a user-
    defined method can be provided instead by supplying the function call to opts['accept_measure'].
    Any such function must take, as a minimum, four arguments: the emulators, the points, the targets,
    and a cutoff. Note that, in accordance with the default behaviour of nthImplausible, if more than
    10 distinct observations are being emulated and matched and an explicit opts['nth'] argument is
    not supplied, then second-maximum implausibility is used as the measure.

    The option opts['seek'] determines how many (if any) points should be chosen that have a higher
    probability of matching targets, as opposed to not missing targets (as is the usual rationale).
    Due to the danger of such an approach, this value should not be too high and should be used sparingly
    at early waves; even at later waves, it is inadvisable to seek more than 10% of the output points
    using this measure. The default is opts['seek'] = 0; the parameter may be provided either as a
    percentage of points desired (in the range [0,1]) or a fixed number of points.

    The default behaviour is as follows. A set of initial points are generated from a large LHD; line
    sampling is performed to identify the boundaries of the non-implausible region; then importance
    sampling is used to fill out the space. The proposed set of points is thinned and both line and
    importance sampling are performed once more; this resampling behaviour aims to ensure uniformity
    of the proposal and the number of resampling steps is controlled by the value of opts['resample'].
    A specification of opts['resample'] = n indicates that n+1 proposal stages will occur, including
    the initial sampling stage.

    In regions where the non-implausible space at a given cutoff value is very hard to find, the point
    proposal will begin at a higher cutoff value than specified to try to find an adequate space-filling
    design. Given such a design (including the use of any methods such as line and importance sampling),
    the function subselects to a lower cutoff value by demanding some percentage of the proposed points
    are retained. The process is repeated at this lower cutoff until either the space cannot be subselected
    to obtain a lower cutoff, or the desired cutoff is attained. This 'laddering' approach is controlled
    by the parameters opts['cutoff_tolerance'] and opts['ladder_tolerance'] to determine the required
    closeness to the desired cutoff and minimum improvement required to continue the process. For instance,
    opts = {..., 'cutoff_tolerance': 0.01, 'ladder_tolerance': 0.1} terminate the ladder process if consecutive
    'rungs' are within 0.1 of one another, or if a ladder cutoff is within 0.01 of the desired cutoff.

    These methods will work slowly, if at all, when the target space is extremely small in comparison with
    the initial non-yet-ruled-out (NROY) space; it may also fail to give a representative sample if the
    target space is formed of disconnected regions of vastly different volumes. In such cases, other methods
    may be more appropriate for finding non-implausible points.

    Arguments within opts
    ---------------------
    accept_measure: str | func
        The 'implausibility' measure to be used; if "default", uses nthImplausible.
    cluster: bool
        Whether output clustering should be used to propose points in the LHD.
    cutoff_tolerance: float
        The closeness to desired cutoff such that a ladder process will terminate.
    nth: int
        The level of nth-implausibility to consider, if using nthImplausible.
    resample: int
        The number of times to perform thin-and-resample.
    seek: int | float
        How many 'good' points should be sought, either as an absolute number or a proportion.
    points.factor: int
        In LHS, how many more points to propose for rejection sampling.
    pca_lhs: bool
        In LHS, whether to apply a PCA transform before proposing.
    n_lines: int
        In Line, how many lines to draw in the space.
    ppl: int
        In Line, how many points to sample along each line.
    imp_distro: str
        In Importance, whether the mixture distribution consists of normal or uniform spherical proposals.
    imp_scale: float
        In Importance, the standard deviation or radius respectively of proposals.
    pca_slice: bool
        In Slice, whether to apply PCA transformation to the parameter space before applying slice sampling.
    seek_distro: str
        In Seek, the distribution to use when looking for 'good' points.

    Parameters
    ----------
    ems: [Emulator]
        A list of Emulator objects trained on previous design points.
    nPoints: int
        The number of points desired in the new proposal.
    z: [Dict] | [List]
        The target values, in {"val": ..., "sigma": ...} or [upper, lower] form.
    method: [str]
        The methods to apply. If default, method = ['lhs', 'line', 'importance'].
    cutoff: float
        The cutoff at which a point is considered non-implausible.
    plausibleSet: DataFrame
        An optional set of known non-implausible points, to avoid LHD sampling.
    verbose: bool
        Should progress statements be printed to console?
    opts: dict
        A named list of options, from the set detailed above.
    
    Returns
    -------
    A DataFrame containing the set of new points upon which to run the model.

    '''
    
    if opts is None:
        opts = OptionDict(kwargs)
    else:
        opts = OptionDict(kwargs | opts)
    if "cutoffInfo" in opts and "cutoff_info" not in opts:
        opts["cutoff_info"] = opts["cutoffInfo"]
    if not hasattr(opts, 'accept_measure'): opts['accept_measure'] = "default"
    if hasattr(opts, "cluster"):
        if opts['cluster'] == 1: opts['cluster'] = True
        if not isinstance(opts['cluster'], bool): opts['cluster'] = False
    if not hasattr(opts, 'cluster'): opts['cluster'] = False
    if not hasattr(opts, 'use_collect'): opts['use_collect'] = True
    if not isinstance(opts['use_collect'], bool): opts['use_collect'] = True
    if not hasattr(opts, 'cutoff_tolerance'): opts['cutoff_tolerance'] = 0.01
    if not isinstance(opts['cutoff_tolerance'], float):
        try: opts['cutoff_tolerance'] = float(opts['cutoff_tolerance'])
        except: opts['cutoff_tolerance'] = 0.01
    if not hasattr(opts, 'ladder_tolerance'): opts['ladder_tolerance'] = 0.1
    else:
        try: opts['ladder_tolerance'] = float(opts['ladder_tolerance'])
        except: opts['ladder_tolerance'] = 0.1
    if not (hasattr(opts, 'nth')): opts['nth'] = None
    else:
        try: opts['nth'] = int(opts['nth'])
        except: opts['nth'] = 1
    if not hasattr(opts, 'resample'): opts['resample'] = 1
    else:
        try: opts['resample'] = int(opts['resample'])
        except: opts['resample'] = 1
    if not hasattr(opts, 'seek'): opts['seek'] = 0
    else:
        try: opts['seek'] = float(opts['seek'])
        except: opts['seek'] = 0
    if opts['seek'] > nPoints:
        opts['seek'] = nPoints
    if not hasattr(opts, 'cutoff_info'): minCut = cutoff
    else: minCut = opts['cutoff_info'][0]
    if hasattr(opts, 'use_collect') and opts['use_collect']:
        ems = collectEmulators(ems)
    theseRanges = getRanges(ems)
    range_names = list(theseRanges.keys())
    if opts['nth'] is None:
        nems = len(np.unique([e.outputName for e in ems]))
        if nems > 10: opts['nth'] = 2
        else: opts['nth'] = 1
    if isinstance(opts['accept_measure'], str) and opts['accept_measure'] == 'default':
        iFunc = lambda ems, x, z, **kwargs: nthImplausible(ems, x, z, n = opts['nth'], **kwargs)
    if isinstance(method, str) and method == "default":
        method = ['lhs', 'line', 'importance']
        userSelect = False
    else:
        userSelect = True
    possibleMethods = ['lhs', 'line', 'importance', 'slice', 'optical']
    whichMethods = [pmeth for pmeth in possibleMethods if pmeth in method]
    nCurrent = 0
    if plausibleSet is None or 'lhs' in whichMethods:
        if verbose: print("Proposing from LHS...")
        if not opts['cluster']:
            lhGen = lhsGen(ems, theseRanges, max(nPoints, 10*len(theseRanges)),
                           z, cutoff, verbose, opts)
            points = lhGen['points']
            thisCutoff = lhGen['cutoff']
        else:
            ## Fill this in.
            lhGen = None
        metaVerbose = True
        if thisCutoff == cutoff and np.shape(points)[0] >= 0.25*nPoints and not(userSelect):
            if verbose: print("LHS has high yield: no other methods required.")
            metaVerbose = False
            while np.shape(points)[0] < nPoints:
                if not opts['cluster']:
                    lhg = lhsGen(ems, theseRanges, max(nPoints, 10*len(theseRanges)),
                                z, cutoff, verbose, opts)
                    newPoints = lhg['points']
                    thisCutoff = lhg['cutoff']
                else:
                    ## Do a thing.
                    lhg = None
                points = pd.concat([points, newPoints])
        if np.shape(points)[0] >= nPoints and thisCutoff == cutoff:
            if verbose and metaVerbose: print("Enough points generated from LHD - no further methods required.")
            if opts['seek'] > 0:
                # Do a thing
                pass
            if np.shape(points)[0] > nPoints - opts['seek']:
                if verbose: print("Selecting final points using maximin criterion...")
                points = maximinSample(points, nPoints - opts['seek'])
            return points
    else:
        plausibleSet = plausibleSet.loc[:,list(theseRanges.keys())]
        if isinstance(opts['accept_measure'], str) and opts['accept_measure'] == "default":
            pointImps = nthImplausible(ems, plausibleSet, z, maxImp = math.inf)
            sortedImps = np.sort(pointImps)
            optimalCut = sortedImps[min([len(pointImps)-1, math.floor(0.8*len(pointImps)), 5*len(theseRanges)])]
            isAsymp = (optimalCut > cutoff and (optimalCut - sortedImps[0]) < opts['cutoff_tolerance'])
        else:
            if hasattr(opts, 'cutoff_info'):
                cDeets = opts['cutoff_info']
                cutoffSequence = [cutoff]
                cExtra = [((cutoff+cDeets)*(20-i) + 2*i*cDeets[1])/40 for i in range(0, 21)]
                cutoffSequence.append(cExtra)
            else:
                cutoffSequence = np.linspace(cutoff, 20, retstep = 20)
            pointsAccept = [0]
            optimalCut = cutoff
            requiredPoints = min(max(1, np.shape(plausibleSet)[0]-1, math.floor(0.8*np.shape(plausibleSet)[0])), 5*len(theseRanges))
            whichMatch = np.full(np.shape(plausibleSet)[0], False, dtype = 'bool')
            for i in cutoffSequence:
                cBools = opts['accept_measure'](ems, plausibleSet.iloc[not whichMatch,:], z, cutoff = i, n = opts['nth'])
                whichMatch[not whichMatch] = cBools
                pointsAccept.append(sum(whichMatch))
                if sum(whichMatch) >= requiredPoints:
                    optimalCut = i
                    break
            minCut = cutoffSequence[[i for i in range(len(pointsAccept)) if not pointsAccept[i] == 0][0]-1]
            isAsymp = (optimalCut > cutoff and (pointsAccept[-2] == 0 or pointsAccept[-1] >= nPoints))
        if isAsymp and optimalCut-cutoff > opts['cutoff_tolerance']:
            if verbose:
                print("Point proposal seems to be asymptoting around cutoff " + str(round(optimalCut, 3)) + " - terminating.")
            if isinstance(opts['accept_measure'], str) and opts['accept_measure'] == "default":
                return {'points': plausibleSet.iloc[pointImps <= optimalCut,:], 'cutoff': optimalCut}
            gPoints = plausibleSet.iloc[opts['accept_measure'](ems, plausibleSet, z, cutoff = optimalCut),:]
            return {'points': gPoints, 'cutoff': optimalCut}
        if optimalCut < cutoff: thisCutoff = cutoff
        else: thisCutoff = round(optimalCut, 3)
        if isinstance(opts['accept_measure'], str) and opts['accept_measure'] == "default":
            points = plausibleSet.iloc[pointImps <= thisCutoff,:]
        else:
            points = plausibleSet.iloc[opts['accept_measure'](ems, plausibleSet, z, n = opts['nth'], cutoff = thisCutoff),:]
    nCurrent = np.shape(points)[0]
    if nCurrent is None or nCurrent == 0:
        warnings.warn("No non-implausible points found from initial step.")
        return points
    if verbose:
        print(str(nCurrent) + " initial valid points generated for I=" + str(round(thisCutoff, 3)))
    if "line" in whichMethods and nCurrent < nPoints:
        if verbose: print("Performing line sampling...")
        points = lineSample(ems, theseRanges, z, points, cutoff = thisCutoff, opts = opts)
        if verbose: print("Line sampling generated " + str(np.shape(points)[0] - nCurrent) + " more points.")
        nCurrent = np.shape(points)[0]
    if "slice" in whichMethods and nCurrent < nPoints:
        if verbose: print("Performing slice sampling...")
        points = sliceSample(ems, theseRanges, nPoints, z, points, cutoff = thisCutoff, opts = opts)
        if verbose: print("Slice sampling generated " + str(np.shape(points)[0] - nCurrent) + " more points.")
        nCurrent = np.shape(points)[0]
    if "importance" in whichMethods and nCurrent < nPoints:
        if verbose: print("Performing importance sampling...")
        points = importanceSample(ems, nPoints, z, points, cutoff = thisCutoff, opts = opts)
        if verbose: print("Importance sampling generated " + str(np.shape(points)[0] - nCurrent) + " more points.")
        nCurrent = np.shape(points)[0]
    if thisCutoff - cutoff > opts['cutoff_tolerance']:
        if isinstance(whichMethods, str) and whichMethods == "lhs":
            if verbose: print("Could not generate points to desired cutoff.")
            return points
        newOpts = OptionDict(opts.copy())
        newOpts['chain_call'] = True
        newOpts['resample'] = 0
        points = generateNewDesign(ems, nPoints, z, [meth for meth in whichMethods if not(meth == "lhs")],
                               cutoff = cutoff, plausibleSet = points, verbose = verbose,
                               opts = newOpts, cutoffInfo = [minCut, thisCutoff])
    else:
        if thisCutoff != cutoff:
            if verbose: print("Point implausibilities within tolerance; proposed points have maximum implausibility " + str(round(thisCutoff, 4)))
    if isinstance(points, dict) and "points" in points and "cutoff" in points:
        cutoff = points['cutoff']
        points = points['points']
    if isinstance(points, pd.DataFrame) and points.shape[1] == len(range_names):
        points = points.copy()
        points.columns = range_names
    if opts['resample'] > 0 and len(list(set(whichMethods).intersection({"importance", "line", "slice"}))) > 0:
        for nsamp in range(opts['resample']):
            points = maximinSample(points, min(np.shape(points)[0], math.ceil(nPoints/2)))
            nCurrent = np.shape(points)[0]
            if verbose: print("Resample " + str(nsamp + 1))
            if "line" in whichMethods:
                if verbose: print("Performing line sampling...")
                points = lineSample(ems, theseRanges, z, points, cutoff = cutoff, opts = opts)
                if verbose: print("Line sampling generated " + str(np.shape(points)[0] - nCurrent) + " more points.")
                nCurrent = np.shape(points)[0]
            if "slice" in whichMethods and nCurrent < nPoints:
                if verbose: print("Performing slice sampling...")
                points = sliceSample(ems, theseRanges, nPoints, z, points, cutoff = cutoff, opts = opts)
                if verbose: print("Slice sampling genertaed " + str(np.shape(points)[0] - nCurrent) + " more points.")
                nCurrent = np.shape(points)[0]
            if "importance" in whichMethods and nCurrent < nPoints:
                if verbose: print("Performing importance sampling...")
                points = importanceSample(ems, nPoints, z, points, cutoff = cutoff, opts = opts)
                if verbose: print("Importance sampling generated " + str(np.shape(points)[0] - nCurrent) + " more points.")
                nCurrent = np.shape(points)[0]
    if opts['seek'] > 0:
        ## Do a thing
        pass
    if np.shape(points)[0] > nPoints:
        if verbose: print("Selecting final points using maximin criteriion...")
        points = maximinSample(points, nPoints)
    if isinstance(points, pd.DataFrame) and points.shape[1] == len(range_names):
        points = points.copy()
        points.columns = range_names
    if hasattr(opts, 'chain_call') and opts['chain_call']:
        return {'points': points, 'cutoff': cutoff}
    return points


def lhsGen(ems, ranges, nPoints, z, cutoff = 3, verbose = False, opts = None):
    if opts is None:
        opts = {'points_factor': 40, 'pca_lhs': False}
    try: opts['points_factor'] = int(opts['points_factor'])
    except: opts['points_factor'] = 40
    if not hasattr(opts, 'pca_lhs'): opts['pca_lhs'] = False
    if not isinstance(opts['pca_lhs'], bool): opts['pca_lhs'] = False
    lhSampler = qmc.LatinHypercube(len(ranges))
    if opts['pca_lhs']:
        trainPoints = ems[0].inData
        initRanges = ems[0].ranges
        actualPoints = evalFuncs(scaleInput, trainPoints, initRanges, False)
        pcaPoints = pcaTransform(actualPoints, actualPoints)
        pcaRanges = dict(zip(pcaPoints.columns, [[0.9*np.min(pcaPoints.iloc[:,i]), 1.1*np.max(pcaPoints.iloc[:,i])] for i in range(np.shape(pcaPoints)[1])]))
        pcaLH = lhSampler.random(n = nPoints * opts['points_factor'])
        pcaLH = pd.DataFrame(qmc.scale(pcaLH, [ran[0] for ran in list(pcaRanges.values())], [ran[1] for ran in list(pcaRanges.values())]),
                             columns = ranges.keys())
        points = pcaTransform(pcaLH, actualPoints, False)
        points = points.iloc[inRange(points, initRanges),:]
    else:
        tempLH = lhSampler.random(n = nPoints * opts['points_factor'])
        points = pd.DataFrame(qmc.scale(tempLH, [ran[0] for ran in list(ranges.values())], [ran[1] for ran in list(ranges.values())]),
                              columns = ranges.keys())
    if isinstance(opts['accept_measure'], str) and opts['accept_measure'] == "default":
        pointImps = nthImplausible(ems, points, z, maxImp = math.inf)
        requiredPoints = min(max(len(pointImps)-1, math.floor(0.8*len(pointImps))), 5*len(ranges))
        if sum(pointImps <= cutoff) < requiredPoints:
            sortedImps = np.sort(pointImps)
            if sortedImps[0] >= 20:
                warnings.warn("""Parameter space has no points below implausibility 20;
                              terminating early. This may not indicate model inadequacy,
                              inspect results and re-run if applicable.""")
                return {"points": points.iloc[pointImps <= 0,:], "cutoff": 0}
            cutoffCurrent = np.sort(pointImps)[requiredPoints-1]
        else:
            cutoffCurrent = cutoff
        return {"points": points.iloc[pointImps <= cutoffCurrent,:], 'cutoff': cutoffCurrent}
    else:
        ibools = np.full(np.shape(points)[0], False, dtype = 'bool')
        requiredPoints = min(max(1, np.shape(points)[0]-1, math.floor(0.8*np.shape(points)[0])), 5*len(ranges))
        cCurrent = cutoff
        while sum(ibools) < requiredPoints:
            if cCurrent > 20:
                warnings.warn("""Parameter space has no points below implausibility 20;
                              terminating early. This may not indicate model inadequacy,
                              inspect results and re-run if applicable.""")
                return {'points': points.iloc[np.full(np.shape(points)[0], False, dtype = 'bool'),:], 'cutoff': 0}
            falseIndices = [i for i in range(len(ibools)) if not ibools[i]]
            impBools = opts['accept_measure'](ems, points.iloc[falseIndices,:], z, cutoff = cCurrent, n = opts['nth'])
            ibools[falseIndices] = ibools[falseIndices] | impBools
            cCurrent = cCurrent + 0.5
        return {'points': points.iloc[ibools,:], 'cutoff': cCurrent-0.5}
            
def lineSample(ems, ranges, z, sPoints, cutoff = 3, opts = None):
    if opts is None: opts = {'n_lines': 20, 'ppl': 50}
    if not hasattr(opts, 'n_lines'): opts['n_lines'] = 20
    try: opts['n_lines'] = int(opts['n_lines'])
    except: opts['n_lines'] = 20
    if not hasattr(opts, 'ppl'): opts['ppl'] = 50
    try: opts['ppl'] = int(opts['ppl'])
    except: opts['ppl'] = 50
    if np.shape(sPoints)[0] < 2: return(sPoints)
    nLines = min(np.shape(sPoints)[0]*(1+np.shape(sPoints)[0])/2, opts['n_lines'])
    if opts['ppl'] % 4 == 1: opts['ppl'] = opts['ppl']+1
    def genLines(data):
        pointInds = random.sample(range(np.shape(data)[0]), 2)
        p1 = data.iloc[pointInds[0],:]
        p2 = data.iloc[pointInds[1],:]
        d = math.sqrt(sum((p1-p2)**2))
        return [p1, p2, d]
    sLines = [genLines(sPoints) for i in range(10*nLines)]
    distList = pd.DataFrame(data = {'d': [line[2] for line in sLines]})
    distDup = list(distList[distList.duplicated()].index)
    sLines = [sLines[i] for i in range(len(sLines)) if not i in distDup]
    sortedIndices = np.argsort([line[2] for line in sLines])[::-1]
    sortedLines = [sLines[i] for i in sortedIndices][0:opts['n_lines']]
    def sampPts(line):
        x0, x1, d = line
        try:
            tempArr = np.empty((opts['ppl']+1, np.shape(x0)[0]))
            for i in range(opts['ppl']+1):
                dVal = i*2*d/opts['ppl']-d
                tempArr[i,:] = list((x0+x1)/2 + dVal * (x1-x0))
            return pd.DataFrame(tempArr, index = range(opts['ppl']+1), columns = x0.index)
        except:
            return None
    propPoints = [sampPts(line) for line in sortedLines]
    propPoints = [pPoint for pPoint in propPoints if not pPoint is None]
    for i in range(len(propPoints)):
        pp = propPoints[i]
        isInRange = inRange(pp, ranges)
        propPoints[i] = pp.iloc[isInRange]
    propPoints = [pPoint for pPoint in propPoints if np.shape(pPoint)[0] > 0]
    if len(propPoints) == 0:
        return sPoints
    if isinstance(opts['accept_measure'], str) and opts['accept_measure'] == "default":
        imps = [nthImplausible(ems, sp, z, n = opts['nth'], cutoff = cutoff) for sp in propPoints]
    else:
        imps = [opts['accept_measure'](ems, sp, z, cutoff = cutoff, n = opts['nth']) for sp in propPoints]
    def findBoundaryPoints(points, implaus):
        isAccept = [implaus[j] and (j == 0 or j == len(implaus)-1 or (not implaus[j-1]) or (not implaus[j+1])) for j in range(len(implaus))]
        return points.iloc[isAccept]
    acceptPoints = [findBoundaryPoints(propPoints[i], imps[i]) for i in range(len(propPoints))]
    outPoints = pd.concat(acceptPoints)
    return pd.concat([sPoints, outPoints]).drop_duplicates()

def importanceSample(ems, nPoints, z, sPoints, cutoff = 3, opts = None):
    if opts is None:
        opts = {'imp_distro': 'sphere', 'imp_scale': 2}
    if not hasattr(opts, 'imp_distro'): opts['imp_distro'] = 'sphere'
    if not isinstance(opts['imp_distro'], str) or not(opts['imp_distro'] in ['sphere', 'normal']): opts['imp_distro'] = 'sphere'
    if not hasattr(opts, 'imp_scale'): opts['imp_scale'] = 2
    try: opts['imp_scale'] = float(opts['imp_scale'])
    except: opts['imp_scale'] = 2
    if np.shape(sPoints)[0] >= nPoints:
        return sPoints
    mPoints = nPoints - np.shape(sPoints)[0]
    ranges = getRanges(ems, False)
    newPoints = sPoints
    def propPoints(sp, sd, howmany = nPoints):
        spTrafo = pcaTransform(sp, sPoints)
        wp = spTrafo.iloc[random.choices(range(np.shape(spTrafo)[0]), k = howmany),:]
        pp = np.empty(np.shape(wp))
        if opts['imp_distro'] == "normal":
            for i in range(np.shape(pp)[0]):
                pp[i,:] = multivariate_normal.rvs(wp.iloc[i,:], sd, size = 1)
        else:
            for i in range(np.shape(pp)[0]):
                pp[i,:] = runifs(1, np.shape(pp)[1], wp.iloc[i,:], sd)
        pp = pd.DataFrame(pp, columns = wp.columns)
        backTraf = pcaTransform(pp, sPoints, False)
        ir = inRange(backTraf, ranges)
        if isinstance(opts['accept_measure'], str) and opts['accept_measure'] == "default":
            ni = nthImplausible(ems, backTraf, z, n = opts['nth'], cutoff = cutoff)
        else:
            ni = opts['accept_measure'](ems, backTraf, z, n = opts['nth'], cutoff = cutoff)
        valid = [ir[k] and ni[k] for k in range(len(ir))]
        if opts["imp_distro"] == "normal":
            tWeights = [1/(np.shape(spTrafo)[0]) * np.sum([multivariate_normal.pdf(pp.iloc[i,:], spTrafo.iloc[j,:], sd) for j in range(np.shape(spTrafo)[0])]) for i in range(np.shape(pp)[0])]
            spWeights = [1/(np.shape(spTrafo)[0]) * np.sum([multivariate_normal.pdf(pp.iloc[i,:], spTrafo.iloc[j,:], sd) for j in range(np.shape(spTrafo)[0])]) for i in range(np.shape(spTrafo)[0])]
            minWeight = np.min(spWeights)
            weights = minWeight/tWeights
        else:
            weights = [np.sum([punifs(pp.iloc[i,:], spTrafo.iloc[j,:], sd) for j in range(np.shape(spTrafo)[0])]) for i in range(np.shape(pp)[0])]
        allow = [np.random.uniform() <= 1 / float(weight) if float(weight) > 0 else False for weight in weights]
        accepted = [a and v for a, v in zip(allow, valid)]
        return backTraf.iloc[accepted,:]
    if opts['imp_distro'] == "normal" and not(len(np.shape(opts['imp_scale'])) == 2):
        sd = np.diag(np.full(len(ranges), opts['imp_scale'], dtype = 'float'))
    else:
        sd = opts['imp_scale']
    acceptRate = None
    upperAccept = 0.225
    lowerAccept = 0.1
    while (acceptRate is None or acceptRate > upperAccept or acceptRate < lowerAccept) and np.shape(newPoints)[0] < nPoints:
        if not (acceptRate is None):
            if acceptRate > upperAccept: sd = sd * 1.1
            else: sd = sd * 0.9
        howMany = max(math.floor(nPoints/4), 1000)
        pPoints = propPoints(sPoints, sd, howMany)
        newPoints = pd.concat([newPoints, pPoints]).drop_duplicates()
        acceptRate = np.shape(pPoints)[0]/howMany
    while np.shape(newPoints)[0] < nPoints:
        pPoints = propPoints(sPoints, sd, math.ceil(1.5*mPoints/acceptRate))
        newPoints = pd.concat([newPoints, pPoints]).drop_duplicates()
    return newPoints

def sliceSample(ems, ranges, nPoints, z, points, cutoff, opts):
    if opts is None:
        opts = {'pca_slice': False}
    if not hasattr(opts, 'pca_slice'): opts['pca_slice'] = False
    if not isinstance(opts['pca_slice'], bool): opts['pca_slice'] = False
    completePoints = points.copy()
    pcaBase = points.copy()
    if opts['pca_slice']:
        points = pcaTransform(points, pcaBase)
        pcaRanges = dict(zip(pcaBase.columns, [[-5,5] for i in range(len(np.shape(pcaBase)[1]))]))
    else:
        pcaRanges = ranges
    indexList = np.full(np.shape(points)[0], 0)
    rangeList = [list(pcaRanges.values())[j].copy() for j in indexList]
    while np.shape(completePoints)[0] < nPoints:
        oldVals = np.empty(np.shape(points)[0])
        for i in range(len(oldVals)):
            oldVals[i] = points.iloc[i, indexList[i]]
            points.iloc[i, indexList[i]] = float(np.random.uniform(rangeList[i][0], rangeList[i][1], 1))
        if opts['pca_slice']:
            testPoints = pcaTransform(points, pcaBase, False)
        else:
            testPoints = points
        if isinstance(opts['accept_measure'], str) and opts['accept_measure'] == 'default':
            imps = nthImplausible(ems, testPoints, z, n = opts['nth'], cutoff = cutoff)
        else:
            imps = opts['accept_measure'](ems, testPoints, z, n = opts['nth'], cutoff = cutoff)
        inr = inRange(points, ranges)
        for i in range(len(imps)):
            if imps[i] and inr[i]:
                if indexList[i] == len(pcaRanges)-1:
                    if opts['pca_slice']:
                        newPoint = pd.DataFrame(dict(zip(completePoints.columns, pcaTransform(points.iloc[i,:], pcaBase, False))), index = [0])
                        completePoints = pd.concat([completePoints, newPoint])
                    else:
                        newPoint = pd.DataFrame(dict(zip(completePoints.columns, points.iloc[i,:])), index = [0])
                        completePoints = pd.concat([completePoints, newPoint])
                    indexList[i] = 0
                else:
                    indexList[i] = indexList[i]+1
                rangeList[i] = list(pcaRanges.values())[indexList[i]].copy()
            else:
                if points.iloc[i, indexList[i]] < oldVals[i]:
                    rangeList[i][0] = float(points.iloc[i, indexList[i]])
                else:
                    rangeList[i][1] = float(points.iloc[i, indexList[i]])
                points.iloc[i, indexList[i]] = oldVals[i]
    return completePoints
