'''
Defines correlation functions and other functions required for
their use (eg distance functions)
'''

#%% Imports
import numpy as np
import pandas as pd
import math
from copy import deepcopy
from sklearn.metrics.pairwise import pairwise_distances

__all__ = [
    'expSq', 'matern', 'ornUhl',
    'gammaExp','ratQuad', 'expSqDiff',
    'maternDiff', 'ratQuadDiff', 'Correlator'
    ]

def getDist(df1, df2):
    '''
    Computes distances between points defined by two DataFrames.
    The DataFrames are such that each row corresponds to a single point.

    :param df1: The first collection of n points
    :param df2: The second collecton of m points
    :return: The (Euclidean) distance, as a numpy.array of size nxm
    '''
    if len(df1.columns.intersection(df2.columns)) != df1.shape[1]:
        raise TypeError("Dimensions of DataFrames do not match.")
    df2 = df2[list(df1.columns.values)]
    df1 = pd.DataFrame(np.real_if_close(df1.to_numpy()).astype(float), columns=df1.columns)
    df2 = pd.DataFrame(np.real_if_close(df2.to_numpy()).astype(float), columns=df2.columns)
    return np.transpose(pairwise_distances(df1, df2))

# def getDist(df1, df2):
#     '''
#     Computes distances between points defined by two DataFrames.
#     The DataFrames are such that each row corresponds to a single point.
    
#     :param df1: The first collection of n points
#     :param df2: The second collection of m points
#     :return: The distances, as a numpy.array of size nxm
#     '''

#     if len(df1.columns.intersection(df2.columns)) != df1.shape[1]:
#         raise TypeError("Dimensions of DataFrames do not match.")
#     df2 = df2[list(df1.columns.values)]
#     a1 = df1.to_numpy()
#     a2 = df2.to_numpy()
#     tempvec = np.array([math.dist(a1[0], a2[j]) for j in range(a2.shape[0])])
#     for i in range(1, a1.shape[0]):
#         tempvec = np.vstack((tempvec, [math.dist(a1[i], a2[j]) for j in range(a2.shape[0])]))
#     return np.transpose(np.array(tempvec.reshape(a1.shape[0], a2.shape[0])))

def outerDist(df1, df2, dimname):
    '''
    Gets the distance between all points in df1 and df2 in dimension dimname.

    :param df1: The first DataFrame
    :param df2: The second DataFrame
    :param dimname: The name of the variable to calculate along
    '''

    vec1 = df1[dimname]
    vec2 = df2[dimname]
    tempvec = np.array([vec1[0] - vec2[j] for j in range(len(vec2))])
    for i in range(1, len(vec1)):
        tempvec = np.vstack((tempvec, [vec1[i]-vec2[j] for j in range(len(vec2))]))
    return np.transpose(np.array(tempvec.reshape(len(vec1), len(vec2))))

def expSq(x, xp, hp = None):
    '''
    Exponential squared correlation function
    For points x, xp and a correlation length theta, gives the exponent
    of the squared (Euclidean) distance between x and xp, weighted by theta^2.
    
    :param x: A DataFrame of rows corresponding to position vectors
    :param xp: A DataFrame of rows corresponding to position vectors
    :param hp: The hyperparameter theta (correlation length) in a named Dict
    :return: The exponential-squared correlation between x and xp
    '''

    if hp is None:
        raise ValueError("No hyperparameters specified.")
    if not('theta' in hp):
        raise ValueError("Correlation length theta not specified.")
    if not(np.isscalar(hp['theta'])):
        if len(hp['theta']) == np.shape(x)[1]:
            dists = np.square(getDist(x / hp['theta'][np.newaxis, :],
                                      xp / hp['theta'][np.newaxis, :]))
            return np.exp(dists*-1)
        hp['theta'] = hp['theta'][0]     
    dists = np.square(getDist(x/hp['theta'], xp/hp['theta']))
    return np.exp(dists*-1)

def expSqDiff(x, xp, hp, xi, xpi = None):
    if hp is None:
        raise ValueError("No hyperparameters specified.")
    if not('theta' in hp):
        raise ValueError("Correlation length theta not specified.")
    if not(xi in x) or not(xi in xp):
        raise IndexError("Direction of differentiation not found in DataFrames.")
    if not(np.isscalar(hp['theta'])):
        if len(hp['theta']) == np.shape(x)[1]:
            thet1 = hp['theta'][x.columns.get_loc(xi)]
            if not(xpi is None):
                thet2 = hp['theta'][x.columns.get_loc(xpi)]
        else:
            thet1 = hp['theta'][0]
            if not(xpi is None):
                thet2 = hp['theta'][0]
    else:
        thet1 = hp['theta']
        if not(xpi is None):
            thet2 = hp['theta']
    diff1 = outerDist(x, xp, xi)
    if xpi is None:
        return -2*diff1*expSq(x, xp, hp)/thet1**2
    if xpi == xi:
        diff2 = diff1
    else:
        diff2 = outerDist(x, xp, xpi)
    basePart = -4/(thet1**2 * thet2**2) * diff1 * diff2 * expSq(x, xp, hp)
    if (xpi != xi):
        return basePart
    return basePart + 2/thet1**2 * expSq(x, xp, hp)
    
def matern(x, xp, hp = None):
    '''
    Matern correlation function
    For points x, xp, and a pair of hyperparameters nu and theta, gives
    the Matern correlation between the two points.

    At present, only half-integer arguments for nu are supported.

    :param x: A DataFrame of rows corresponding to position vectors
    :param xp: A DataFrame of rows corresponding to position vectors
    :param hp: The hyperparameters nu (smoothness) and theta (correlation length) in a named Dict
    :return: The Matern correlation between x and xp.
    '''

    if hp is None:
        raise ValueError("Hyperparameters not specified.")
    if not('theta' in hp):
        raise ValueError("Correlation length theta not specified.")
    if not('nu' in hp):
        raise ValueError("Smoothness parameter nu not specified.")
    if math.floor(hp['nu']) == hp['nu'] or math.floor(2*hp['nu']) != 2*hp['nu']:
        raise ArithmeticError("Smoothness parameter nu must be half-integer.")
    p = int(hp['nu']-0.5)
    d = getDist(x, xp)
    dPart = d*math.sqrt(2*p+1)/hp['theta']
    expPart = np.exp(dPart*-1)
    pSum = sum([math.factorial(p+i)/(math.factorial(i)*math.factorial(p-i))*((dPart*2)**(p-i)) for i in range(p+1)])
    return np.multiply(pSum * expPart, math.factorial(p)/math.factorial(2*p))

def maternDiff(x, xp, hp, xi, xpi = None):
    if hp is None:
        raise ValueError("Hyperparameters not specified.")
    if not('theta' in hp):
        raise ValueError("Correlation length theta not specified.")
    if not ('nu' in hp):
        raise ValueError("Smoothness parameter nu not specified.")
    if math.floor(hp['nu']) == hp['nu'] or math.floor(2*hp['nu']) != 2*hp['nu']:
        raise ArithmeticError("Smoothness parameter nu must be half-integer.")
    if math.floor(hp['nu']) < 1:
        raise ArithmeticError("Cannot differentiate non-differentiable function.")
    if math.floor(hp['nu']) < 2 and not(xpi is None):
        raise ArithmeticError("Cannot differentiate a once-differentiable function twice.")
    p = int(hp['nu']-0.5)
    innerArg = math.sqrt(2*p+1) * getDist(x, xp)/hp['theta']
    diff1 = -1*outerDist(x, xp, xi)
    if xpi is None:
        nonSum = -4*hp['nu']/hp['theta']**2 * diff1 * math.factorial(p)/math.factorial(2*p) * np.exp(-1*innerArg)
        hasSum = sum([math.factorial(p+i-1)/(math.factorial(i)*math.factorial(p-1-i))*((2*innerArg)**(p-i-1)) for i in range(p)])
        return nonSum*hasSum
    if xpi == xi:
        extraBitNoSum = 4*hp['nu']/hp['theta']**2 * math.factorial(p)/math.factorial(2*p) * np.exp(-1*innerArg)
        extraBitSum = sum([math.factorial(p+i-1)/(math.factorial(i)*math.factorial(p-i-1)) * ((2*innerArg)**(p-i-1)) for i in range(p)])
        extraBit = extraBitNoSum*extraBitSum
    else:
        extraBit = 0
    diff2 = -1*outerDist(x, xp, xpi)
    nonSum = -16*hp['nu']**2/hp['theta']**4 * diff1 * diff2 * math.factorial(p)/math.factorial(2*p) * np.exp(-1*innerArg)
    hasSum = sum([math.factorial(p-2+i)/(math.factorial(i)*math.factorial(p-2-i)) * (2*innerArg)**(p-2-i) for i in range(p-1)])
    return(extraBit + nonSum*hasSum)

def ornUhl(x, xp, hp = None):
    '''
    Ornstein-Uhlenbeck correlation function

    For points x, xp, and a hyperparameter theta, gives the
    Ornstein-Uhlenbeck correlation between the two points.

    This correlation function can be seen as a specific case of the
    Matern correlation function when nu = 1/2.

    :param x: A DataFrame of rows corresponding to position vectors
    :param xp: A DataFrame of rows corresponding to position vectors
    :param hp: The hyperparameter theta (correlation length) in a named Dict
    :return: The Ornstein-Uhlenbeck correlation between the points.
    '''

    if hp is None:
        raise ValueError("Hyperparameter not specified.")
    if not('theta' in hp):
        raise ValueError("Correlation length theta not specified.")
    dists = getDist(x, xp)
    return np.exp(dists*(-1/hp['theta']))

def gammaExp(x, xp, hp = None):
    '''
    Gamma-Exponential correlation function

    For points x, xp, and a pair of hyperparameters gamma and theta,
    gives the gamma-exponential correlation between the two points.

    The gamma-exponential correlation function, for d=|x-x'|, is given
    by exp(-(d/theta)^gamma). Gamma must be between 0 (exclusive) and 2 (inclusive).

    :param x: A DataFrame of rows corresponding to position vectors
    :param xp: A DataFrame of rows corresponding to position vectors
    :param hp: The hyperparameters theta (correlation length) and gamma (exponent) in a named list
    :return: The gamma-exponential correlation between x and xp.
    '''

    if hp is None:
        raise ValueError("Hyperparameters not specified.")
    if not('theta' in hp):
        raise ValueError("Correlation length theta not specified.")
    if not('gamma' in hp):
        raise ValueError("Exponent gamma not specified.")
    if hp['gamma'] <= 0 or hp['gamma'] > 2:
        raise ArithmeticError("Exponent must be between 0 (exclusive) and 2 (inclusive).")
    dists = getDist(x, xp)
    return np.exp(-1*np.power(dists*(1/hp['theta']), hp['gamma']))

def ratQuad(x, xp, hp = None):
    '''
    Rational Quadratic correlation function

    For points x, xp, and a pair of hyperparameters alpha and theta,
    gives the rational quadratic correlation between the two points.

    This correlation function, for d=|x-x'|, has the form (1+d^2/(2*alpha*theta^2))^(-alpha),
    and can be seen as a superposition of exponential-squared correlation
    functions.

    :param x: A DataFrame of rows corresponding to position vectors
    :param xp: A DataFrame of rows corresponding to position vectors
    :param hp: The hyperparameters alpha (exponent and scale) and theta (correlation length) as a named list
    :return: The rational quadratic correlation between x and xp.
    '''

    if hp is None:
        raise ValueError("Hyperparameters not specified.")    
    if not('theta' in hp):
        raise ValueError("Correlation length theta not specified.")
    if not('alpha' in hp):
        raise ValueError("Exponent alpha not specified.")
    dists = np.square(getDist(x, xp))
    return np.power(1+dists/(2*hp['alpha']*hp['theta']**2), -hp['alpha'])

def ratQuadDiff(x, xp, hp, xi, xpi = None):
    if hp is None:
        raise ValueError("Hyperparameters not specified.")    
    if not('theta' in hp):
        raise ValueError("Correlation length theta not specified.")
    if not('alpha' in hp):
        raise ValueError("Exponent alpha not specified.")
    dists = getDist(x, xp)**2
    diff1 = -1*outerDist(x, xp, xi)
    if xpi is None:
        return -diff1/hp['theta']**2 * (1+dists/(2*hp['alpha']*hp['theta']**2))**(-hp['alpha']-1)
    diff2 = -1*outerDist(x, xp, xpi)
    if xi == xpi:
        extraBit = (1+dists/(2*hp['alpha']*hp['theta']**2))**(-hp['alpha']-1)
    else:
        extraBit = 0
    return -(hp['alpha']+1)/hp['alpha'] * diff1 * diff2/hp['theta']**4 * (1+dists/(2*hp['alpha']*hp['theta']**2))**(-hp['alpha']-2) + extraBit

class Correlator:
    """
    Correlator Object

    A correlator has three main elements: the type of correlation function,
    the associated hyperparameters, and the nugget term. The nugget term is
    broadly separate from the other two parameters, being type-independent.

    Options for Correlations
    ------------------------
    The default choice (and that chosen by other functions in this package,
    not least emulator_from_data) is exponential-squared 'expSq', due to its
    wide applicability and differentiability. However, one can instantiate a
    Correlator object with a different structure from built-in alternatives:
        - Matern ('matern'): differentiable n times if nu > n
        - Ornstein-Uhlenbeck (ornUhl): not differentiable
        - Rational Quadratic (ratQuad): differentiable
    One further correlation function, gamma-exponential (gammaExp) is included
    but not widely used or recommended. It is, however, a good example of how
    one might construct custom correlation functions.

    A user-defined correlation function can be provided to the Correlator
    constructor: it must accept DataFrame objects as its first and second
    arguments, a named list of hyperparameters as its third, and return a
    matrix of correlations between rows of the DataFrames. If a derivative
    exists for a user-defined correlation function, it can also be supplied
    and used: if the correlation function is 'myCorr', then the derivative
    function should be 'myCorrDiff' and follow the same input structure as
    built-ins: i.e. function argument (x, p1, xp, p2, actives).

    If defining a custom correlation function, care should be taken with
    hyperparameter estimation in other functions: for example, in emulator_from_data.

    Attributes
    ----------
    corrName: str
        A string corresponding to the correlation type (default 'expSq')
    corrFunc: function
        The corresponding function call for corrName
    hyperp: dict
        A dictionary of requisite hyperparameters (default: {'theta': 0.1})
    nugget: float
        The value of the 'nugget' term
    
    Methods
    -------
    getCorr(x, xp = None, actives = None, useNugget = True)
        Calculates correlation between the active variable subset of two DataFrames
        of points. If xp is not provided, getCorr(x,x,...) is returned.
    getCorrDiff(x, p1, xp = None, p2 = None, actives = None)
        Calculates the derivative of the correlation between x and xp.
        Requires at least one differential direction, p1, and the existence
        of a differential function with naming convention <corrname>Diff.
    setHyperp(newHP, nug = None)
        Create Correlator from current object with hyperparameters newHP
        (and possibly a changed nugget term nug). Note that this does not
        modify the original Correlator object.
    getHyperp()
        Returns the Correlator hyperparameters.
    """

    def __init__(self, corr = 'expSq', hp = {'theta': 0.1}, nug = 0):
        """
        Parameters
        ----------
        corr: str
            A string corresponding to the correlation type (default 'expSq')
        hp: dict
            A dictionary containing the values of the hyperparameters (default {'theta': 0.1})
        nug: float
            The value of the nugget.
        """

        self.corrName = corr
        try:
            self.corrFunc = globals()[self.corrName]
        except KeyError:
            print("No correlation function of type '" + corr + "' recognised. Reverting to 'expSq'")
            self.corrName = "expSq"
            self.corrFunc = globals()[self.corrName]
        self.hyperp = hp
        self.nugget = nug
    
    def getCorr(self, x, xp = None, actives = None, useNugget = True):
        """
        Returns the correlation between x and xp.

        If xp is not provided, then the correlation between the DataFrame
        x with itself is returned. All variables are assumed to be active
        unless otherwise stated in actives, which can be a Boolean vector
        of length equal to the number of columns of x and xp.

        Parameters
        ----------
        x: DataFrame
            The first collection of points: each row corresponds to a different point
        xp: DataFrame, optional
            The second collection of points, if required
        actives: Array[Boolean], optional
            An array of Boolean values: True indicates that a given variable is active.
            Default: None (indicating all variables are active)
        useNugget: Boolean
            Whether to include a nugget term. Default: True
        
        Returns
        -------
        An array with m rows and n columns (where x consists of n rows
        and xp of m rows); the (i,j)th element is the correlation between
        x_j and xp_i.

        """

        if xp is None:
            if np.shape(x)[0] == 1:
                return 1
            return self.getCorr(x, x, actives, useNugget)
        if actives is None:
            return self.corrFunc(x, xp, self.hyperp)
        res = self.corrFunc(x.loc[:,actives], xp.loc[:,actives], self.hyperp)
        if not useNugget:
            return res
        extra = getDist(x, xp)
        extra[extra < 1e-10] = 1
        extra[extra != 1] = 0
        return (1-self.nugget) * res + self.nugget * extra
    def getCorrDiff(self, x, p1, xp = None, p2 = None, actives = None):
        """
        Returns the derivative of the correlation between x and xp.

        If xp is not provided, then the correlation between the DataFrame
        x with itself is returned. All variables are assumed to be active
        unless otherwise stated in actives, which can be a Boolean vector
        of length equal to the number of columns of x and xp. At least one
        derivative direction must be supplied in p1, which should be the
        name of the parameter in question. Second derivatives are calculated
        by supplying an additional direction to p2.

        Parameters
        ----------
        x: DataFrame
            The first collection of points: each row corresponds to a different point
        p1: str
            String corresponding to a derivative direction
        xp: DataFrame, optional
            The second collection of points, if required
        p2: str, optional
            String corresponding to second derivative direction
        actives: Array[Boolean], optional
            An array of Boolean values: True indicates that a given variable is active.
            Default: None (indicating all variables are active)
        
        Returns
        -------
        An array with m rows and n columns (where x consists of n rows
        and xp of m rows); the (i,j)th element is the derivative of the 
        correlation between x_j and xp_i.

        Raises
        ------
        NotImplementedError
            If the correlation function in question is not differentiable.
        """

        if actives is None:
            actives = np.full(np.shape(x)[1], True)
        if xp is None:
            return self.getCorrDiff(x, p1, x, p2, actives)
        if not(p1 in x.loc[:,actives].columns) or (not(p2 is None) and not(p2 in x.loc[:,actives].columns)):
            return np.zeros(shape = (np.shape(xp)[0], np.shape(x)[0]))
        try:
            dfunc = globals()[self.corrName + "Diff"]
        except KeyError:
            raise NotImplementedError("No derivative expression exists for correlation" + self.corrName + ".")
        return dfunc(x.loc[:,actives], xp.loc[:,actives], self.hyperp, p1, p2)
        
    def setHyperp(self, newHP, nug = None):
        """
        Creates a Correlator based on this one with modified hyperparameters.

        The new Correlator object maintains the correlation type of this object,
        but has hyperparameter specifications given by newHP, and potentially
        a different nugget term given by nug. Note that this does not modify-in-place,
        and that the original object is accessible after use of setHyperp.
        If hyperparameters are missing from the new specification, original
        values are maintained.

        Parameters
        ----------
        newHP: dict
            A dictionary of the new hyperparameter values
        nug: float, optional
            The new nugget term, if required. Default: None
        
        Returns
        -------
        A new Correlator object with the required specifications.
        """

        if nug is None:
            nug = self.nugget
        newCorr = deepcopy(self)
        for key, value in newHP.items():
            if key in newCorr.hyperp:
                newCorr.hyperp[key] = value
        newCorr.nugget = nug
        return newCorr
    def getHyperp(self):
        """
        Returns hyperparameter values.

        Returns
        -------
            A dictionary containing the hyperparameter values.
        """

        return self.hyperp
    def __str__(self, prepend = ""):
        outString = prepend + "Correlation type: " + self.corrName + "\n"
        outString = outString + prepend + "Hyperparameters: "
        for key in self.hyperp.keys():
            outString = outString + key + ": " + str(round(self.hyperp[key],4)) + "; "
        outString = outString[:-2] + "\n" + prepend + "Nugget term: " + str(round(self.nugget, 6)) + "\n"
        return f"{outString}"
