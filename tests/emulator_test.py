import unittest
import pandas as pd
import numpy as np
from hmepy.correlations import *
from hmepy.emulator import *
from hmepy.utils import *

class testEmBasic(unittest.TestCase):
    basisF = [lambda x: 1, lambda x: np.asarray(x)[0]]
    bet = {'mu': [1,1], 'sigma': np.zeros((2,2))}
    u = {'sigma': 2, 'corr': Correlator()}
    ranges = {'x': [0,2]}
    tem = Emulator(basisF, bet, u, ranges)
    def test_basic(self):
        self.assertEqual(self.__class__.tem.activeVars, [True])
    def test_beta_call(self):
        self.assertIsNone(np.testing.assert_equal(self.__class__.tem.bSig, np.zeros((2,2))))
    def test_basis_f(self):
        calc = evalFuncs(self.__class__.tem.bF, pd.DataFrame(data = {'x': [2]}))
        self.assertIsNone(np.testing.assert_equal(calc, [[1],[2]]))
    def test_sigma(self):
        self.assertEqual(self.__class__.tem.uSig, 2)
    def test_corr_call(self):
        self.assertEqual(self.__class__.tem.corr.corrName, "expSq")
    def test_corr_hp(self):
        self.assertEqual(self.__class__.tem.corr.hyperp['theta'], 0.1)
        self.assertEqual(self.__class__.tem.corr.getHyperp()['theta'], 0.1)

class testEmPredict(unittest.TestCase):
    data = pd.DataFrame(data = {
        'x': np.arange(-1, 1.1, 0.2),
        'y': np.arange(0, 2.1, 0.2),
    })
    data['f'] = 0.8*np.sin((data['x']-0.2)*np.pi/0.35)
    bF = [lambda x: 1]
    bet = {'mu': [1], 'sigma': np.zeros((1,1))}
    u = {'corr': Correlator('matern', {'theta': 0.2, 'nu': 1.5}),
         'sigma': 1}
    rang = {'x': [-1, 1], 'y': [0,2]}
    dEmNoDat = Emulator(bF, bet, u, rang)
    print(dEmNoDat)
    tdf = pd.DataFrame(data = {'x': [-0.2, 0, 0.2], 'y': [0.8, 1, 1.2]})
    def test_untrained_exp(self):
        calc = self.__class__.dEmNoDat.getExp(self.__class__.tdf)
        self.assertIsNone(np.testing.assert_equal(calc, np.ones(3)))
    def test_untrained_cov(self):
        calc = self.__class__.dEmNoDat.getCov(self.__class__.tdf)
        self.assertIsNone(np.testing.assert_equal(calc, np.ones(3)))
    dEm = Emulator(bF, bet, u, rang, outName = 'f', data = data)
    def test_adjusted(self):
        calc = self.__class__.dEm.getExp(self.__class__.tdf)
        self.assertIsNone(np.testing.assert_almost_equal(calc, np.array(self.__class__.data.iloc[4:7,2])))
    def test_adjusted_cov(self):
        calc = self.__class__.dEm.getCov(self.__class__.tdf)
        self.assertIsNone(np.testing.assert_almost_equal(calc, np.zeros(3)))

if __name__ == "__main__":
    unittest.main()