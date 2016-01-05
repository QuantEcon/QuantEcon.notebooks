"""
filename: test_lucastree.py

Authors: Joao Brogueira and Fabian Schuetze 

This file contains for different test for the 
lucastree.py file 

Functions
---------
    compute_lt_price()      [Status: Tested in test_ConstantPDRatio, test_ConstantF,
                            test_slope_f, test_shape_f]

"""

import unittest
from lucastree import LucasTree  # This relative importing doesn't work!
import numpy as np


class Testlucastree(unittest.TestCase):

    """
    Test Suite for lucastree.py based on the outout of the 
    LucasTree.compute_lt_price() function.

    """
    # == Parameter values applicable to all test cases == #
    beta = 0.95
    sigma = 0.1

    # == Paramter values for different tests == #
    ConstantPD = np.array([2, 1])
    ConstantF = np.array([2, 0])
    FunctionalForm = np.array([[2, 0.75], [2, 0.5], [2, 0.25], [0.5, 0.75], [
                              0.5, 0.5], [0.5, 0.25], [0.5, -0.75], [0.5, -0.5], [0.5, -0.25]])

    # == Tolerance Criteria == #
    Tol = 1e-2

    def setUp(self):
        self.storage = lambda parameter0, parameter1: LucasTree(gamma=parameter0, beta=self.beta, alpha=parameter1,
                                                                sigma=self.sigma)

    def test_ConstantPDRatio(self):
        """
        Test whether the numerically computed price dividend ratio is 
        identical to its theoretical counterpart when dividend 
        growth follows an idd process

        """
        gamma, alpha = self.ConstantPD
        tree = self.storage(gamma, alpha)
        grid = tree.grid
        theoreticalPDRatio = np.ones(len(grid)) * self.beta * np.exp(
            (1 - gamma)**2 * self.sigma**2 / 2) / (1 - self.beta * np.exp((1 - gamma)**2 * self.sigma**2 / 2))
        self.assertTrue(
            np.allclose(theoreticalPDRatio, tree.compute_lt_price() / grid, atol=self.Tol))

    def test_ConstantF(self):
        """
        Tests whether the numericlaly obtained solution, math:`f` 
        to the functional equation :math:`f(y) = h(y) + \beta \int_Z f(G(y,z')) Q(z')`
        is identical to its theoretical counterpart, when divideds follow an 
        iid process 

        """
        gamma, alpha = self.ConstantF
        tree = self.storage(gamma, alpha)
        grid = tree.grid
        theoreticalF = np.ones(len(
            grid)) * self.beta * np.exp((1 - gamma)**2 * self.sigma**2 / 2) / (1 - self.beta)
        self.assertTrue(np.allclose(
            theoreticalF, tree.compute_lt_price() * grid**(-gamma), atol=self.Tol))

    def test_slope_f(self):
        """
        Tests whether the first difference of the numerically obtained function 
        :math:`f` is has the same sign as the first difference of the function 
        :math:`h`.

        Notes
        -----
        This test is motivated by Theorem 9.7 ans exercise 9.7c) of the 
        book by Stokey, Lucas and Prescott (1989)

        """
        for parameters in self.FunctionalForm:
            gamma, alpha = parameters
            tree = self.storage(gamma, alpha)
            f = tree.compute_lt_price() * tree.grid ** (-gamma)
            h = tree.h(tree.grid)
            fdiff, hdiff = np.ediff1d(f), np.ediff1d(h)
            if all(hdiff > 0):
                self.assertTrue(all(fdiff > 0))
            elif all(hdiff < 0):
                self.assertTrue(all(fdiff < 0))

    def test_shape_f(self):
        """
        Tests whether the second difference of the numerically obtained function 
        :math:`f` is has the same sign as the second difference of the function 
        :math:`h`.

        Notes
        -----
        This test is motivated by Theorem 9.8 ans exercise 9.7d) of the 
        book by Stokey, Lucas and Prescott (1989)

        """
        for parameters in self.FunctionalForm:
            gamma, alpha = parameters
            tree = self.storage(gamma, alpha)
            f = tree.compute_lt_price() * tree.grid ** (-gamma)
            h = tree.h(tree.grid)
            fdiff, hdiff = np.ediff1d(f), np.ediff1d(h)
            fdiff2, hdiff2 = np.ediff1d(fdiff), np.ediff1d(hdiff)
            if all(hdiff2 > 0):
                self.assertTrue(all(fdiff2 > 0))
            elif all(hdiff2 < 0):
                self.assertTrue(all(fdiff2 < 0))

    def tearDown(self):
        pass