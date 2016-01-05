r"""
Filename: lucastree.py

Authors: Joao Brogueira and Fabian Schuetze 

This file is a slight modification of the lucastree.py file 
by Thomas Sargent, John Stachurski, Spencer Lyon under the 
quant-econ project. We don't claim authorship of the entire file,
but full responsability for it and any existing mistakes.

Solves the price function for the Lucas tree in a continuous state
setting, using piecewise linear approximation for the sequence of
candidate price functions.  The consumption endownment follows the log
linear AR(1) process

.. math::

    log y' = \alpha log y + \sigma \epsilon

where y' is a next period y and epsilon is an iid standard normal shock.
Hence

.. math::

    y' = y^{\alpha} * \xi,

where

.. math::

    \xi = e^(\sigma * \epsilon)

The distribution phi of xi is

.. math::

    \phi = LN(0, \sigma^2),
    
where LN means lognormal.

"""
#from __future__ import division  # == Omit for Python 3.x == #
import numpy as np
from scipy.stats import lognorm
from scipy.integrate import fixed_quad
from quantecon.compute_fp import compute_fixed_point


class LucasTree(object):

    """
    Class to solve for the price of a tree in the Lucas
    asset pricing model

    Parameters
    ----------
    gamma : scalar(float)
        The coefficient of risk aversion in the investor's CRRA utility
        function
    beta : scalar(float)
        The investor's discount factor
    alpha : scalar(float)
        The correlation coefficient in the shock process
    sigma : scalar(float)
        The volatility of the shock process
    grid : array_like(float), optional(default=None)
        The grid points on which to evaluate the asset prices. Grid
        points should be nonnegative. If None is passed, we will create
        a reasonable one for you

    Attributes
    ----------
    gamma, beta, alpha, sigma, grid : see Parameters
    grid_min, grid_max, grid_size : scalar(int)
        Properties for grid upon which prices are evaluated
    init_h : array_like(float)
        The functional values h(y) with grid points being arguments 
    phi : scipy.stats.lognorm
        The distribution for the shock process

    Notes
    -----
    This file is a slight modification of the lucastree.py file 
    by Thomas Sargent, John Stachurski, Spencer Lyon, [SSL]_ under the 
    quant-econ project. We don't claim authorship of the entire file,
    but full responsability for it and any existing mistakes.

    References
    ----------
    .. [SSL] Thomas Sargent, John Stachurski and Spencer Lyon, lucastree.py,
    GitHub repository, 
    https://github.com/QuantEcon/QuantEcon.py/blob/master/quantecon/models/lucastree.py

    Examples
    --------
    >>> tree = LucasTree(gamma=2, beta=0.95, alpha=0.90, sigma=0.1)
    >>> grid, price_vals = tree.grid, tree.compute_lt_price()

    """

    def __init__(self, gamma, beta, alpha, sigma, grid=None):
        self.gamma = gamma
        self.beta = beta
        self.alpha = alpha
        self.sigma = sigma

        # == set up grid == #
        if grid is None:
            (self.grid, self.grid_min,
             self.grid_max, self.grid_size) = self._new_grid()
        else:
            self.grid = np.asarray(grid)
            self.grid_min = min(grid)
            self.grid_max = max(grid)
            self.grid_size = len(grid)

        # == set up distribution for shocks == #
        self.phi = lognorm(sigma)

        # == set up integration bounds. 4 Standard deviations. Make them
        # private attributes b/c users don't need to see them, but we
        # only want to compute them once. == #
        self._int_min = np.exp(-5 * sigma)
        self._int_max = np.exp(5 * sigma)

        # == Set up h for the Lucas Operator == #
        self.init_h = self.h(self.grid)

    def h(self, x):
        """
        Compute the function values of h in the Lucas operator. 

        Parameters
        ----------
        x : array_like(float)
        The arguments over which to computer the function values 

        Returns
        -------
        h : array_like(float)
        The functional values 

        Notes
        -----
        Recall the functional form of h 

        .. math:: h(x) &= \beta * \int_Z u'(G(x,z)) phi(dz)
                       &= \beta x**((1-\gamma)*\alpha) * \exp((1-\gamma)**2 *\sigma /2) 

        """
        alpha, gamma, beta, sigma = self.alpha, self.gamma, self.beta, self.sigma
        h = beta * x**((1 - gamma) * alpha) * \
            np.exp((1 - gamma)**2 * sigma**2 / 2) * np.ones(x.size)

        return h

    def _new_grid(self):
        """
        Construct the default grid for the problem

        This is defined to be np.linspace(0, 10, 100) when alpha >= 1
        and 100 evenly spaced points covering 4 standard deviations
        when alpha < 1

        """
        grid_size = 50
        if abs(self.alpha) >= 1.0:
            grid_min, grid_max = 0.1, 10
        else:
            # == Set the grid interval to contain most of the mass of the
            # stationary distribution of the consumption endowment == #
            ssd = self.sigma / np.sqrt(1 - self.alpha**2)
            grid_min, grid_max = np.exp(-4 * ssd), np.exp(4 * ssd)

        grid = np.linspace(grid_min, grid_max, grid_size)

        return grid, grid_min, grid_max, grid_size

    def integrate(self, g, int_min=None, int_max=None):
        """
        Integrate the function g(z) * self.phi(z) from int_min to
        int_max.

        Parameters
        ----------
        g : function
            The function which to integrate

        int_min, int_max : scalar(float), optional
            The bounds of integration. If either of these parameters are
            `None` (the default), they will be set to 4 standard
            deviations above and below the mean.

        Returns
        -------
        result : scalar(float)
            The result of the integration

        """
        # == Simplify notation == #
        phi = self.phi
        if int_min is None:
            int_min = self._int_min
        if int_max is None:
            int_max = self._int_max

        # == set up integrand and integrate == #
        integrand = lambda z: g(z) * phi.pdf(z)
        result, error = fixed_quad(integrand, int_min, int_max, n=20)
        return result, error

    def Approximation(self, x, grid, f):
        r"""
        Approximates the function f at given sample points x.

        Parameters
        ----------
        x: array_like(float)
            Sample points over which the function f is evaluated

        grid: array_like(float) 
            The grid values representing the domain of f 

        f: array_like(float)
            The function values of f over the grid 

        Returns:
        --------
        fApprox: array_like(float)
            The approximated function values at x

        Notes
        -----
        Interpolation is done by the following function:

        .. math:: f(x) = f(y_L) + \dfrac{f(y_H) - f(y_L)}{h(y_H) - h(y_L)} (h(x) - h(y_L) ).

        Extrapolation is done as follows:

        .. math:: f(x) = 
        \begin{cases}
        f(y_1) + \dfrac{f(y_1) - f(y_2)}{h(y_1) - h(y_2)} \left(h(x) - h(y_1) \right) & \text{if } x < y_1,\\
        f(y_N) + \dfrac{f(y_N) - f(y_{N-1})}{h(y_N) - h(y_{N-1})} \left( h(x) - h(y_N) \right) & \text{if } x > y_N.
        \end{cases}

        The approximation routine imposes the functional
        form of the function :math:`h` onto the function math:`f`, as stated
        in chapter 9.2 (in particular theorem 9.6 and 9.7 and exercise 9.7) of the 
        book by Stokey, Lucas and Prescott (1989).

        """
        # == Initalize and create empty arrays to be filled in the == #
        gamma, sigma, beta = self.gamma, self.sigma, self.beta
        hX, hGrid = self.h(x), self.init_h
        fL, fH, fApprox = np.empty_like(x), np.empty_like(x), np.empty_like(x)
        hL, idxL, idxH, hH = np.empty_like(x), np.empty_like(
            x), np.empty_like(x), np.empty_like(x)

        # == Create Boolean array to determine which sample points are used for interpoltion
        # and which are used for extrapolation == #
        lower, middle, upper = (x < grid[0]), (x > grid[0]) & (
            x < grid[-1]), (x > grid[-1])

        # == Calcualte the indices of y_L, idxL[index], and y_H ,idxH[index], that are below and above a sample point, called value.
        # In the notation of the interpolation routine, these indices are used to pick the function values
        # f(y_L),f(y_H),h(y_L) and h(y_H) == #
        for index, value in enumerate(x):
            # Calculates the indices of y_L
            idxL[index] = (np.append(grid[grid <= value], grid[0])).argmax()
            idxH[index] = min(idxL[index] + 1, len(grid) - 1)
            fL[index] = f[idxL[index]]
            fH[index] = f[idxH[index]]
            hL[index] = hGrid[idxL[index]]
            hH[index] = hGrid[idxH[index]]

        # == Interpolation == #
        if self.alpha != 0:
            ratio = (fH[middle] - fL[middle]) / (hH[middle] - hL[middle])
        elif self.alpha == 0:
            # If self.alpha ==0, `ratio` is zero, as hH == hL
            ratio = (hH[middle] - hL[middle])
        fApprox[middle] = fL[middle] + ratio * (hX[middle] - hL[middle])

        # == Extrapolation == #
        if self.alpha != 0:
            fApprox[lower] = f[
                0] + (f[0] - f[1]) / (hGrid[0] - hGrid[1]) * (hX[lower] - hGrid[0])
            fApprox[upper] = f[-1] + \
                (f[-1] - f[-2]) / (hGrid[-1] - hGrid[-2]) * \
                (hX[upper] - hGrid[-1])
        elif self.alpha == 0:
            fApprox[lower] = f[0]
            fApprox[upper] = f[-1]

        return fApprox

    def lucas_operator(self, f, Tf=None):
        """
        The approximate Lucas operator, which computes and returns the
        updated function Tf on the grid points.

        Parameters
        ----------
        f : array_like(float)
            A candidate function on R_+ represented as points on a grid
            and should be flat NumPy array with len(f) = len(grid)

        Tf : array_like(float)
            Optional storage array for Tf

        Returns
        -------
        Tf : array_like(float)
            The updated function Tf

        Notes
        -----
        The argument `Tf` is optional, but recommended. If it is passed
        into this function, then we do not have to allocate any memory
        for the array here. As this function is often called many times
        in an iterative algorithm, this can save significant computation
        time.

        """
        grid,  h = self.grid, self.init_h
        alpha, beta = self.alpha, self.beta

        # == set up storage if needed == #
        if Tf is None:
            Tf = np.empty_like(f)

        # == Apply the T operator to f == #
        Af = lambda x: self.Approximation(x, grid, f)

        for i, y in enumerate(grid):
            Tf[i] = h[i] + beta * self.integrate(lambda z: Af(y**alpha * z))[0]

        return Tf

    def compute_lt_price(self, error_tol=1e-7, max_iter=600, verbose=0):
        """
        Compute the equilibrium price function associated with Lucas
        tree lt

        Parameters
        ----------
        error_tol, max_iter, verbose
            Arguments to be passed directly to
            `quantecon.compute_fixed_point`. See that docstring for more
            information


        Returns
        -------
        price : array_like(float)
            The prices at the grid points in the attribute `grid` of the
            object

        """
        # == simplify notation == #
        grid, grid_size = self.grid, self.grid_size
        lucas_operator, gamma = self.lucas_operator, self.gamma

        # == Create storage array for compute_fixed_point. Reduces  memory
        # allocation and speeds code up == #
        Tf = np.empty(grid_size)

        # == Initial guess, just a vector of ones == #
        f_init = np.ones(grid_size)
        f = compute_fixed_point(lucas_operator, f_init, error_tol,
                                max_iter, verbose, Tf=Tf)

        price = f * grid**gamma

        return price