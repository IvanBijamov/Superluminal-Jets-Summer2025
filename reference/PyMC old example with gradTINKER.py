#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 15:48:40 2025

@author: mseifer1

Drawn from older example on GitHub:
    https://github.com/pymc-devs/pymc-examples/blob/6f2eb44c81896f7e827623ccb12e06731726b6bc/examples/howto/blackbox_external_likelihood_numpy.ipynb
"""

import arviz as az
import IPython
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scripts.simulationImport import importCSV

# import warnings

# warnings.simplefilter('error',RuntimeWarning)
dataSource = "/Users/ivanbijamov/Library/Mobile Documents/com~apple~CloudDocs/SuperluminalJets/isotropic_sims/300/data_3957593320521_xx_1.2_yy_1.2_zz_1.2.csv"


print(f"Running on PyMC v{pm.__version__}")

az.style.use("arviz-darkgrid")


def my_model(theta, x):
    c = theta

    return 1


# log of probability function from notes
def my_loglike(theta, x, data, sigma):
    c = theta
    if c <= 1:
        arg = np.maximum(data**2 + c**2 - 1, 1e-12)  # Keep inside sqrt positive
        # arg = data**2 + c**2 - 1
        numerator = (c**2) * np.sqrt(arg) + (data**2 - c**2) * np.arctan(np.sqrt(arg))
        denominator = (data**2 + c**2) ** 2
        return np.sum(np.log(numerator / denominator))
    # revisit function
    elif c > 1:
        function = ((2 * c * data * (c - 1)) / (data**2 + c**2) ** 2) + (
            (data * c + (data**2 - c**2) * np.arctan(data / c)) / (data**2 + c**2) ** 2
        )
        return np.sum(np.log(function))


def normal_gradients(theta, x, data, sigma):
    """
    Calculate the partial derivatives of a function at a set of values. The
    derivatives are calculated using the central difference, using an iterative
    method to check that the values converge as step size decreases.

    Parameters
    ----------
    theta: array_like
        A set of values, that are passed to a function, at which to calculate
        the gradient of that function
    x, data, sigma:
        Observed variables as we have been using so far


    Returns
    -------
    grads: array_like
        An array of gradients for each non-fixed value.
    """

    grads = np.empty(1)
    c = theta[0]
    d = data

    # TODO: Put partial deriviative here
    # This needs to be HEAVILY sanitized, there are so many runtime warnings with overflows and invalid
    # values
    # Fast light case
    if c >= 1:
        sqrRoot = np.sqrt(c**2 + d**2 - 1)
        at = np.arctan(sqrRoot)

        denum = np.where(
            np.isnan(at) | np.isinf(at),
            np.nan,
            sqrRoot * (c**2 + d**2) * (sqrRoot * c**2 + (d**2 - c**2) * at),
        )
        numer = np.where(
            np.isnan(at) | np.isinf(at),
            np.nan,
            c
            * (
                (1 + d**2) * c**2
                - c**4
                + (d**2) * (-1 + 2 * d**2)
                + 2 * (c**2 - 3 * d**2) * sqrRoot * at
            ),
        )
        grads = np.where(np.isnan(numer) | np.isnan(denum), np.nan, numer / denum)
    # Slow Light Case
    elif c > 1:
        at = np.arctan(data / theta[0])

        denum = np.where(
            np.isnan(at) | np.isinf(at),
            np.nan,
            (c**2 + d**2) * (c * (-1 + 2 * c) * d + (d**2 - c**2) * at),
        )
        numer = np.where(
            np.isnan(at) | np.isinf(at),
            np.nan,
            -2 * d * (-2 * c**2 + 2 * c**3 + d**2 - 2 * c * d**2)
            + 2 * (c**3 - 3 * c * d**2) * at,
        )

        grads = np.where(np.isnan(numer) | np.isnan(denum), np.nan, numer / denum)
    return grads


# define a pytensor Op for our likelihood function
class LogLikeWithGrad(pt.Op):
    itypes = [pt.dvector]  # expects a vector of parameter values when called
    otypes = [pt.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike, data, x, sigma):
        """
        Initialise with various things that the function requires. Below
        are the things that are needed in this particular example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that out function requires.
        """

        # add inputs as class attributes
        self.likelihood = loglike
        self.data = data
        self.x = x
        self.sigma = sigma

        # initialise the gradient Op (below)
        self.logpgrad = LogLikeGrad(self.data, self.x, self.sigma)

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        (theta,) = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = self.likelihood(theta, self.x, self.data, self.sigma)

        outputs[0][0] = np.array(logl)  # output the log-likelihood

    def grad(self, inputs, g):
        # the method that calculates the gradients - it actually returns the
        # vector-Jacobian product - g[0] is a vector of parameter values
        (theta,) = inputs  # our parameters
        return [g[0] * self.logpgrad(theta)]


class LogLikeGrad(pt.Op):
    """
    This Op will be called with a vector of values and also return a vector of
    values - the gradients in each dimension.
    """

    itypes = [pt.dvector]
    otypes = [pt.dvector]

    def __init__(self, data, x, sigma):
        """
        Initialise with various things that the function requires. Below
        are the things that are needed in this particular example.

        Parameters
        ----------
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that out function requires.
        """

        # add inputs as class attributes
        self.data = data
        self.x = x
        self.sigma = sigma

    def perform(self, node, inputs, outputs):
        (theta,) = inputs

        # calculate gradients
        grads = normal_gradients(theta, self.x, self.data, self.sigma)

        outputs[0][0] = grads


# set up our data
N = 10  # number of data points
sigma = 1.0  # standard deviation of noise
# x = np.linspace(0.0, 9.0, N)
x = 0

ctrue = 1.0  # true y-intercept
# Omitting because it doesnt do anything atm
# truemodel = my_model([ctrue], x)

# make data
dataAll = importCSV(dataSource)
data = [sublist[3] for sublist in dataAll]

# not sure why this makes it run, check my_loglike
# something weird happening with inputting an array
data = np.array(data, dtype=np.float64)

# create our Op
logl = LogLikeWithGrad(my_loglike, data, x, sigma)

# use PyMC to sampler from log-likelihood
with pm.Model() as opmodel:
    # uniform priors on m and c
    c = pm.TruncatedNormal("c", lower=0, sigma=0.5)
    # convert m and c to a tensor vector
    theta = pt.as_tensor_variable([c])

    # use a Potential
    pm.Potential("likelihood", logl(theta))

    idata_grad = pm.sample(10000, tuning=10000, target_accept=0.95)

# plot the traces
az.plot_trace(idata_grad, lines=[("c", {}, ctrue)])
