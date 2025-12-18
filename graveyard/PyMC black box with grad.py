#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 15:10:44 2025

@author: mseifer1

Drawn from current example on PyMC documentation page:  
    https://www.pymc.io/projects/examples/en/latest/howto/blackbox_external_likelihood_numpy.html

However, the gradient function does not seem to work in this code;  this is a known issue:
    https://discourse.pymc.io/t/error-in-black-box-likelihood-function-example-with-pymc/16418/4
    
So this file is deprecated.  Use the file "PyMC old example with grad.py" 
instead, as that one seems to work.
"""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

from pytensor.graph import Apply, Op
from scipy.optimize import approx_fprime

print(f"Running on PyMC v{pm.__version__}")

az.style.use("arviz-darkgrid")

def my_model(m, c, x):
    return m * x + c

def my_loglike(m, c, sigma, x, data):
    # We fail explicitly if inputs are not numerical types for the sake of this tutorial
    # As defined, my_loglike would actually work fine with PyTensor variables!
    for param in (m, c, sigma, x, data):
        if not isinstance(param, (float, np.ndarray)):
            raise TypeError(f"Invalid input type to loglike: {type(param)}")
    model = my_model(m, c, x)
    return -0.5 * ((data - model) / sigma) ** 2 - np.log(np.sqrt(2 * np.pi)) - np.log(sigma)

def finite_differences_loglike(m, c, sigma, x, data, eps=1e-7):
    """
    Calculate the partial derivatives of a function at a set of values. The
    derivatives are calculated using scipy approx_fprime function.

    Parameters
    ----------
    m, c: array_like
        The parameters of the function for which we wish to define partial derivatives
    x, data, sigma:
        Observed variables as we have been using so far


    Returns
    -------
    grad_wrt_m: array_like
        Partial derivative wrt to the m parameter
    grad_wrt_c: array_like
        Partial derivative wrt to the c parameter
    """

    def inner_func(mc, sigma, x, data):
        return my_loglike(*mc, sigma, x, data)

    grad_wrt_mc = approx_fprime([m, c], inner_func, [eps, eps], sigma, x, data)

    return grad_wrt_mc[:, 0], grad_wrt_mc[:, 1]

class LogLikeWithGrad(Op):
    def make_node(self, m, c, sigma, x, data) -> Apply:
        # Same as before
        m = pt.as_tensor(m)
        c = pt.as_tensor(c)
        sigma = pt.as_tensor(sigma)
        x = pt.as_tensor(x)
        data = pt.as_tensor(data)
        
        inputs = [m, c, sigma, x, data]
        outputs = [data.type()]
        return Apply(self, inputs, outputs)

    def perform(self, node: Apply, inputs: list[np.ndarray], outputs: list[list[None]]) -> None:
        # Same as before
        m, c, sigma, x, data = inputs  # this will contain my variables
        loglike_eval = my_loglike(m, c, sigma, x, data)
        outputs[0][0] = np.asarray(loglike_eval)

    def grad(
        self, inputs: list[pt.TensorVariable], g: list[pt.TensorVariable]
    ) -> list[pt.TensorVariable]:
        # NEW!
        # the method that calculates the gradients - it actually returns the vector-Jacobian product
        m, c, sigma, x, data = inputs

        # Our gradient expression assumes these are scalar parameters
        if m.type.ndim != 0 or c.type.ndim != 0:
            raise NotImplementedError("Gradient only implemented for scalar m and c")

        grad_wrt_m, grad_wrt_c = loglikegrad_op(m, c, sigma, x, data)

        # out_grad is a tensor of gradients of the Op outputs wrt to the function cost
        [out_grad] = g
        return [
            pt.sum(out_grad * grad_wrt_m),
            pt.sum(out_grad * grad_wrt_c),
            # We did not implement gradients wrt to the last 3 inputs
            # This won't be a problem for sampling, as those are constants in our model
            pytensor.gradient.grad_not_implemented(self, 2, sigma),
            pytensor.gradient.grad_not_implemented(self, 3, x),
            pytensor.gradient.grad_not_implemented(self, 4, data),
        ]
    
class LogLikeGrad(Op):
    def make_node(self, m, c, sigma, x, data) -> Apply:
        m = pt.as_tensor(m)
        c = pt.as_tensor(c)
        sigma = pt.as_tensor(sigma)
        x = pt.as_tensor(x)
        data = pt.as_tensor(data)

        inputs = [m, c, sigma, x, data]
        # There are two outputs with the same type as data,
        # for the partial derivatives wrt to m, c
        outputs = [data.type(), data.type()]

        return Apply(self, inputs, outputs)

    def perform(self, node: Apply, inputs: list[np.ndarray], outputs: list[list[None]]) -> None:
        m, c, sigma, x, data = inputs

        # calculate gradients
        grad_wrt_m, grad_wrt_c = finite_differences_loglike(m, c, sigma, x, data)

        outputs[0][0] = grad_wrt_m
        outputs[1][0] = grad_wrt_c


# Initialize the Ops
loglikewithgrad_op = LogLikeWithGrad()
loglikegrad_op = LogLikeGrad()

# set up our data
N = 10  # number of data points
sigma = 1.0  # standard deviation of noise
x = np.linspace(0.0, 9.0, N)

mtrue = 0.4  # true gradient
ctrue = 3.0  # true y-intercept

truemodel = my_model(mtrue, ctrue, x)

# make data
rng = np.random.default_rng(716743)
data = sigma * rng.normal(size=N) + truemodel
        
m = pt.scalar("m")
c = pt.scalar("c")
out = loglikewithgrad_op(m, c, sigma, x, data)
eval_out = out.eval({m: mtrue, c: ctrue})
print(eval_out)
assert np.allclose(eval_out, my_loglike(mtrue, ctrue, sigma, x, data))

def test_fn(m, c):
    return loglikewithgrad_op(m, c, sigma, x, data)
# This raises an error if the gradient output is not within a given tolerance
pytensor.gradient.verify_grad(test_fn, pt=[mtrue, ctrue], rng=np.random.default_rng(123))

def custom_dist_loglike(data, m, c, sigma, x):
    # data, or observed is always passed as the first input of CustomDist
    return loglikewithgrad_op(m, c, sigma, x, data)

# use PyMC to sampler from log-likelihood
with pm.Model() as grad_model:
    # uniform priors on m and c
    m = pm.Uniform("m", lower=-10.0, upper=10.0)
    c = pm.Uniform("c", lower=-10.0, upper=10.0)

    # use a CustomDist with a custom logp function
    likelihood = pm.CustomDist(
        "likelihood", m, c, sigma, x, observed=data, logp=custom_dist_loglike
    )
    
with grad_model:
    # Use custom number of draws to replace the HMC based defaults
    idata_grad = pm.sample()

# plot the traces
az.plot_trace(idata_grad, lines=[("m", {}, mtrue), ("c", {}, ctrue)]);