#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 14:39:30 2025

@author: mseifer1

Code-only version of https://github.com/pymc-devs/pymc-examples/blob/main/examples/howto/blackbox_external_likelihood_numpy.ipynb
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

# %config InlineBackend.figure_format = 'retina'
az.style.use("arviz-darkgrid")

def my_model(c, x):
    return 1
#overall, x doesnt do anything, so its presence can be ignored for now
def my_loglike(c, sigma, x, data):
    # We fail explicitly if inputs are not numerical types for the sake of this tutorial
    # As defined, my_loglike would actually work fine with PyTensor variables!
    for param in (c, sigma, x, data):
        if not isinstance(param, (float, np.ndarray)):
            raise TypeError(f"Invalid input type to loglike: {type(param)}")
    model = my_model(c, x)
    return np.log(((c**2)*np.sqrt(data**2+c**2-1)+(data**2-c**2)*\
            np.arctan(np.sqrt(data**2+c**2-1)))/((data**2+c**2)**2))
    # This is the log-likelihood for a Gaussian; we'd want to change this accordingly
    

# define a pytensor Op for our likelihood function

class LogLike(Op):
    def make_node(self, c, sigma, x, data) -> Apply:
        # Convert inputs to tensor variables
        c = pt.as_tensor(c)
        sigma = pt.as_tensor(sigma)
        x = pt.as_tensor(x)
        data = pt.as_tensor(data)

        inputs = [c, sigma, x, data]
        # Define output type, in our case a vector of likelihoods
        # with the same dimensions and same data type as data
        # If data must always be a vector, we could have hard-coded
        # outputs = [pt.vector()]
        outputs = [data.type()]

        # Apply is an object that combines inputs, outputs and an Op (self)
        return Apply(self, inputs, outputs)

    def perform(self, node: Apply, inputs: list[np.ndarray], outputs: list[list[None]]) -> None:
        # This is the method that compute numerical output
        # given numerical inputs. Everything here is numpy arrays
        c, sigma, x, data = inputs  # this will contain my variables

        # call our numpy log-likelihood function
        loglike_eval = my_loglike(c, sigma, x, data)

        # Save the result in the outputs list provided by PyTensor
        # There is one list per output, each containing another list
        # pre-populated with a `None` where the result should be saved.
        outputs[0][0] = np.asarray(loglike_eval)
        
# set up our data
N = 10  # number of data points
sigma = 1.0  # standard deviation of noise
#Revisit
ctrue = 1.0  # true y-intercept
x=0
truemodel = my_model(ctrue, x)

# make data
rng = np.random.default_rng(716743)
data = sigma * rng.normal(size=N) + truemodel

# create our Op
loglike_op = LogLike()

test_out = loglike_op(ctrue, sigma, x, data)

def custom_dist_loglike(data, c, sigma, x):
    # data, or observed is always passed as the first input of CustomDist
    return loglike_op(c, sigma, x, data)

# use PyMC to sampler from log-likelihood
with pm.Model() as no_grad_model:
    # uniform priors on m and c
    
    #This is a bit confusing and I'm unsure of what the proper
    # values here would be
    #m = pm.Uniform("m", lower=0, upper=1, initval=0.5)
    c = pm.Uniform("c", lower=0, upper=1, initval=0.5)
    #c = pm.Deterministic("c", ctrue)
    # use a CustomDist with a custom logp function
    likelihood = pm.CustomDist(
        "likelihood", c, sigma, x, observed=data, logp=custom_dist_loglike
    )
    
ip = no_grad_model.initial_point()

with no_grad_model:
    # Use custom number of draws to replace the HMC based defaults
    idata_no_grad = pm.sample(3000, tune=1000)

# plot the traces
az.plot_trace(idata_no_grad, lines=[("c", {}, ctrue)]);
