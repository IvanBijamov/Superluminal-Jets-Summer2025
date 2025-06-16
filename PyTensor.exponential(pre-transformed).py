#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 08:53:03 2025

@author: mseifer1
"""

import arviz as az
import matplotlib.pyplot 
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import os

from pytensor.tensor import as_tensor_variable

from simulationImport import importCSV

pytensor.config.exception_verbosity = "high"

def loglike(wt: pt.TensorVariable, wc: pt.TensorVariable) -> pt.TensorVariable:
    # wt = pt.vector("wt", dtype="float64")
    # wc = pt.scalar("wc", dtype="float64")
    
    # Compute squared terms
    wt_squared = pt.pow(wt, 2)
    wc_squared = pt.pow(wc, 2)
    sum_squares = wc_squared + wt_squared

    # First expression (used when wc < 1)
    square_root = pt.sqrt(sum_squares - 1)
    at = pt.arctan(square_root)

    numer1 = (wc_squared * square_root) + ((wt_squared - wc_squared) * at)
    denum1 = pt.pow(sum_squares, 2)
    expr1 = pt.log(numer1 / denum1)

    # Second expression (used when wc >= 1)
    wt_times_wc = wt * wc
    at_ratio = pt.arctan(wt / wc)
    # Single-argument arctan is OK here because wt & wc are always positive

    numer2 = 2 * wt_times_wc * (wc - 1) + wt_times_wc + (wt_squared - wc_squared) * at_ratio
    denum2 = pt.pow(sum_squares, 2)
    expr2 = pt.log(numer2 / denum2)
   
    condition = pt.lt(wc, 1)
    
    result = expr1 # pt.switch(condition, expr1, expr2)

    return result

# Get the directory this code file is stored in
dir_path = os.path.dirname(os.path.realpath(__file__))

# find the path for the data source;  this should work on everyone's system now
dataset = "/isotropic_sims/10000/data_3957522615761_xx_0.8_yy_0.8_zz_0.8.csv"
#dataset = "/isotropic_sims/10000/data_3957522615600_xx_1.2_yy_1.2_zz_1.2.csv"

dataSource = (
    dir_path + dataset
)

print(f"Running on PyMC v{pm.__version__}")

# Import data from file
dataAll = importCSV(dataSource)
radec_data = [sublist[1:3] for sublist in dataAll]
wt_data = [sublist[3] for sublist in dataAll]
# wt_data = pytensor.shared(wt_dataOLD, name="wt_shared")
wt_min = np.min(wt_data)
wc_min = np.sqrt(1-wt_min**2)

model = pm.Model()

with model:
    # Priors for unknown model parameters.  I defined q to be wc - 1, to avoid
    # confusion between the model parameter and the inverse speed of light as a
    # function of the parameters.
    y = pm.Normal("y", mu=0, sigma=10)

    # Expected value of wc, in terms of unknown model parameters and observed "X" values.
    # Right now this is very simple.  Eventually it will need to accept more parameter
    # values, along with RA & declination.
    #wc = q + 1
    
    wc = ((wc_min + y + np.sqrt(y**2 - 2 * wc_min * y + 5 * wc_min**2)))/2

    # Likelihood (sampling distribution) of observations
    wt_obs = pm.CustomDist("wt_obs", wc, observed=wt_data, logp=loglike)
    trace = pm.sample(10000, target_accept=0.9)


summ = az.summary(trace)
print(summ)
az.plot_trace(trace, show=True)
az.plot_posterior(trace, round_to=3, figsize=[8,4], textsize=10)
