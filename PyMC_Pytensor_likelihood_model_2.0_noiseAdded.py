#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 08:53:03 2025

@author: mseifer1
"""

import arviz as az
import matplotlib.pyplot as plt
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
    w_cut = pt.sqrt(1 - pt.pow(wc, 2))
    sigma = 0.01
    delta_u = (w_cut - wt) / sigma
    # Compute squared terms
    wt_squared = pt.pow(wt, 2)
    wc_squared = pt.pow(wc, 2)
    sum_squares = wc_squared + wt_squared

    # First expression (used when wc < 1)
    square_root = pt.sqrt(sum_squares - 1)
    at = pt.arctan(square_root)

    numer1 = (wc_squared * square_root) + ((wt_squared - wc_squared) * at)
    denom = pt.pow(sum_squares, 2)
    expr1 = pt.log(numer1 / denom)

    # Second expression (used when wc >= 1)
    wt_times_wc = wt * wc
    at_ratio = pt.arctan(wt / wc)
    # Single-argument arctan is OK here because wt & wc are always positive

    numer2 = (
        2 * wt_times_wc * (wc - 1) + wt_times_wc + (wt_squared - wc_squared) * at_ratio
    )

    expr2 = pt.log(numer2 / denom)

    condition = pt.lt(wc, 1)

    far_out_bit = pt.switch(condition, expr1, expr2)

    chunk1_out = 0.56419 * pt.pow((1 - pt.pow(wc, 2)), 5 / 4)

    chunk2_in = 1.03045 - (
        (0.0112281 * (pt.sqrt(1 - wc**2) - wt) ** 5) / pt.pow(sigma, 5)
    )

    chunk3_in = (0.0322015 * (pt.sqrt(1 - wc**2) - wt) ** 4) / pt.pow(sigma, 4)

    chunk4_in = (0.089825 * (pt.sqrt(1 - wc**2) - wt) ** 3) / pt.pow(sigma, 3)

    chunk5_in = (0.257612 * (pt.sqrt(1 - wc**2) - wt) ** 2) / pt.pow(sigma, 2)

    chunk6_in = (1.0779 * (pt.sqrt(1 - wc**2) - wt)) / sigma

    middle_bridge = (
        chunk1_out
        * (chunk2_in - chunk3_in + chunk4_in + chunk5_in - chunk6_in)
        * pt.sqrt(sigma)
    )

    far_left_bit = (pt.exp(-pt.pow(delta_u, 2))) / pt.pow(delta_u, 3 / 2)
    upper_bound = 0.61
    lower_bound = 0.60
    all_composed = pt.where(
        w_cut < upper_bound,
        pt.where(
            (w_cut < upper_bound) & (w_cut > lower_bound), middle_bridge, far_left_bit
        ),
        far_out_bit,
    )

    return all_composed


# Get the directory this code file is stored in
dir_path = os.path.dirname(os.path.realpath(__file__))

# find the path for the data source;  this should work on everyone's system now
dataset = "/isotropic_sims/10000/data_3957522615761_xx_0.8_yy_0.8_zz_0.8.csv"
# dataset = "/isotropic_sims/10000/data_3957522615600_xx_1.2_yy_1.2_zz_1.2.csv"

dataSource = dir_path + dataset

print(f"Running on PyMC v{pm.__version__}")

# Import data from file
dataAll = importCSV(dataSource)
radec_data = [sublist[1:3] for sublist in dataAll]
wt_data = [sublist[3] for sublist in dataAll]
wt_min = np.min(wt_data)
wc_min = np.sqrt(1 - wt_min**2)

model = pm.Model()


def transform(y, min):
    wc = min * (np.exp(y) + 1)
    return wc


with model:
    # Priors for unknown model parameters.  I defined q to be wc - 1, to avoid
    # confusion between the model parameter and the inverse speed of light as a
    # function of the parameters.

    q = pm.TruncatedNormal("q", sigma=10, lower=-1)
    wc = q + 1

    # something with potential??
    b = 1.5
    c = 10
    # potential = pm.Potential("wc_constraint", 1 / pt.softplus(c + b * (wc - wc_min)))
    # w = pm.CustomDist("w",wc,logp=loglike)
    # sigma = pm.HalfNormal("sigma", sigma=1.0)
    #
    # wt = pm.Normal("wt", sigma=sigma,initval=w, observed=wt_data)
    # Expected value of wc, in terms of unknown model parameters and observed "X" values.
    # Right now this is very simple.  Eventually it will need to accept more parameter
    # values, along with RA & declination.

    # Likelihood (sampling distribution) of observations
    wt_obs = pm.CustomDist("wt_obs", wc, observed=wt_data, logp=loglike)
    step = pm.Metropolis()
    trace = pm.sample(10000, tune=1000, target_accept=0.9)

    # trace_transform = trace.map(lambda y: wc_min*(np.exp(y)+1), groups="posterior")


summ = az.summary(trace)
print(summ)
az.plot_trace(trace, show=True)
az.plot_posterior(trace, round_to=3, figsize=[8, 4], textsize=10)
