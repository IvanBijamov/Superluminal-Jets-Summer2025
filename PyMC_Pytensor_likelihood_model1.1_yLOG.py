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

    result = expr1  # pt.switch(condition, expr1, expr2)

    return result


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
    y = pm.Normal("y", mu=0, sigma=10)
    # Expected value of wc, in terms of unknown model parameters and observed "X" values.
    # Right now this is very simple.  Eventually it will need to accept more parameter
    # values, along with RA & declination.
    # wc = q + 1

    # Log version
    # y = np.log((wc/wt_min)-1)
    wc = wc_min * (np.exp(y) + 1)
    # TODO reparameter, almost gets rid of divergences, but values are off. Fix somehow? Redefining and whatnot

    # Likelihood (sampling distribution) of observations
    wt_obs = pm.CustomDist("wt_obs", wc, observed=wt_data, logp=loglike)

    trace = pm.sample(1000, target_accept=0.96)

    trace_transform = trace.map(
        lambda y: wc_min * (np.exp(y) + 1) - 1, groups="posterior"
    )

summ = az.summary(trace_transform)
# print(summ)
az.plot_trace(trace_transform, show=True)
summary_with_quartiles = az.summary(
    trace_transform,
    stat_funcs={
        "25%": lambda x: np.percentile(x, 25),
        "50%": lambda x: np.percentile(x, 50),  # This is the median
        "75%": lambda x: np.percentile(x, 75),
    },
)

print(summary_with_quartiles)
