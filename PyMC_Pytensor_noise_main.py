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

    # PSEUDOTHEORY
    # make w= wt+n*sigma
    # N(w) = (pt.exp(-(w-wt)**2/2*sigma**2)*1/pt.sqrt(2*pi)*sigma
    # can use the below for P(w) just mdofiy for same input of wt+n*sigma
    sum = 0
    sigma = 0.02
    delta_w = sigma
    wt_regular = wt
    for n in range(-5, 5):

        # check if inputs are corret
        # Compute squared terms
        # TODO fix naming, it's very jack hammered atm
        w = wt_regular + n * sigma

        def function(wt):

            # CDF VERSION

            # wt = w + delta_w / 2

            wt_squared = pt.pow(wt, 2)
            wc_squared = pt.pow(wc, 2)
            sum_squares = wc_squared + wt_squared

            # First expression (used when wc < 1)
            square_root = pt.sqrt(sum_squares - 1)
            at = pt.arctan(square_root)

            numer1 = wt * (square_root - at)
            denom = sum_squares
            expr1 = numer1 / denom

            # Second expression (used when wc >= 1)
            wt_times_wc = wt * wc
            at_ratio = pt.arctan(wt / wc)
            # Single-argument arctan is OK here because wt & wc are always positive

            piece2_1 = (wc - 1) / ((wc**3) * (wt**2))

            piece2_2 = wc / sum_squares

            piece2_3 = (wt * at_ratio) / sum_squares

            expr2 = -piece2_1 - piece2_2 - piece2_3

            condition = pt.lt(wc, 1)

            result = pt.switch(condition, expr1, expr2)
            return result

        Nfunc = (pt.exp(-((n * sigma) ** 2) / (2 * sigma**2))) / (
            pt.sqrt(2 * pt.pi) * sigma
        )
        sum += Nfunc * (function(w + delta_w / 2) - function(w - delta_w / 2))
    return pt.log(sum)


def main():
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

    # Trying the "smeared" distribution idea
    sigma = 0.5

    model = pm.Model()

    def transform(y, min):
        wc = min * (np.exp(y) + 1)
        return wc

    with model:
        # Priors for unknown model parameters.  I defined q to be wc - 1, to avoid
        # confusion between the model parameter and the inverse speed of light as a
        # function of the parameters.

        q = pm.TruncatedNormal("q", sigma=0.01, lower=-1)
        wc = q + 1

        # Expected value of wc, in terms of unknown model parameters and observed "X" values.
        # Right now this is very simple.  Eventually it will need to accept more parameter
        # values, along with RA & declination.

        # Likelihood (sampling distribution) of observations
        wt_obs = pm.CustomDist("wt_obs", wc, observed=wt_data, logp=loglike)
        step = pm.Metropolis()
        trace = pm.sample(8000, tune=1000, step=step)

        # trace_transform = trace.map(lambda y: wc_min*(np.exp(y)+1), groups="posterior")

    summ = az.summary(trace)
    # print(summ)
    az.plot_trace(trace, show=True)
    az.plot_posterior(trace, round_to=3, figsize=[8, 4], textsize=10)
    summary_with_quartiles = az.summary(
        trace,
        stat_funcs={
            "25%": lambda x: np.percentile(x, 25),
            "50%": lambda x: np.percentile(x, 50),
            "75%": lambda x: np.percentile(x, 75),
        },
    )

    print(summary_with_quartiles)


if __name__ == "__main__":
    main()
