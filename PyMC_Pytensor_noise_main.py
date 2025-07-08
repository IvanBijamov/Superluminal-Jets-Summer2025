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

pytensor.config.cxx = "/usr/bin/clang++"
pytensor.config.exception_verbosity = "high"


sigma = 0.02
n_val = 10


def loglike(
    wt: pt.TensorVariable,
    wc: pt.TensorVariable,
    sigma=sigma,
    n_val=n_val,
) -> pt.TensorVariable:
    # wt = pt.vector("wt", dtype="float64")
    # wc = pt.scalar("wc", dtype="float64")

    wt_regular = wt

    sum = 0

    # configurables

    delta_w = sigma / 5

    for n in range(-n_val, n_val + 1):

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

            at2 = pt.arctan(wt / wc)
            numer2 = wt**2 - wt * at2
            denom2 = sum_squares
            expr2 = numer2 / denom2

            fastslowcondition = pt.lt(wc, 1)

            result = pt.switch(fastslowcondition, expr1, expr2)

            # expr1 if wc < 1, expr2 if wc > 1

            negwtcondition = pt.lt(wt, 0)
            sumsquarecondition = pt.lt(sum_squares, 1)

            result = pt.switch(negwtcondition | sumsquarecondition, 0, result)

            # expr1 if wc < 1 & wt > 0, expr2 if wc > 1 & wt >0, 0 if wt < 0 or sum_squares < 1

            return result

        # coefficient = (-(n * sigma) / (pt.sqrt(2 * pt.pi) * sigma**3)) * pt.exp(
        #     -((n * sigma) ** 2) / (2 * sigma**2)
        # )
        coefficient = ((n * delta_w) / (pt.sqrt(2 * pt.pi) * sigma**3)) * pt.exp(
            (-((n * delta_w) ** 2)) / (2 * sigma**2)
        )

        # print(function(w+delta_w/2))
        # print(function(w+delta_w/2))
        # if pt.isnan(function(w + delta_w/2)):
        #     print("w + ∆w/2 is ", w+delta_w/2)
        # if pt.isnan(function(w - delta_w/2)):
        #     print("w - ∆w/2 is ", w-delta_w/2)
        sum += coefficient * function(w) * delta_w
    # sum = pt.where(sum < 1, 0, sum)

    return pt.log(sum)


def main():
    # Get the directory this code file is stored in
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # find the path for the data source;  this should work on everyone's system now
    # dataset = "/isotropic_sims/10000/data_3957522615761_xx_0.8_yy_0.8_zz_0.8.csv"
    # dataset = "/isotropic_sims/10000/data_3957522615600_xx_1.2_yy_1.2_zz_1.2.csv"
    dataset = "/generated_sources.csv"

    dataSource = dir_path + dataset

    print(f"Running on PyMC v{pm.__version__}")

    # Import data from file
    dataAll = importCSV(dataSource)
    radec_data = [sublist[1:3] for sublist in dataAll]
    wt_data = [sublist[3] for sublist in dataAll]
    wt_min = np.min(wt_data)
    # wc_min = np.sqrt(1 - wt_min**2)

    model = pm.Model()

    with model:
        # Priors for unknown model parameters.  I defined q to be wc - 1, to avoid
        # confusion between the model parameter and the inverse speed of light as a
        # function of the parameters.

        q = pm.TruncatedNormal("q", sigma=0.01, lower=-1)
        wc = q + 1
        # sigma = pm.Data("sigma_obs", sigma_array)
        # Expected value of wc, in terms of unknown model parameters and observed "X" values.
        # Right now this is very simple.  Eventually it will need to accept more parameter
        # values, along with RA & declination.

        # Likelihood (sampling distribution) of observations
        wt_obs = pm.CustomDist("wt_obs", wc, observed=wt_data, logp=loglike)
        step = pm.Metropolis()
        trace = pm.sample(9000, tune=1000, step=step)
    # summ = az.summary(trace)
    # print(summ)
    az.plot_trace(trace)
    plt.gcf().suptitle("sigma = " + str(sigma), fontsize=16)
    plt.show()
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
