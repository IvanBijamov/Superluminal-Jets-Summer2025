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

from csv_file_imp import regenerate_data
from simulationImport import importCSV
from graphofloglike import make_plot_like

pytensor.config.exception_verbosity = "high"


sigma_default = 1

regenerate_data(sigma_default)

n_val_default = 20


def loglike_borked(
    wt: pt.TensorVariable,
    wc: pt.TensorVariable,
    sigma=sigma_default,
    n_val=n_val_default,
) -> pt.TensorVariable:

    # configurables
    delta_w = sigma / 5
    w_range = pt.linspace(-n_val * delta_w, n_val * delta_w, 2 * n_val + 1)

    wt_bcast = wt[:, None]  # Shape becomes (n_obs,  1)

    # `w_range` is a 1D vector (shape: (n _steps,)). Reshape it to a row.
    w_range_bcast = w_range[None, :]  # Shape becomes (1, n_steps)

    # Broadcasting adds them to a 2D grid of shape (n_obs, n_steps)
    w = wt_bcast + w_range_bcast
    w_minus_wt = w - wt_bcast
    # need to somehow get w in here as a pytensor thing using pytensor computations
    # check if inputs are corret
    # Compute squared terms
    # TODO fix naming, it's very jack hammered atm

    def CDF_function(w_inner):

        wt_squared = pt.pow(w_inner, 2)
        wc_squared = pt.pow(wc, 2)
        sum_squares = wc_squared + wt_squared

        # First expression (used when wc < 1)
        square_root = pt.sqrt(sum_squares - 1)
        at = pt.arctan(square_root)

        numer1 = w_inner * (square_root - at)
        denom = sum_squares
        expr1 = numer1 / denom

        at2 = pt.arctan(w_inner / wc)
        numer2 = w_inner**2 - w_inner * at2
        denom2 = sum_squares
        expr2 = numer2 / denom2

        fastslowcondition = pt.lt(wc, 1)

        result = pt.switch(fastslowcondition, expr1, expr2)

        # expr1 if wc < 1, expr2 if wc > 1

        negwtcondition = pt.lt(w_inner, 0)
        sumsquarecondition = pt.lt(sum_squares, 1)

        result = pt.switch(negwtcondition | sumsquarecondition, 0, result)

        # expr1 if wc < 1 & wt > 0, expr2 if wc > 1 & wt >0, 0 if wt < 0 or sum_squares < 1

        return result

    coefficient = (w_minus_wt / (pt.sqrt(2 * pt.pi) * sigma**3)) * pt.exp(
        (-(w_minus_wt**2)) / (2 * sigma**2)
    )
    summand = coefficient * CDF_function(w)
    sum_over_w = pt.sum(summand, axis=1)

    # Final Probability and Log -Probability
    # Multiply by Δw to get the final probability for each observation.
    P_obs = sum_over_w * delta_w
    return pt.sum(pt.log(P_obs + 1e-9))


def solve_wc_pt(δ, Bº, B_vec, n_hat):

    dot = pt.dot(B_vec, n_hat)

    a = δ * Bº**2 - 1
    b = 2 * δ * Bº * dot
    c = δ * dot**2 + 1
    disc = b**2 - 4 * a * c

    wc0 = 1 / pt.sqrt(1 - δ * Bº**2)
    wc1 = (-b + pt.sqrt(disc)) / (2 * a)
    wc2 = (-b - pt.sqrt(disc)) / (2 * a)

    # when B_vec is  zero
    case_zero = pt.switch(
        pt.lt(δ * Bº**2, 1),
        pt.switch(pt.gt(wc0, 0), pt.stack([wc0]), pt.constant([])),
        pt.constant([]),
    )

    # when B_vec is not zero
    roots = pt.stack([wc1, wc2])
    pos_roots = roots[pt.gt(roots, 0)]
    case_nonzero = pt.switch(pt.lt(disc, 0), pt.constant([]), pos_roots)

    return pt.switch(pt.all(pt.eq(B_vec, 0)), case_zero, case_nonzero)


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

        # q = pm.TruncatedNormal("q", sigma=1, lower=-1)
        δ = -0.1
        Bº = 1.0
        B_vec = np.array([1.0, 0.0, 0.0])
        n_hat_base = pm.Normal("n_hat_base", shape=(3, len(wt_data)))

        norm = pt.linalg.norm(n_hat_base, axis=1, keepdims=True)
        n_hat = pm.Deterministic("n_hat", n_hat_base / norm)
        wc = pm.Deterministic("wc", solve_wc_pt(δ, Bº, B_vec, n_hat))
        # wc = q + 1
        # sigma = pm.Data("sigma_obs", sigma_array)
        # Expected value of wc, in terms of unknown model parameters and observed "X" values.
        # Right now this is very simple.  Eventually it will need to accept more parameter
        # values, along with RA & declination.

        # Likelihood (sampling distribution) of observations
        wt_obs = pm.CustomDist("wt_obs", wc, observed=wt_data, logp=loglike_borked)
        # step = pm.Metropolis()
        trace = pm.sample(4000, tune=1000)
    # summ = az.summary(trace)
    # print(summ)
    summary_with_quartiles = az.summary(
        trace,
        stat_funcs={
            "25%": lambda x: np.percentile(x, 25),
            "50%": lambda x: np.percentile(x, 50),
            "75%": lambda x: np.percentile(x, 75),
        },
    )
    axes = az.plot_trace(trace, combined=False)
    plt.gcf().suptitle("sigma = " + str(sigma_default), fontsize=16)

    axes_flat = np.array(axes).flatten()
    left_ax = axes_flat[0]
    # density_axes = axes_flat[0::2]

    qmin, qmax = left_ax.get_xlim()
    dummy, scale = left_ax.get_ylim()

    # TODO: get vertical scale
    make_plot_like(sigma_default, left_ax, qmin, qmax, scale)
    # axes = az_plot.axes.flatten()

    plt.show()
    # az.plot_posterior(trace, round_to=3, figsize=[8, 4], textsize=10)

    print(summary_with_quartiles)


if __name__ == "__main__":
    main()
