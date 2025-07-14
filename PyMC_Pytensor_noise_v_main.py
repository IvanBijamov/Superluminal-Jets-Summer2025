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

from csv_file_imp_v import regenerate_data
from simulationImport import importCSV
from graphofloglike_v import make_plot_like

pytensor.config.exception_verbosity = "high"


sigma_default = 0.1

regenerate_data(sigma_default)

n_val_default = 20


def loglike_borked(
    vt: pt.TensorVariable,
    wc: pt.TensorVariable,
    sigma=sigma_default,
    n_val=n_val_default,
) -> pt.TensorVariable:

    # configurables
    delta_v = sigma / 3
    
    # Define offsets relative to vt in Riemann sum
    v_range = pt.linspace(-n_val * delta_v, n_val * delta_v, 2 * n_val + 1)

    vt_bcast = vt[:, None]  # Shape becomes (n_obs,  1)

    # `v_range` is a 1D vector (shape: (n _steps,)). Reshape it to a row.
    v_range_bcast = v_range[None, :]  # Shape becomes (1, n_steps)

    # Broadcasting adds them to a 2D grid of shape (n_obs, n_steps)
    v = vt_bcast + v_range_bcast
    v_minus_vt = v - vt_bcast
    v_plus_vt = v + vt_bcast

    # need to somehow get w in here as a pytensor thing using pytensor computations
    # check if inputs are corret
    # Compute squared terms
    # TODO fix naming, it's very jack hammered atm

    def CDF_function(v_inner):

        w_inner = pt.pow(v_inner, -1)
        wt_squared = pt.pow(w_inner, 2)
        wc_squared = pt.pow(wc, 2)
        sum_squares = wc_squared + wt_squared

        # First expression (used when wc < 1)
        square_root = pt.sqrt(sum_squares - 1)
        at = pt.arctan(square_root)

        numer1 = w_inner * (square_root - at)
        # denom = sum_squares
        expr1 = 1 - numer1 / sum_squares

        at2 = pt.arctan(w_inner / wc)
        numer2 = w_inner**2 - w_inner * at2
        expr2 = 1 - numer2 / sum_squares

        fastslowcondition = pt.lt(wc, 1)

        result = pt.switch(fastslowcondition, expr1, expr2)
        # result is now: 
            # expr1 if wc < 1
            # expr2 if wc > 1

        sumsquarecondition = pt.lt(sum_squares, 1)
        result = pt.switch(sumsquarecondition, 1, result)
        # result is now:    
            # 1 if wc < 1 and wt^2 + wc^2 < 1
            # expr1 if wc < 1 and wt^2 + wc^2 > 1
            # expr2 if wc > 1

        negvcondition = pt.lt(v_inner, 0)
        result = pt.switch(negvcondition, 0, result)
        # result is now:    
            # 0 if vt < 0
            # 1 if wc < 1 and wt^2 + wc^2 < 1
            # expr1 if wc < 1 and wt^2 + wc^2 > 1
            # expr2 if wc > 1

        return result

    # coefficient = (w_minus_wt / (pt.sqrt(2 * pt.pi) * sigma**3)) * pt.exp(
    #     (-(w_minus_wt**2)) / (2 * sigma**2)
    # )
    
    coefficient = ( 
                    v_minus_vt * pt.exp( - (v_minus_vt**2) / (2 * sigma**2) )
                    + v_plus_vt * pt.exp( - (v_plus_vt**2) / (2 * sigma**2) )
                   ) / ( 
                       pt.sqrt(2 * pt.pi) * sigma**3 
                       )
    
    summand = coefficient * CDF_function(v)
    sum_over_v = pt.sum(summand, axis=1)

    # Final Probability and Log -Probability
    # Multiply by Δw to get the final probability for each observation.
    P_obs = sum_over_v * delta_v
    return pt.sum(pt.log(P_obs + 1e-9))


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
    vt_data = [sublist[3] for sublist in dataAll]
    print(vt_data[:10])
    wt_data = np.pow(vt_data, -1.0)
    # wc_min = np.sqrt(1 - wt_min**2)

    model = pm.Model()

    with model:
        # Priors for unknown model parameters.  I defined q to be wc - 1, to avoid
        # confusion between the model parameter and the inverse speed of light as a
        # function of the parameters.

        q = pm.TruncatedNormal("q", sigma=3, lower=-1)
        wc = q + 1
        # sigma = pm.Data("sigma_obs", sigma_array)
        # Expected value of wc, in terms of unknown model parameters and observed "X" values.
        # Right now this is very simple.  Eventually it will need to accept more parameter
        # values, along with RA & declination.

        # Likelihood (sampling distribution) of observations
        vt_obs = pm.CustomDist("vt_obs", wc, observed=vt_data, logp=loglike_borked)
        # step = pm.Metropolis()
        trace = pm.sample(1000, tune=1000, target_accept = 0.95)
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
    make_plot_like(sigma_default, n_val_default, left_ax, qmin, qmax, scale)
    # axes = az_plot.axes.flatten()

    plt.show()
    # az.plot_posterior(trace, round_to=3, figsize=[8, 4], textsize=10)

    print(summary_with_quartiles)


if __name__ == "__main__":
    main()
