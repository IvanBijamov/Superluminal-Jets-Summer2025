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

# pytensor.config.exception_verbosity = "high"

# Prof. Seifert needs the code line below to run PyMC on his machine
# Please just comment out instead of deleting it!
pytensor.config.cxx = "/usr/bin/clang++"

# pytensor.config.mode = "NanGuardMode"


# sigma_default = 0.1

regenerate_data()

n_val_default = 20


def create_unit_vectors(size):

    vecs = np.random.normal(size=(size, 3))

    norms = np.linalg.norm(vecs, axis=1, keepdims=True)

    norms = np.where(norms == 0, 1, norms)
    n_hats = vecs / norms

    zero_vec_mask = (n_hats == 0).all(axis=1)
    n_hats[zero_vec_mask] = [1.0, 0.0, 0.0]
    return np.array(n_hats)


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


def loglike(
    vt_stack: pt.TensorVariable,
    # sigma: pt.TensorVariable,
    wc: pt.TensorVariable,
    n_val=n_val_default,
) -> pt.TensorVariable:
    vt = vt_stack[:, 0]
    sigma = vt_stack[:, 1]
    # configurables
    delta_v = sigma

    sigma_bcast = sigma[:, None]

    v_offsets = pt.linspace(-n_val, n_val, 2 * n_val + 1)

    vt_bcast = vt[:, None]
    delta_v_bcast = delta_v[:, None]
    v_offsets_bcast = v_offsets[None, :]

    v = vt_bcast + v_offsets_bcast * delta_v_bcast
    v_minus_vt = v - vt_bcast
    v_plus_vt = v + vt_bcast

    # need to somehow get w in here as a pytensor thing using pytensor computations
    # check if inputs are corret
    # Compute squared terms
    # TODO fix naming, it's very jack hammered atm

    def CDF_function(v_inner):

        w_inner = pt.switch(pt.eq(v_inner, 0), 0, pt.pow(v_inner, -1))
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
        v_minus_vt * pt.exp(-(v_minus_vt**2) / (2 * sigma_bcast**2))
        + v_plus_vt * pt.exp(-(v_plus_vt**2) / (2 * sigma_bcast**2))
    ) / (pt.sqrt(2 * pt.pi) * sigma_bcast**3)

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
    dataset = "/mojave_cleaned.csv"
    # dataset = "/generated_sources.csv"

    dataSource = dir_path + dataset

    print(f"Running on PyMC v{pm.__version__}")

    # Import data from file
    # dataAll = importCSV(dataSource, filetype="Schindler")
    dataAll = importCSV(dataSource, filetype="Mojave")
    # radec_data = [sublist[1:3] for sublist in dataAll]
    # vt_data = np.array([sublist[3] for sublist in dataAll])
    # vt_data_noNaN = vt_data[~np.isnan(vt_data)]
    # sigma_default = np.array([sublist[4] for sublist in dataAll])
    # sigma_default_noNaN = sigma_default[~np.isnan(sigma_default)]
    # vt_data_with_sigma = np.stack([vt_data_noNaN, sigma_default_noNaN], axis=1)

    # Mojave
    # vt_and_sigma = np.stack(
    #     [[sublist[3] for sublist in dataAll], [sublist[4] for sublist in dataAll]],
    #     axis=1,
    # )
    # Warren
    vt_data = [sublist[3] for sublist in dataAll]
    sigmas = [sublist[4] for sublist in dataAll]
    # sigmas = [0.2] * len(vt_data)
    vt_and_sigma = np.stack(
        [vt_data, sigmas],
        axis=1,
    )

    # print(vt_and_sigma[:10])

    vt_and_sigma_noNaN = vt_and_sigma[~np.isnan(vt_and_sigma).any(axis=1)]

    # Test that removes sigma values bigger than the data value in case something funky
    # vt_data_with_sigma = vt_and_sigma_noNaN[
    #     vt_and_sigma_noNaN[:, 1] <= vt_and_sigma_noNaN[:, 0]
    # ]
    vt_data_with_sigma = vt_and_sigma_noNaN
    # print(vt_data_with_sigma)
    size = len(vt_data_with_sigma)
    n_hats = create_unit_vectors(size)

    model = pm.Model()

    with model:
        # Priors for unknown model parameters.  I defined q to be wc - 1, to avoid
        # confusion between the model parameter and the inverse speed of light as a
        # function of the parameters.

        # q = pm.TruncatedNormal("q", sigma=3, lower=-1)
        # wc = q + 1

        n_hat_data = pm.Data("n_hat_data", n_hats)
        Bº = pm.HalfNormal("Bº", sigma=10)

        # bx = pm.Normal("bx", mu=0, sigma=10)
        # by = pm.Normal("by", mu=0, sigma=10)
        # bz = pm.Normal("bz", mu=0, sigma=10)
        B_vec = pm.Normal("B_vec", mu=0, sigma=10, shape=3)
        # TODO figure out how to do B_vec stuff with dot product
        B_n = pm.math.dot(n_hat_data, B_vec)
        # B_n = pt.sum(B_vec * n_hats, axis=1)

        wc = pm.Deterministic(
            "wc", ((-Bº * B_n + pt.sqrt(1 + (Bº**2 - B_n**2) ** 2)) / (1 + Bº**2))
        )
        # sigma = pm.Data("sigma_obs", sigma_array)
        # Expected value of wc, in terms of unknown model parameters and observed "X" values.
        # Right now this is very simple.  Eventually it will need to accept more parameter
        # values, along with RA & declination.

        # Likelihood (sampling distribution) of observations
        vt_obs = pm.CustomDist("vt_obs", wc, observed=vt_data_with_sigma, logp=loglike)
        # step = pm.Metropolis()

        print(model.debug(verbose=True))

        trace = pm.sample(2000, tune=1000, target_accept=0.90)

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
    sigma_temp = 0.1
    axes = az.plot_trace(trace, combined=False)
    # plt.gcf().suptitle("sigma = " + str(sigma_temp), fontsize=16)
    #
    # axes_flat = np.array(axes).flatten()
    # left_ax = axes_flat[0]
    # # density_axes = axes_flat[0::2]
    #
    # qmin, qmax = left_ax.get_xlim()
    # dummy, scale = left_ax.get_ylim()
    #
    # # TODO: get vertical scale
    # # temp for plot
    #
    # make_plot_like(n_val_default, left_ax, qmin, qmax, scale)
    # axes = az_plot.axes.flatten()
    # plt.savefig("Mojaveplot.pdf")
    # plt.close()
    # plt.show()
    az.plot_posterior(trace, round_to=3, figsize=[8, 4], textsize=10)

    print(summary_with_quartiles)


if __name__ == "__main__":
    main()
