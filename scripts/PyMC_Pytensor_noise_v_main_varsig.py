#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Isotropic model main code.

Samples a single scalar parameter q (wc = q + 1) using a custom log-likelihood
for transverse velocity data.  Runs on MOJAVE or simulated data.

@author: mseifer1
"""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import os

from simulationImport import importCSV

# Prof. Seifert needs the code line below to run PyMC on his machine
# Please just comment out instead of deleting it!
pytensor.config.cxx = "/usr/bin/clang++"

n_val_default = 20


def loglike(
    vt_stack: pt.TensorVariable,
    wc: pt.TensorVariable,
    n_val=n_val_default,
) -> pt.TensorVariable:
    vt = vt_stack[:, 0]
    sigma = vt_stack[:, 1]
    delta_v = sigma

    sigma_bcast = sigma[:, None]

    v_offsets = pt.linspace(-n_val, n_val, 2 * n_val + 1)

    vt_bcast = vt[:, None]
    delta_v_bcast = delta_v[:, None]
    v_offsets_bcast = v_offsets[None, :]

    v = vt_bcast + v_offsets_bcast * delta_v_bcast
    v_minus_vt = v - vt_bcast
    v_plus_vt = v + vt_bcast

    def CDF_function(v_inner):

        w_inner = pt.switch(pt.eq(v_inner, 0), 0, pt.pow(v_inner, -1))
        wt_squared = pt.pow(w_inner, 2)
        wc_squared = pt.pow(wc, 2)
        sum_squares = wc_squared + wt_squared

        # First expression (used when wc < 1)
        square_root = pt.sqrt(sum_squares - 1)
        at = pt.arctan(square_root)

        numer1 = w_inner * (square_root - at)
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

    coefficient = (
        v_minus_vt * pt.exp(-(v_minus_vt**2) / (2 * sigma_bcast**2))
        + v_plus_vt * pt.exp(-(v_plus_vt**2) / (2 * sigma_bcast**2))
    ) / (pt.sqrt(2 * pt.pi) * sigma_bcast**3)

    summand = coefficient * CDF_function(v)
    sum_over_v = pt.sum(summand, axis=1)

    P_obs = sum_over_v * delta_v
    return pt.sum(pt.log(pt.clip(P_obs, 1e-12, np.inf)))


def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.abspath(os.path.join(dir_path, os.pardir))

    dataset = "generated_sources.csv"
    # dataset = "mojave_cleaned.csv"

    dataSource = os.path.join(project_root, dataset)

    print(f"Running on PyMC v{pm.__version__}")
    if dataset == "generated_sources.csv":
        filetype_choice = "Simulated"
    elif dataset == "mojave_cleaned.csv":
        filetype_choice = "Mojave"

    dataAll = importCSV(dataSource, filetype=filetype_choice)

    vt_data = [sublist[3] for sublist in dataAll]
    sigmas = [sublist[4] for sublist in dataAll]

    vt_and_sigma = np.stack(
        [vt_data, sigmas],
        axis=1,
    )

    vt_and_sigma_noNaN = vt_and_sigma[~np.isnan(vt_and_sigma).any(axis=1)]
    vt_data_with_sigma = vt_and_sigma_noNaN

    model = pm.Model()

    with model:
        # q = wc - 1, to avoid confusion between the model parameter
        # and the inverse speed of light as a function of the parameters.
        q = pm.TruncatedNormal("q", sigma=3, lower=-1)
        wc = q + 1

        vt_obs = pm.CustomDist("vt_obs", wc, observed=vt_data_with_sigma, logp=loglike)

        trace = pm.sample(1000, tune=1000, target_accept=0.95)

    summary_with_quartiles = az.summary(
        trace,
        stat_funcs={
            "25%": lambda x: np.percentile(x, 25),
            "50%": lambda x: np.percentile(x, 50),
            "75%": lambda x: np.percentile(x, 75),
        },
    )

    axes = az.plot_trace(trace, combined=False)
    plt.savefig(os.path.join(dir_path, "trace_iso.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(summary_with_quartiles)


if __name__ == "__main__":
    main()
