import os

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt


pytensor.config.exception_verbosity = "high"

from simulationImport import importCSV


def loglike_broadcast(
    wt_reg: pt.TensorVariable,
    sigma: pt.TensorVariable,
    wc: pt.TensorVariable,
    n_val: int = 10,
    delta_w: float = 0.2,
) -> pt.TensorVariable:

    n_vec = pt.arange(-n_val, n_val)
    n_mat = n_vec.dimshuffle(0, "x")
    wt_mat = wt_reg.dimshuffle("x", 0)
    sig_mat = sigma.dimshuffle("x", 0)
    wc_mat = wc.dimshuffle("x", 0) if wc.ndim > 0 else wc

    w = wt_mat + n_mat * sig_mat

    w2 = w**2
    wc2 = wc_mat**2
    s2 = w2 + wc2
    root = pt.sqrt(s2 - 1.0)

    # expr1: wc < 1
    at1 = pt.arctan(root)
    expr1 = (w * (root - at1)) / s2

    # expr2: wc >= 1
    at2 = pt.arctan(w / wc_mat)
    expr2 = (w**2 - w * at2) / s2

    # pick branch
    expr = pt.switch(pt.lt(wc_mat, 1.0), expr1, expr2)

    # zero out invalid regions (wt<0 or sum_squares<1)
    mask = pt.lt(w, 0.0) | pt.lt(s2, 1.0)
    f_mat = pt.switch(mask, 0.0, expr)  # (2*n_val, N)

    # coefficient for each slice
    coeff = (n_mat * sig_mat) / (pt.sqrt(2 * pt.pi) * sig_mat**3)
    coeff = coeff * pt.exp(-0.5 * (n_mat * sig_mat) ** 2 / sig_mat**2)

    # weighted sum over the n‐axis, then log
    weighted = coeff * f_mat * delta_w  # (2*n_val, N)
    summed = pt.sum(weighted, axis=0)  # (N,)
    return pt.log(summed)  # (N,)


def main():
    # locate data file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dataset = "/isotropic_sims/10000/data_3957522615761_xx_0.8_yy_0.8_zz_0.8.csv"
    dataSource = dir_path + dataset

    print(f"Running on PyMC v{pm.__version__}")

    dataAll = importCSV(dataSource)
    wt_data = np.array([row[3] for row in dataAll], dtype=float)
    sigma_array = np.full_like(wt_data, 0.2)  # same σ for each wt

    # build the model
    model = pm.Model()
    with model:

        q = pm.TruncatedNormal("q", mu=0.0, sigma=0.01, lower=-1.0)
        wc = q + 1.0

        wt_reg = pm.Data("wt_reg", wt_data)
        sigma = pm.Data("sigma", sigma_array)

        pm.CustomDist(
            name="wt_obs",
            theta=sigma,
            lam=wc,
            observed=wt_reg,
            logp=loglike_broadcast,
        )

        trace = pm.sample(4000, tune=1000, return_inferencedata=True)

    az.plot_trace(trace, compact=True)
    az.plot_posterior(trace, round_to=3, figsize=(8, 4))
    summary = az.summary(
        trace,
        stat_funcs={
            "25%": lambda x: np.percentile(x, 25),
            "50%": lambda x: np.percentile(x, 50),
            "75%": lambda x: np.percentile(x, 75),
        },
    )
    print(summary)

    plt.show()


if __name__ == "__main__":
    main()
