#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anisotropic model main code.

- Regenerates a simulated dataset or reads from an actual (MOJAVE) dataset.
- Loads velocity and sigma measurements from CSV.
- Constructs a PyMC probabilistic model involving Bº, B_vec, and wc.
- Defines a custom log-likelihood for vt and sigma pairs.
- Runs MCMC sampling.
- Produces visualization plots.
"""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import os

from aniso_simulated_data_gen_v import regenerate_data
from simulationImport import importCSV
from matplotlib import colors as mcolors


# =============================================================================
# Configuration  —  adjust these before each run
# =============================================================================

# -- PyTensor compiler --------------------------------------------------------
# Prof. Seifert needs the code line below to run PyMC on his machine
# Please just comment out instead of deleting it!
pytensor.config.cxx = "/usr/bin/clang++"

# -- Dataset ------------------------------------------------------------------
# Choose one:  "generated_sources.csv"  or  "mojave_cleaned.csv"
DATASET = "generated_sources.csv"

# -- Likelihood ---------------------------------------------------------------
# Integration resolution for the likelihood summation (higher = slower but more accurate)
N_VAL = 20

# -- Outlier filtering --------------------------------------------------------
# Set True to remove the top 10% highest-velocity sources before sampling
ENABLE_STRIP_TOP_10_PERCENT = False

# -- MCMC sampler -------------------------------------------------------------
DRAWS = 1000
TUNE = 1000
TARGET_ACCEPT = 0.93
CHAINS = 4
CORES = 4
INIT_METHOD = "jitter+adapt_diag"
# Fixed random seed for reproducibility during debugging.
# Set to None for production runs.
RANDOM_SEED = 42


# =============================================================================
# Log-likelihood
# =============================================================================

def loglike(
    vt_stack: pt.TensorVariable,
    wc: pt.TensorVariable,
    n_val=N_VAL,
) -> pt.TensorVariable:
    r"""
    Custom PyMC log-likelihood for transverse velocities.

    $$
    \mathcal{P}_{\mathrm{obs}}(v_t) \approx\sum_{v = v_t - m\sigma_v}^{\,v_t + m\sigma_v}\frac{v - v_t}{\sqrt{2\pi}\,\sigma_v^{3}}\exp\!\left[-\frac{(v - v_t)^2}{2\sigma_v^{2}}\right]C_v(v)\,\Delta v
    $$

    Parameters
    ----------
    vt_stack : TensorVariable
        A 2-column array where:
        - index 0 contains observed transverse velocities `vt`
        - index 1 contains corresponding uncertainties `sigma`
    wc : TensorVariable
        Inverse of the true velocity derived from Bº, B_vec, and n_hat.
    n_val : int, optional
        Integration resolution for the likelihood.

    Returns
    -------
    TensorVariable
        The summed log-likelihood of all observations.

    Notes
    -----
    - Uses clipping to avoid log(0) situations.

    This function is passed to `pm.CustomDist` and must therefore return
    a **scalar** log-probability for the entire dataset.
    """

    vt = vt_stack[:, 0]
    sigma = vt_stack[:, 1]
    delta_v = sigma

    sigma_bcast = sigma[:, None]

    # Broadcast wc per observation across the integration axis
    wc_bcast = (
        pt.flatten(wc)[:, None] if wc.ndim > 0 else pt.ones_like(vt)[:, None] * wc
    )

    v_offsets = pt.linspace(-n_val, n_val, 2 * n_val + 1)

    vt_bcast = vt[:, None]
    delta_v_bcast = delta_v[:, None]
    v_offsets_bcast = v_offsets[None, :]

    v = vt_bcast + v_offsets_bcast * delta_v_bcast
    v_minus_vt = v - vt_bcast
    v_plus_vt = v + vt_bcast

    # This is the CDF function or C(v) in the distribution notes
    def CDF_function(v_inner):

        w_inner = pt.switch(pt.eq(v_inner, 0), 0, pt.pow(v_inner, -1))
        wt_squared = pt.pow(w_inner, 2)
        wc_squared = pt.pow(wc_bcast, 2)
        sum_squares = wc_squared + wt_squared

        # First expression (used when wc < 1)
        square_root = pt.sqrt(pt.clip(sum_squares - 1, 1e-12, np.inf))
        at = pt.arctan(square_root)

        numer1 = w_inner * (square_root - at)
        expr1 = 1 - numer1 / sum_squares

        at2 = pt.arctan(w_inner / wc_bcast)
        numer2 = w_inner**2 - w_inner * at2
        expr2 = 1 - numer2 / sum_squares

        fastslowcondition = pt.lt(wc_bcast, 1)

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

    # Multiply by Δv to get the final probability for each observation.
    P_obs = sum_over_v * delta_v
    return pt.sum(pt.log(pt.clip(P_obs, 1e-12, np.inf)))


# =============================================================================
# Main workflow
# =============================================================================

def main():
    """
    Run the full anisotropy analysis workflow.

    Steps
    -----
    1. Regenerate or load observational data (vt, sigma, n_hat).
    2. Clean NaNs and optionally remove top-percentile velocity outliers.
    3. Construct a PyMC model:
       - Bº ~ HalfNormal
       - B_vec defined via a normalized reparameterization of raw vector
       - wc expression from model geometry
       - Custom vt likelihood using ``loglike``
    4. Sample the posterior using NUTS with configured tuning settings.
    5. Print summaries and diagnostics.
    6. Produce optional Mollweide sky maps and posterior trace plots.
    """

    dir_path = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.abspath(os.path.join(dir_path, os.pardir))

    # ---- Data loading -------------------------------------------------------
    regenerate_data()

    dataSource = os.path.join(project_root, DATASET)

    print(f"Running on PyMC v{pm.__version__}")
    if DATASET == "generated_sources.csv":
        filetype_choice = "Simulated"
    elif DATASET == "mojave_cleaned.csv":
        filetype_choice = "Mojave"

    dataAll = importCSV(dataSource, filetype=filetype_choice)
    radec_data = [sublist[0:3] for sublist in dataAll]

    vt_data = [sublist[3] for sublist in dataAll]
    sigmas = [sublist[4] for sublist in dataAll]

    # Print the top 10 highest velocities from vt_data (descending)
    try:
        vt_arr = np.asarray(vt_data, dtype=float)
        vt_arr = vt_arr[~np.isnan(vt_arr)]
        if vt_arr.size:
            top10 = np.sort(vt_arr)[-10:][::-1]
            print(
                "Top 10 vt_data velocities (desc):",
                np.array2string(top10, precision=3, separator=", "),
            )
    except Exception as _e:
        print(f"Failed to compute top 10 vt_data velocities: {_e}")

    # ---- NaN masking & outlier stripping ------------------------------------
    vt_and_sigma = np.stack(
        [vt_data, sigmas],
        axis=1,
    )

    mask = ~np.isnan(vt_and_sigma).any(axis=1)
    vt_and_sigma_noNaN = vt_and_sigma[mask]

    idx_keep = None
    if ENABLE_STRIP_TOP_10_PERCENT:
        try:
            vt_vals = vt_and_sigma_noNaN[:, 0]
            thresh = np.percentile(vt_vals, 80)
            idx_keep = vt_vals <= thresh
            removed = int(vt_vals.size - np.sum(idx_keep))
            print(
                f"Top 10% strip enabled: threshold={thresh:.3f}, removed {removed}/{vt_vals.size}"
            )
        except Exception as _e:
            print(f"Top 10% strip computation failed: {_e}")
            idx_keep = None

    vt_data_with_sigma = vt_and_sigma_noNaN
    if idx_keep is not None:
        vt_data_with_sigma = vt_data_with_sigma[idx_keep]

    # Load observed n_hat from generated CSV (first three columns), align with NaN mask
    # TODO incorporate this into simulationImport
    n_hat_full = np.genfromtxt(
        dataSource, delimiter=",", skip_header=1, usecols=(0, 1, 2)
    )
    n_hats = n_hat_full[mask]
    if idx_keep is not None:
        n_hats = n_hats[idx_keep]

    # ---- PyMC model ---------------------------------------------------------
    model = pm.Model()

    with model:

        n_hat_data = pm.Data("n_hat_data", n_hats)
        Bº = pm.HalfNormal("Bº", sigma=3)

        # Reparameterize B_vec to keep ||B_vec|| < 1
        b_raw = pm.Normal("b_raw", mu=0, sigma=1, shape=3)
        r_raw = pt.sqrt(pt.sum(b_raw**2))
        u = b_raw / (r_raw + 1e-9)
        rho = pm.Beta("rho", alpha=2, beta=2)
        B_vec = pm.Deterministic("B_vec", rho * u)
        start_point = {"Bº": 0.1, "b_raw": np.zeros(3), "rho": 0.5}

        B_n = pm.math.dot(n_hat_data, B_vec)

        # wc from quadratic solver (δ=-1): positive root of -(B0²+1)wc² - 2B0·Bn·wc + (1-Bn²) = 0
        wc_expr = (
            -Bº * B_n + pt.sqrt(pt.clip(1 + Bº**2 - B_n**2, 1e-12, np.inf))
        ) / (1 + Bº**2)

        # Likelihood (sampling distribution) of observations
        vt_obs = pm.CustomDist(
            "vt_obs",
            wc_expr,
            observed=vt_data_with_sigma,
            logp=loglike,
        )

        trace = pm.sample(
            draws=DRAWS,
            tune=TUNE,
            target_accept=TARGET_ACCEPT,
            chains=CHAINS,
            cores=CORES,
            init=INIT_METHOD,
            random_seed=RANDOM_SEED,
            var_names=["Bº", "B_vec"],
            initval=start_point,
        )

    # ---- Diagnostics --------------------------------------------------------
    summ = az.summary(trace)
    print(summ)

    summary_with_quartiles = az.summary(
        trace,
        stat_funcs={
            "25%": lambda x: np.percentile(x, 25),
            "50%": lambda x: np.percentile(x, 50),
            "75%": lambda x: np.percentile(x, 75),
        },
    )

    try:
        div_total = int(trace.sample_stats["diverging"].values.sum())
    except Exception:
        div_total = -1
    try:
        max_tree = int(trace.sample_stats["tree_depth"].values.max())
    except Exception:
        max_tree = -1

    print("Sampler diagnostics:")
    print(f"  Divergences total: {div_total}")
    print(f"  Max tree depth: {max_tree}")
    print(summary_with_quartiles)

    # ---- Plots --------------------------------------------------------------

    # Mollweide scatter plot of observed speeds vt by direction
    try:
        vt_obs_vals = vt_data_with_sigma[:, 0].astype(float)
        nh = np.asarray(n_hats, dtype=float)

        # Ensure arrays are aligned in length (defensive)
        m = min(len(vt_obs_vals), nh.shape[0])
        vt_obs_vals = vt_obs_vals[:m]
        nh = nh[:m]

        # Convert Cartesian (x,y,z) on unit sphere to sky coords for Mollweide
        x, y, z = nh[:, 0], nh[:, 1], nh[:, 2]
        lon = -np.arctan2(y, x)  # RA-style: flip sign for conventional sky orientation
        lon = (lon + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
        lat = np.arcsin(np.clip(z, -1.0, 1.0))

        # Plot only v > 1
        mask_gt1 = np.asarray(vt_obs_vals) > 1.0
        if np.any(mask_gt1):
            lon_f = lon[mask_gt1]
            lat_f = lat[mask_gt1]
            vt_f = vt_obs_vals[mask_gt1]

            vmin = float(np.nanmin(vt_f))
            vmax = float(np.nanmax(vt_f))
            if not np.isfinite(vmin) or not np.isfinite(vmax):
                raise ValueError("Non-finite vt values for color scaling")
            if vmin == vmax:
                vmax = vmin + 1e-12
            norm = mcolors.PowerNorm(gamma=1.0, vmin=vmin, vmax=vmax)

            fig = plt.figure(figsize=(10, 5))
            ax_mw = fig.add_subplot(111, projection="mollweide")
            sc = ax_mw.scatter(
                lon_f,
                lat_f,
                c=vt_f,
                s=12,
                cmap="viridis",
                norm=norm,
                alpha=0.9,
                edgecolors="none",
            )
            ax_mw.grid(True, linestyle=":", alpha=0.6)
            ax_mw.set_xticklabels([])
            ax_mw.set_yticklabels([])
            ax_mw.tick_params(labelbottom=False, labelleft=False)

            ax_mw.set_title("Observed speeds by direction (Mollweide, v > 1)", pad=18)
            cbar = fig.colorbar(
                sc, ax=ax_mw, orientation="horizontal", pad=0.06, fraction=0.06
            )
            cbar.set_label("Observed speed v_t (v > 1)")
            plt.show()
            plt.close(fig)
            print("Mollweide success")
        else:
            print("No velocities above 1 to plot; skipping Mollweide.")
    except Exception as e:
        print(f"Failed to produce Mollweide plot: {e}")

    az.plot_pair(
        trace,
        var_names=["Bº", "B_vec"],
        kind="kde",
        divergences=True,
        textsize=18,
    )

    try:
        axes = az.plot_trace(trace, combined=False, legend=True)
        plt.savefig(os.path.join(dir_path, "trace.png"), dpi=150, bbox_inches="tight")
        plt.show()
        plt.close()
        az.plot_posterior(trace, round_to=3, figsize=[8, 4], textsize=10)
        plt.show()
        plt.close()
    except Exception as e:
        print(f"Plot saving failed: {e}")


if __name__ == "__main__":
    main()
