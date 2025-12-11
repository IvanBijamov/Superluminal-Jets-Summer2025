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
from matplotlib import colors as mcolors
from graphofloglike_v import make_plot_like

# pytensor.config.exception_verbosity = "high"

# Prof. Seifert needs the code line below to run PyMC on his machine
# Please just comment out instead of deleting it!
pytensor.config.cxx = "/usr/bin/clang++"

# pytensor.config.mode = "NanGuardMode"


# sigma_default = 0.1


n_val_default = 20


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

    # need to somehow get w in here as a pytensor thing using pytensor computations
    # check if inputs are corret
    # Compute squared terms
    # TODO fix naming, it's very jack hammered atm

    def CDF_function(v_inner):

        w_inner = pt.switch(pt.eq(v_inner, 0), 0, pt.pow(v_inner, -1))
        wt_squared = pt.pow(w_inner, 2)
        wc_squared = pt.pow(wc_bcast, 2)
        sum_squares = wc_squared + wt_squared

        # First expression (used when wc < 1)
        square_root = pt.sqrt(pt.clip(sum_squares - 1, 1e-12, np.inf))
        at = pt.arctan(square_root)

        numer1 = w_inner * (square_root - at)
        # denom = sum_squares
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
    return pt.sum(pt.log(pt.clip(P_obs, 1e-12, np.inf)))


def main():
    # Get the directory this code file is stored in
    dir_path = os.path.dirname(os.path.realpath(__file__))
    regenerate_data()

    # find the path for the data source;  this should work on everyone's system now
    # dataset = "/isotropic_sims/10000/data_3957522615761_xx_0.8_yy_0.8_zz_0.8.csv"
    # dataset = "/isotropic_sims/10000/data_3957522615600_xx_1.2_yy_1.2_zz_1.2.csv"
    # dataset = "/mojave_cleaned.csv"
    dataset = "/generated_sources.csv"

    dataSource = dir_path + dataset

    print(f"Running on PyMC v{pm.__version__}")

    # Import data from file
    dataAll = importCSV(dataSource, filetype="Schindler")
    # dataAll = importCSV(dataSource, filetype="Mojave")
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

    vt_and_sigma = np.stack(
        [vt_data, sigmas],
        axis=1,
    )

    # print(vt_and_sigma[:10])

    mask = ~np.isnan(vt_and_sigma).any(axis=1)
    vt_and_sigma_noNaN = vt_and_sigma[mask]

    # Toggle: set to False to disable stripping the top 5% highest-velocity sources
    ENABLE_STRIP_TOP_10_PERCENT = False

    # Index mask for rows to keep (None means no stripping)
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
            print(f"Top 5% strip computation failed: {_e}")
            idx_keep = None

    # Test that removes sigma values bigger than the data value in case something funky
    # vt_data_with_sigma = vt_and_sigma_noNaN[
    #     vt_and_sigma_noNaN[:, 1] <= vt_and_sigma_noNaN[:, 0]
    # ]
    vt_data_with_sigma = vt_and_sigma_noNaN
    if idx_keep is not None:
        vt_data_with_sigma = vt_data_with_sigma[idx_keep]
    # print(vt_data_with_sigma)
    size = len(vt_data_with_sigma)
    # Load observed n̂ from generated CSV (first three columns), align with NaN mask
    n_hat_full = np.genfromtxt(
        dataSource, delimiter=",", skip_header=1, usecols=(0, 1, 2)
    )
    n_hats = n_hat_full[mask]
    if idx_keep is not None:
        n_hats = n_hats[idx_keep]

    # Mollweide scatter plot of observed speeds vt by direction
    # Uses n_hats (unit vectors) aligned with vt_data_with_sigma after masking/stripping
    try:
        vt_obs_vals = vt_data_with_sigma[:, 0].astype(float)
        nh = np.asarray(n_hats, dtype=float)

        # Ensure arrays are aligned in length (defensive)
        m = min(len(vt_obs_vals), nh.shape[0])
        vt_obs_vals = vt_obs_vals[:m]
        nh = nh[:m]

        # Convert Cartesian (x,y,z) on unit sphere to sky coords for Mollweide
        # Longitude in [-pi, pi], latitude in [-pi/2, pi/2]
        x, y, z = nh[:, 0], nh[:, 1], nh[:, 2]
        lon = -np.arctan2(y, x)  # RA-style: flip sign for conventional sky orientation
        lon = (lon + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
        lat = np.arcsin(np.clip(z, -1.0, 1.0))

        # plot only v > 1

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

            # Plotting
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
            # Remove degree labeling from axes
            ax_mw.set_xticklabels([])
            ax_mw.set_yticklabels([])
            ax_mw.tick_params(labelbottom=False, labelleft=False)

            ax_mw.set_title("Observed speeds by direction (Mollweide, v > 1)", pad=18)
            cbar = fig.colorbar(
                sc, ax=ax_mw, orientation="horizontal", pad=0.06, fraction=0.06
            )
            cbar.set_label("Observed speed v_t (v > 1)")
            plt.show()
            # out_path_mw = os.path.join(dir_path, "mollweide_speeds.png")

            # plt.savefig(out_path_mw, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Mollweide success")
        else:
            print("No velocities above 1 to plot; skipping Mollweide.")
    except Exception as e:
        print(f"Failed to produce Mollweide plot: {e}")
    model = pm.Model()

    with model:

        n_hat_data = pm.Data("n_hat_data", n_hats)
        Bº = pm.HalfNormal("Bº", sigma=3)
        # Bº = pm.TruncatedNormal("Bº", lower=0, upper=1, sigma=3)

        # Reparameterize B_vec to keep ||B_vec|| < 1
        b_raw = pm.Normal("b_raw", mu=0, sigma=1, shape=3)
        r_raw = pt.sqrt(pt.sum(b_raw**2))
        u = b_raw / (r_raw + 1e-9)
        # beta to get r^2 based curve to better match signmoid skewing
        rho = pm.Beta("rho", alpha=3, beta=1)
        r_unit = pm.math.sigmoid(rho)
        B_vec = pm.Deterministic("B_vec", r_unit * u)
        # TODO
        start_point = {"Bº": 0.0, "B_vec": 0.0}
        # find better reparamterization, this seems to work for now though
        B_n = pm.math.dot(n_hat_data, B_vec)

        wc_expr = (-Bº * B_n + pt.sqrt(1 + (Bº**2 - B_n**2) ** 2)) / (1 + Bº**2)

        # Likelihood (sampling distribution) of observations
        vt_obs = pm.CustomDist(
            "vt_obs",
            wc_expr,
            observed=vt_data_with_sigma,
            logp=loglike,
        )
        # step = pm.Metropolis()

        trace = pm.sample(
            draws=1000,
            tune=1000,
            target_accept=0.90,
            chains=4,
            cores=4,
            init="jitter+adapt_diag",
            random_seed=42,
            var_names=["Bº", "B_vec"],
            initval=start_point,
        )

    # summ = az.summary(trace)
    # print(summ)
    # Diagnostics and plots
    summary_with_quartiles = az.summary(
        trace,
        stat_funcs={
            "25%": lambda x: np.percentile(x, 25),
            "50%": lambda x: np.percentile(x, 50),
            "75%": lambda x: np.percentile(x, 75),
        },
    )
    # Plot to establish relation between B0 and B_vec parameter
    az.plot_pair(
        trace,
        var_names=["Bº", "B_vec"],
        kind="kde",
        divergences=True,
        textsize=18,
    )
    # Divergences and tree depth
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

    # Save plots instead of showing (Agg backend)
    try:
        axes = az.plot_trace(trace, combined=False, legend=True)
        plt.savefig(os.path.join(dir_path, "trace.png"), dpi=150, bbox_inches="tight")
        plt.show()
        plt.close()
        az.plot_posterior(trace, round_to=3, figsize=[8, 4], textsize=10)
        plt.show()
        plt.savefig(
            os.path.join(dir_path, "posterior.png"), dpi=150, bbox_inches="tight"
        )
        # plt.show()
        # plt.close()
        # az.plot_energy(trace)
        # plt.show()
        # plt.savefig(os.path.join(dir_path, "energy.png"), dpi=150, bbox_inches="tight")
        # plt.close()
    except Exception as e:
        print(f"Plot saving failed: {e}")


if __name__ == "__main__":
    main()
