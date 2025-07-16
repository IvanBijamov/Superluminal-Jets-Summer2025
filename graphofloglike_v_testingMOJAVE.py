import numpy as np
import matplotlib.pyplot as plt
import pytensor
import pytensor.tensor as pt
from pytensor.compile.nanguardmode import NanGuardMode

from simulationImport import importCSV
import os

# n_val = 15


def make_plot_like(n_val_default, ax, bound_min, bound_max, scale, summed=True):
    # sigma = sigma_val

    def loglike(
        vt_stack: pt.TensorVariable,
        # sigma: pt.TensorVariable,
        wc: pt.TensorVariable,
        n_val=n_val_default,
    ) -> pt.TensorVariable:
        vt = vt_stack[:, 0]
        sigma = vt_stack[:, 1]
        # configurables
        delta_v = sigma / 100

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
            v_minus_vt * pt.exp(-(v_minus_vt**2) / (2 * sigma_bcast**2))
            + v_plus_vt * pt.exp(-(v_plus_vt**2) / (2 * sigma_bcast**2))
        ) / (pt.sqrt(2 * pt.pi) * sigma_bcast**3)

        summand = coefficient * CDF_function(v)
        sum_over_v = pt.sum(summand, axis=1)

        # Final Probability and Log -Probability
        # Multiply by Δw to get the final probability for each observation.
        P_obs = sum_over_v * delta_v
        return pt.log(P_obs + 1e-9)

        # return sum

    dir_path = os.path.dirname(os.path.realpath(__file__))

    # find the path for the data source;  this should work on everyone's system now
    # dataset = "/isotropic_sims/10/data_3959143911168_xx_0.8_yy_0.8_zz_0.8.csv"
    # dataset = "/isotropic_sims/10000/data_3957522615761_xx_0.8_yy_0.8_zz_0.8.csv"
    # dataset = "/isotropic_sims/10000/data_3957522615600_xx_1.2_yy_1.2_zz_1.2.csv"
    dataset = "/mojave_cleaned.csv"
    # dataset = "/generated_sources.csv"
    # dataset = "/isotropic_sims/10/data_3959143911168_xx_1.2_yy_1.2_zz_1.2.csv"
    dataSource = dir_path + dataset

    dataAll = importCSV(dataSource)
    vt_and_sigma = np.stack(
        [[sublist[3] for sublist in dataAll], [sublist[4] for sublist in dataAll]],
        axis=1,
    )

    vt_and_sigma_noNaN = vt_and_sigma[~np.isnan(vt_and_sigma).any(axis=1)]
    
    datasetsize = len(vt_and_sigma_noNaN)

    nstart = 0 # First source
    nsources = 2000 # Number of sources in plot
    nsources = min(datasetsize - nstart, nsources) # In case we try to grab more sources than are available
    vt_data_with_sigma = vt_and_sigma_noNaN[nstart:nstart + nsources]
    vt_stack_sym = pt.dmatrix("vt_stack_sym")
    wc_sym = pt.dscalar("wc_sym")

    f_loglike = pytensor.function(
        [vt_stack_sym, wc_sym],
        loglike(vt_stack_sym, wc_sym),
    )

    ngraphpoints = 100
    q_array = np.linspace(bound_min, bound_max, ngraphpoints)

    log_likelihood_values = np.empty((ngraphpoints,nsources))

    for i, q_val in enumerate(q_array):
        wc_val = q_val + 1

        log_likelihood_values[i,:] = f_loglike(vt_data_with_sigma, wc_val)
    
    # Print any array indices where the log_likelihood is a Nan
    if not(summed):
        print(np.argwhere(np.isnan(log_likelihood_values)))
    
    total_log_likelihood = np.sum(log_likelihood_values, axis=1)

    if summed:
        Z_max = np.nanmax(total_log_likelihood)
        ax.plot(
            q_array,
            total_log_likelihood - Z_max,
            marker="",
            # label="",
        )
    else:
        labels = [str(nstart + i) for i in range(nsources)]
        lw = max(min(1.0, 10/nsources),0.2)
        ax.plot(
            q_array,
            log_likelihood_values,
            marker="",
            label=labels,
            linewidth=lw
        )

    return ax


def main_test():
    """
    Sets up a plot, calls the likelihood plotting function, and displays it.
    """

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    n_val = 20
    q_min = -1.0
    q_max = 2.0
    plot_scale = 1.0
    make_plot_like(
        n_val_default=n_val, ax=ax1, bound_min=q_min, bound_max=q_max, scale=plot_scale,summed=False
    )
    make_plot_like(
        n_val_default=n_val, ax=ax2, bound_min=q_min, bound_max=q_max, scale=plot_scale
    )

    ax1.set_xlabel("q")
    ax1.set_ylabel("Individual log-likelihood")
    ax1.set_title("Individual log-likelihoods of Parameter q")
    ax1.grid(True)
    if len(ax1.lines) <= 10:
        ax1.legend()

    ax2.set_xlabel("q")
    ax2.set_ylabel("Total log-likelihood")
    ax2.set_title("Total log-likelihood of Parameter q")
    ax2.grid(True)
    # ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    main_test()
