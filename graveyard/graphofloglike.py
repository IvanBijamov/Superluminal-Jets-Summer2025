import numpy as np
import pytensor
import pytensor.tensor as pt

from scripts.simulationImport import importCSV
import os


# n_val = 15


def make_plot_like(sigma_val, ax, bound_min, bound_max, scale):
    sigma = sigma_val
    n_val = 20

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
        w_vals = pt.linspace(
            wt_regular - n_val * delta_w, wt_regular + n_val * delta_w, 2 * n_val + 1
        )

        def function(wt_input):

            # CDF VERSION

            # wt = w + delta_w / 2

            wt_squared = pt.pow(wt_input, 2)
            wc_squared = pt.pow(wc, 2)
            sum_squares = wc_squared + wt_squared

            # First expression (used when wc < 1)
            square_root = pt.sqrt(sum_squares - 1)
            at = pt.arctan(square_root)

            numer1 = wt_input * (square_root - at)
            denom = sum_squares
            expr1 = numer1 / denom

            at2 = pt.arctan(wt_input / wc)
            numer2 = wt_input**2 - wt_input * at2
            denom2 = sum_squares
            expr2 = numer2 / denom2

            fastslowcondition = pt.lt(wc, 1)

            result = pt.switch(fastslowcondition, expr1, expr2)

            # expr1 if wc < 1, expr2 if wc > 1

            negwtcondition = pt.lt(wt_input, 0)
            sumsquarecondition = pt.lt(sum_squares, 1)

            result = pt.switch(negwtcondition | sumsquarecondition, 0, result)

            return result

        # expr1 if wc < 1 & wt > 0, expr2 if wc > 1 & wt >0, 0 if wt < 0 or sum_squares < 1

        coefficient = (
            (w_vals - wt_regular) / (pt.sqrt(2 * pt.pi) * sigma**3)
        ) * pt.exp((-((w_vals - wt_regular) ** 2)) / (2 * sigma**2))

        total = pt.sum(coefficient * function(w_vals) * delta_w)
        # sum = pt.where(sum < 1, 0, sum)

        return pt.log(total)

    dir_path = os.path.dirname(os.path.realpath(__file__))

    # find the path for the data source;  this should work on everyone's system now
    # dataset = "/isotropic_sims/10/data_3959143911168_xx_0.8_yy_0.8_zz_0.8.csv"
    # dataset = "/isotropic_sims/10000/data_3957522615761_xx_0.8_yy_0.8_zz_0.8.csv"
    # dataset = "/isotropic_sims/10000/data_3957522615600_xx_1.2_yy_1.2_zz_1.2.csv"
    dataset = "/generated_sources.csv"
    # dataset = "/isotropic_sims/10/data_3959143911168_xx_1.2_yy_1.2_zz_1.2.csv"
    dataSource = dir_path + dataset

    dataAll = importCSV(dataSource)
    radec_data = [sublist[1:3] for sublist in dataAll]
    wt_data = [sublist[3] for sublist in dataAll]

    wt_sym = pt.dscalar("wt")
    wc_sym = pt.dscalar("wc")

    f_loglike = pytensor.function(
        [wt_sym, wc_sym],
        loglike(wt_sym, wc_sym),
        # mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=False),
    )

    # wt_array = np.linspace(0, 3, 75)
    wt_array = wt_data
    # wt_array = 1.2
    q_array = np.linspace(bound_min, bound_max, 200)  # 100 values for wc
    # wc_array = [1.2]

    WT, WC = np.meshgrid(wt_array, q_array + 1)

    Z = np.empty(WT.shape)

    for i in range(WT.shape[0]):
        for j in range(WT.shape[1]):
            wt_val = WT[i, j]
            wc_val = WC[i, j]
            Z[i, j] = f_loglike(wt_val, wc_val)

    Z_combine = np.sum(Z, axis=1)
    Z_max = np.max(Z_combine)

    # Z_collapsed = np.trapz(Z, x=wt_array, axis=1)

    ax.plot(
        q_array,
        scale * np.exp(Z_combine - Z_max),
        marker="",
        linestyle="dashed",
        label=f"log-like (σ={sigma_val}, n={n_val})",
        color="red",
        linewidth=2,
    )
    # ax.set_xlabel("q")
    # ax.set_ylabel("scaled likelihood")
    # ax.grid(True)
    # ax.legend()
    #
    # ax.plt.figure(figsize=(8, 6))
    # ax.plt.plot(q_array, np.exp(Z_combine - Z_max), marker="o", linestyle="-")
    # ax.plt.xlabel("q")
    # ax.plt.ylabel("log-like")
    # ax.plt.title("log-like vs. wc - sigma = " + str(sigma) + ", n_val = " + str(n_val))
    # # plt.gcf().suptitle("sigma = " + str(sigma), fontsize=16)
    # ax.plt.grid(True)
    # # plt.show()
    return ax
