import numpy as np
import matplotlib.pyplot as plt
import pytensor
import pytensor.tensor as pt

from scripts.simulationImport import importCSV
import os

# n_val = 15


def make_plot_like(n_val_default, ax, bound_min, bound_max, scale):
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
        delta_v = sigma / 3

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
        return pt.sum(pt.log(P_obs + 1e-9))

        # return sum

    dir_path = os.path.dirname(os.path.realpath(__file__))

    # find the path for the data source;  this should work on everyone's system now
    # dataset = "/isotropic_sims/10/data_3959143911168_xx_0.8_yy_0.8_zz_0.8.csv"
    # dataset = "/isotropic_sims/10000/data_3957522615761_xx_0.8_yy_0.8_zz_0.8.csv"
    # dataset = "/isotropic_sims/10000/data_3957522615600_xx_1.2_yy_1.2_zz_1.2.csv"
    dataset = "/mojave_cleaned.csv"
    # dataset = "/isotropic_sims/10/data_3959143911168_xx_1.2_yy_1.2_zz_1.2.csv"
    dataSource = dir_path + dataset

    dataAll = importCSV(dataSource)
    radec_data = [sublist[1:3] for sublist in dataAll]
    vt_data = [sublist[3] for sublist in dataAll]
    vt_data_noNaN = [value for value in vt_data if not np.isnan(value)]
    sigma_default = [sublist[4] for sublist in dataAll]
    sigma_default_noNaN = [value for value in sigma_default if not np.isnan(value)]
    # sigma_default = np.random.uniform(low=0.01, high=0.5, size=len(vt_data)).tolist()
    vt_data_with_sigma = np.stack([vt_data_noNaN, sigma_default_noNaN], axis=1)

    vt_sym = pt.dvector("vt")
    wc_sym = pt.dscalar("wc")

    f_loglike = pytensor.function(
        [vt_sym, wc_sym],
        loglike(vt_sym, wc_sym),
        # mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=False),
    )

    # wt_array = np.linspace(0, 3, 75)
    vt_array = vt_data_with_sigma
    # wt_array = 1.2
    q_array = np.linspace(bound_min, bound_max, 200)  # 100 values for wc
    # wc_array = [1.2]

    VT, WC = np.meshgrid(vt_array, q_array + 1)

    Z = np.empty(VT.shape)

    for i in range(VT.shape[0]):
        for j in range(VT.shape[1]):
            vt_val = VT[i, j]
            wc_val = WC[i, j]
            Z[i, j] = f_loglike(vt_val, wc_val)

    Z_combine = np.sum(Z, axis=1)
    Z_max = np.nanmax(Z_combine)
    # Z_min = np.min()

    # print(Z_combine)
    print(Z_max)

    # Z_collapsed = np.trapz(Z, x=wt_array, axis=1)

    ax.plot(
        q_array,
        scale * np.exp(Z_combine - Z_max),
        marker="",
        linestyle="dashed",
        label=f"log-like (σ=varied, n={n_val_default})",
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


def main():
    sigma_val = 0.01
    n_val = 12

    fig, ax = plt.subplots()
    bound_min = -1
    bound_max = 1
    scale = 1
    make_plot_like(n_val, ax, bound_min, bound_max, scale)

    plt.show()


if __name__ == "__main__":
    main()
