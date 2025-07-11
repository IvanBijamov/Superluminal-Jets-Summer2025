import numpy as np
import matplotlib.pyplot as plt
import pytensor
import pytensor.tensor as pt
from pytensor.compile.nanguardmode import NanGuardMode

from simulationImport import importCSV
import os

# n_val = 15


def make_plot_like(sigma_val, n_val, ax, bound_min, bound_max, scale):
    sigma = sigma_val

    def loglike(
        vt: pt.TensorVariable, #observed transverse velocities
        wc: pt.TensorVariable,
        sigma=sigma,
        n_val=n_val,
    ) -> pt.TensorVariable:
        # wt = pt.vector("wt", dtype="float64")
        # wc = pt.scalar("wc", dtype="float64")

        sum = 0

        # configurables

        delta_v = sigma / 3

        for n in range(-n_val, n_val + 1):

            # check if inputs are corret
            # Compute squared terms
            # TODO fix naming, it's very jack hammered atm
            v_samples = vt + n * delta_v

            def function(v_arg): # Define CDF in terms of v

                # CDF VERSION

                wt = pt.pow(v_arg, -1)
                wt_squared = pt.pow(v_arg, -2)
                wc_squared = pt.pow(wc, 2)
                sum_squares = wc_squared + wt_squared

                # First expression (used when wc < 1)
                square_root = pt.sqrt(sum_squares - 1)
                at = pt.arctan(square_root)
                numer1 = wt * (square_root - at)
                expr1 = 1 - numer1 / sum_squares
                
                # Second expression (used when wc > 1)
                at2 = pt.arctan(wt / wc)
                numer2 = wt**2 - wt * at2
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

                negvcondition = pt.lt(v_arg, 0)
                result = pt.switch(negvcondition, 0, result)
                # result is now:    
                    # 0 if vt < 0
                    # 1 if wc < 1 and wt^2 + wc^2 < 1
                    # expr1 if wc < 1 and wt^2 + wc^2 > 1
                    # expr2 if wc > 1

                return result

            # coefficient = (-(n * sigma) / (pt.sqrt(2 * pt.pi) * sigma**3)) * pt.exp(
            #     -((n * sigma) ** 2) / (2 * sigma**2)
            # )
            coefficient = (v_samples - vt) / (pt.sqrt(2 * pt.pi) * sigma**3) * pt.exp(
                (-((vt - v_samples) ** 2)) / (2 * sigma**2)
            )

            # print(function(w+delta_w/2))
            # print(function(w+delta_w/2))
            # if pt.isnan(function(w + delta_w/2)):
            #     print("w + ∆w/2 is ", w+delta_w/2)
            # if pt.isnan(function(w - delta_w/2)):
            #     print("w - ∆w/2 is ", w-delta_w/2)
            sum += coefficient * function(v_samples) * delta_v
        
        return pt.log(sum)

        # return sum

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
    vt_data = [sublist[3] for sublist in dataAll]

    vt_sym = pt.dscalar("vt")
    wc_sym = pt.dscalar("wc")

    f_loglike = pytensor.function(
        [vt_sym, wc_sym],
        loglike(vt_sym, wc_sym),
        # mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=False),
    )

    # wt_array = np.linspace(0, 3, 75)
    vt_array = vt_data
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
        scale*np.exp(Z_combine - Z_max),
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

# def main():
#     sigma_val = 0.01
#     n_val = 12
    
#     fig, ax = plt.subplots() 
#     bound_min = -1
#     bound_max = 1
#     scale = 1
#     make_plot_like(sigma_val, n_val, ax, bound_min, bound_max, scale)
    
#     plt.show()
    

# if __name__ == "__main__":
#     main()
