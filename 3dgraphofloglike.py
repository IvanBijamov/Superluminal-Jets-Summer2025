import numpy as np
import matplotlib.pyplot as plt
import pytensor
import pytensor.tensor as pt
from pytensor.compile.nanguardmode import NanGuardMode

from simulationImport import importCSV
import os

sigma = 0.2
n_val = 10


def loglike(
    wt: pt.TensorVariable, wc: pt.TensorVariable, sigma=sigma, n_val=n_val
) -> pt.TensorVariable:
    # wt = pt.vector("wt", dtype="float64")
    # wc = pt.scalar("wc", dtype="float64")

    # PSEUDOTHEORY
    # make w= wt+n*sigma
    # N(w) = (pt.exp(-(w-wt)**2/2*sigma**2)*1/pt.sqrt(2*pi)*sigma
    # can use the below for P(w) just mdofiy for same input of wt+n*sigma
    sum = 0
    # sigma = 0.02
    delta_w = sigma
    wt_regular = wt
    for n in range(-n_val, n_val):

        # check if inputs are corret
        # Compute squared terms
        # TODO fix naming, it's very jack hammered atm
        w = wt_regular + n * sigma

        def function(wt):

            # CDF VERSION

            # wt = w + delta_w / 2

            wt_squared = pt.pow(wt, 2)
            wc_squared = pt.pow(wc, 2)
            sum_squares = wc_squared + wt_squared

            # First expression (used when wc < 1)
            square_root = pt.sqrt(sum_squares - 1)
            at = pt.arctan(square_root)

            numer1 = wt * (square_root - at)
            denom = sum_squares
            expr1 = numer1 / denom

            # Second expression (used when wc >= 1)
            # wt_times_wc = wt * wc
            # at_ratio = pt.arctan(wt / wc)
            # Single-argument arctan is OK here because wt & wc are always positive
            # this one isnt right
            # piece2_1 = (wc - 1) / ((wc**3) * (wt**2))
            #
            # piece2_2 = wc / sum_squares
            #
            # piece2_3 = (wt * at_ratio) / sum_squares
            #
            # expr2 = -piece2_1 - piece2_2 - piece2_3
            #
            # at = pt.arctan(w / wc)

            # this one also maybe isnt

            # numer2 = wc**2 + w * at
            # denom2 = w**2 + wc**2
            # expr2 = 1 - numer2 / denom2

            # newst test

            at2 = pt.arctan(wt / wc)
            numer2 = wt**2 - wt * at2
            denom2 = sum_squares
            expr2 = numer2 / denom2

            fastslowcondition = pt.lt(wc, 1)

            result = pt.switch(fastslowcondition, expr1, expr2)

            # expr1 if wc < 1, expr2 if wc > 1

            negwtcondition = pt.lt(wt, 0)
            sumsquarecondition = pt.lt(sum_squares, 1)

            result = pt.switch(negwtcondition | sumsquarecondition, 0, result)

            # expr1 if wc < 1 & wt > 0, expr2 if wc > 1 & wt >0, 0 if wt < 0 or sum_squares < 1

            return result

        coefficient = ((n * sigma) / (pt.sqrt(2 * pt.pi) * sigma**3)) * pt.exp(
            -((n * sigma) ** 2) / (2 * sigma**2)
        )

        # print(function(w+delta_w/2))
        # print(function(w+delta_w/2))
        # if pt.isnan(function(w + delta_w/2)):
        #     print("w + ∆w/2 is ", w+delta_w/2)
        # if pt.isnan(function(w - delta_w/2)):
        #     print("w - ∆w/2 is ", w-delta_w/2)
        sum += coefficient * function(w) * delta_w

    return pt.log(sum)
    # return sum
    # return sum


dir_path = os.path.dirname(os.path.realpath(__file__))

# find the path for the data source;  this should work on everyone's system now
# dataset = "/isotropic_sims/10/data_3959143911168_xx_0.8_yy_0.8_zz_0.8.csv"
dataset = "/isotropic_sims/10000/data_3957522615761_xx_0.8_yy_0.8_zz_0.8.csv"
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
wc_array = np.linspace(0, 3, 100)  # 100 values for wc
# wc_array = [1.2]

WT, WC = np.meshgrid(wt_array, wc_array)


Z = np.empty(WT.shape)


for i in range(WT.shape[0]):
    for j in range(WT.shape[1]):
        wt_val = WT[i, j]
        wc_val = WC[i, j]
        Z[i, j] = f_loglike(wt_val, wc_val)

Z_combine = np.sum(Z, axis=1)


# Z_collapsed = np.trapz(Z, x=wt_array, axis=1)


plt.figure(figsize=(8, 6))
plt.plot(wc_array, Z_combine, marker="o", linestyle="-")
plt.xlabel("wc")
plt.ylabel("log-like")
plt.title("log-like vs. wc - sigma = " + str(sigma) + ", n_val = " + str(n_val))
plt.grid(True)
plt.show()
