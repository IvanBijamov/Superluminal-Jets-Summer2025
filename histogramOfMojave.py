from simulationImport import importCSV
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import rv_continuous
import pandas as pd

dir_path = os.path.dirname(os.path.realpath(__file__))

# find the path for the data source;  this should work on everyone's system now
# dataset = "/isotropic_sims/10000/data_3957522615761_xx_0.8_yy_0.8_zz_0.8.csv"
# dataset = "/isotropic_sims/10000/data_3957522615600_xx_1.2_yy_1.2_zz_1.2.csv"
dataset = "/mojave_cleaned.csv"
# dataset = "/generated_sources.csv"

dataSource = dir_path + dataset

dataAll = importCSV(dataSource)
N_bins = 2000
vt_data = [sublist[3] for sublist in dataAll]
plt.hist(vt_data, bins=N_bins, density=True)
# plt.xlim(0, 3)
plt.title("Histogram of Mojave Data")
plt.xlabel("Value")
plt.yscale("log")
plt.ylabel("Frequency")


class lorentzInvar(rv_continuous):
    "Lorentz Invariant Probability"

    def _pdf(self, x):
        x = 1 / x
        return ((x + (x**2 - 1) * np.arctan(x)) / ((x**2 + 1) ** 2)) / (1 / x**2)


lorentzInvariant = lorentzInvar(name="lorentzInvariant")


x = np.linspace(0.000001, 35, 100 * 12)


pdf_values = lorentzInvariant.pdf(x)

# Plotting the probability distribution
plt.plot(
    x,
    pdf_values * len(vt_data) * 1 / N_bins,
    label="Prob. dist for $w_t$($v_c$=1)",
    color="red",
)
plt.xlabel("$v_t$")
plt.ylabel("P($v_t$)")
plt.title("Lorentz Invariant Probability Distribution")
plt.yscale("log")
plt.legend()
plt.grid(True)

plt.show()
