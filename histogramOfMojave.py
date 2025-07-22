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
N_bins = 20
vt_data = [sublist[3] for sublist in dataAll]
plt.hist(vt_data, bins="auto", density=True, label="MOJAVE data")
# plt.xlim(0, 3)
plt.title("Histogram of Mojave Data")
plt.xlabel("Value")
# plt.yscale("log")
plt.ylabel("Frequency")


class lorentzInvar(rv_continuous):
    "Lorentz Invariant Probability"

    # TODO: This throws an error if 0 is fed in, but it returns the cor
    def _pdf(self, x):
        return np.where( x <0, 0, 
                        np.where( x == 0, np.pi/2, (x + (1 - x**2)*np.arctan(1/x))/(1+x**2)))
    
        
        # if x == 0:
        #     return np.pi()/2
        # elif x < 0:
        #     return 0
        # else:
        #     x = 1 / x
        #     return ((x + (x**2 - 1) * np.arctan(x)) / ((x**2 + 1) ** 2)) / (1 / x**2)


lorentzInvariant = lorentzInvar(name="lorentzInvariant")

ymin, ymax = plt.ylim()
vmin, vmax = plt.xlim()
v_values = np.linspace(vmin, vmax, 1000)
# pdf_values = np.where( v_values <= 0, np.pi()/2, (v_values + (1 - v_values**2)*np.arctan(1/v_values))/(1+x**2)**2)


pdf_values = lorentzInvariant.pdf(v_values)

# Plotting the probability distribution
plt.plot(
    v_values,
    pdf_values,
    label="Expected prob. dist ($v_c=1$)",
    color="red",
)
plt.xlabel("$v_t$")
plt.ylabel("P($v_t$)")
plt.title("Lorentz Invariant Probability Distribution")
# plt.yscale("log")
plt.legend()
plt.ylim(top=ymax)
plt.grid(True)

plt.show()
