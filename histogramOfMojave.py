from simulationImport import importCSV
import matplotlib.pyplot as plt
import numpy as np
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

# find the path for the data source;  this should work on everyone's system now
# dataset = "/isotropic_sims/10000/data_3957522615761_xx_0.8_yy_0.8_zz_0.8.csv"
# dataset = "/isotropic_sims/10000/data_3957522615600_xx_1.2_yy_1.2_zz_1.2.csv"
dataset = "/mojave_cleaned.csv"
# dataset = "/generated_sources.csv"

dataSource = dir_path + dataset

dataAll = importCSV(dataSource)

vt_data = [sublist[3] for sublist in dataAll]
plt.hist(vt_data, bins=150)
plt.title("Histogram of Mojave Data")
plt.xlabel("Value")
plt.ylabel("Frequency")


plt.show()
