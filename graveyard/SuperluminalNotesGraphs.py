#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 09:11:33 2025

@author: ivanbijamov
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rv_continuous
import pandas as pd
# %%
#The Graph of the Lorentz-invariant case
class lorentzInvar(rv_continuous):
    "Lorentz Invariant Probability"
    def _pdf(self, x):
        return ((x+(x**2-1)*np.arctan(x))/((x**2+1)**2))
lorentzInvariant = lorentzInvar(name='lorentzInvariant')
    
    
x = np.linspace(0, 3, 100)


pdf_values = lorentzInvariant.pdf(x)

# Plotting the probability distribution
plt.plot(x, pdf_values, label="Prob. dist for $w_t$($v_c$=1)", color="blue")
plt.xlabel("$w_t$")
plt.ylabel("P($w_t$)")
plt.title("Lorentz Invariant Probability Distribution")
plt.legend()
plt.grid(True)
plt.show()
# %%
#z is w_c and x is w_t
def functionFastLight(x,z):
    return ((z**2)*np.sqrt(x**2+z**2-1)+(x**2-z**2)*\
            np.arctan(np.sqrt(x**2+z**2-1)))/((x**2+z**2)**2)
class fastLight(rv_continuous):
    "Fast Light Case"
    def _pdf(self,x):
        return functionFastLight(x,z)
fastLight = fastLight(name='fastLight')
x = np.linspace(0, 3, 100)
z_values = [1, 0.8, 0.6]
colors = ["blue", "orange", "green"]
labels = ["$w_c$ = 1", "$w_c$ = 0.8", "$w_c$ = 0.6"]

for z, color, label in zip(z_values, colors, labels):
    pdf_values = functionFastLight(x, z)
    pdf_values = np.nan_to_num(pdf_values)
    plt.plot(x, pdf_values, label=label, color=color)
#Replace zeroes with a veeeeerrrry small number that can be logged
plt.xlabel("$w_t$")
plt.ylabel("P($w_t$)")
plt.title("Fast Light Case")
plt.legend()
plt.grid(True)
plt.show()
    
    
    
# %%

def functionSlowLight(x,z):
    return ((2*z*x*(z-1))/(x**2+z**2)**2)\
        +((x*z+(x**2-z**2)*np.arctan(x/z))/(x**2+z**2)**2)
class slowLight(rv_continuous):
    "slow Light Isotropic case"
    def _pdf(self,x):
        return functionSlowLight(x,z)
slowLight = slowLight(name='slowLight')
x = np.linspace(0,3,2000)
z_values = [1,1.2,1.5]
colors = ["blue","orange","green"]
labels = ["$w_c$ = 1","$w_c$ = 1.2","$w_c$ = 1.5"]

for z,color,label in zip(z_values,colors,labels):
    pdf_values = functionSlowLight(x,z)
    plt.plot(x,pdf_values,label=label,color=color)
plt.xlabel("$w_t$")
plt.ylabel("P($w_t$)")
plt.title("Slow Light Case")
plt.legend()
plt.grid(True)
plt.show()
# %%