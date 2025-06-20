#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 11:39:35 2025

@author: mseifer1
"""

import numpy as np

δ = -0.1
Bº = 1.0
B_vec = np.array([1.0, 0.0, 0.0])  


#np.random.seed(42) 
#reproducibility seed if we want same values

#1 generate a 3D Gaussian radnom vector (v_rand)
v_rand = np.random.normal(size=3)

#2 normalize for v_hat
norm_vec = np.linalg.norm(v_rand)
v_hat = v_rand / norm_vec

#3
def solve_wc(δ, Bº, B_vec, n_hat):
    dot_product = np.dot(B_vec, n_hat)

    if np.allclose(B_vec, 0):
        if δ * Bº**2 >= 1:
            return None
        wc = 1 / np.sqrt(1 - δ * Bº**2)
        return [wc] if wc > 0 else None

    a = δ * Bº**2 - 1
    b = 2 * δ * Bº * dot_product
    c = δ * dot_product**2 + 1

    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return None

    sqrt_disc = np.sqrt(discriminant)
    wc1 = (-b + sqrt_disc) / (2 * a)
    wc2 = (-b - sqrt_disc) / (2 * a)

    positive_wc = [wc for wc in (wc1, wc2) if wc > 0]
    return positive_wc if positive_wc else None

#3a generate a uniform velocity v [0, 1]
v_raw = np.random.uniform(0.0, 1.0)  

#3b cerenkov braking if v > 1 / wc
wc_values = solve_wc(δ, Bº, B_vec, v_hat)

if wc_values:
    wc_val = min(wc_values)
    v_braked = min(v_raw, 1 / wc_val)
else:
    wc_val = None
    v_braked = v_raw  #braking if wc is undefined wont occur


#4 generate r_rand 3D gaussian vector
r_rand = np.random.normal(size=3)


#5 normalize r_rand to get n_hat
r_norm = np.linalg.norm(r_rand)
n_hat = r_rand / r_norm


#6 calculate wc(n_hat) with solve_wc -> (n_hat, parameteres)
wc_nhat = solve_wc(δ, Bº, B_vec, n_hat)
v_c_val = min(wc_nhat) if wc_nhat else 1.0


#7 calculate w_true
    
def w_true(v, v_hat, n_hat, v_c_val):
    
    """
    Parameters used:
    v: velocity scalar
    v_hat: unit velocity vector (3D gaussian)
    n_hat: unit direction vector (3D gaussian)
    v_c_val: v_c(n_hat)
    """
    
    dot = np.dot(n_hat, v_hat)  # n̂ ⋅ v̂
   
    numer = (1 + (v / v_c_val) * dot)
    denom = v * np.sqrt(1 - dot**2)
    if denom == 0:  
        return np.inf  #avoid error- divide by 0
    return numer / denom

w_true_value = w_true(v_braked, v_hat, n_hat, v_c_val)
    
#8 generate sigma [0,1]
sigma = np.random.uniform(0.0, 1.0)

#9 generate w_obs -> normal distribution (wobs, sigma)
w_obs = np.random.normal(loc=w_true_value, scale=sigma)

#10 output w_obs, sigma, n_hat, w_true

print(f"n_hat   : {n_hat}")        
print(f"w_true  : {w_true_value}")    
print(f"sigma   : {sigma}")        
print(f"w_obs   : {w_obs}")
