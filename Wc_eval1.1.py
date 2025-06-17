
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 14:43:08 2025

@author: mseifer1
"""

import numpy as np

def solve_wc(δ, Bº, B_vec, n_hat):
    
    P = np.dot(B_vec, n_hat) 

    a = δ * Bº**2 - 1
    b = 2 * δ * Bº * P
    c = δ * P**2 + 1

    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return None
    

    sqrt_disc = np.sqrt(discriminant)
    wc1 = (-b + sqrt_disc) / (2 * a)
    wc2 = (-b - sqrt_disc) / (2 * a)

    positive_wc = [wc for wc in (wc1, wc2) if wc > 0]
    return positive_wc if positive_wc else None

def normalized_rand_nhat(n_samples=10000):
    rand_vectors = np.random.normal(size=(n_samples, 3)) 
    #creates random vectors in 3D space
    norms = np.linalg.norm(rand_vectors, axis=1, keepdims=True)
    #computes norm for each vector
    #(keepdims=True) -> axes which are normed over are left in the result as dimensions with size one
    #aka shape is preserved
    return rand_vectors / norms #normalization

def test_wc_min(δ, Bº, B_vec, n_samples=10000):
    n_hats = normalized_rand_nhat(n_samples)
    min_wc = None  #initialization
    max_wc = None  #initialization
    failures = 0   #initialization

    for nhat in n_hats:       #loop over all nhats
        roots = solve_wc(δ, Bº, B_vec, nhat)  #solve wc with each rand nhat
        if roots:
            min_root = min(roots)
            max_root = max(roots)

            if min_wc is None or min_root < min_wc:
                min_wc = min_root
            if max_wc is None or max_root > max_wc:
                max_wc = max_root
        else:
            failures += 1 #add 1 to sum every time direction doesnt have +/real root

    print(f"min w_c found: {min_wc}")
    print(f"max w_c found: {max_wc}")
    print(f"no solution: {failures}")
    if min_wc is not None and min_wc > 1:
        print("all w_c values are greater than 1")
    else:
        print("certain nhat values result in wc being less than one")
        

#test
δ = 0.1
Bº = 0
B_vec = np.array([0.0, 0.0, -1.0]) #B_vec = 0 for isotropic
test_wc_min(δ, Bº, B_vec, n_samples=50000)

