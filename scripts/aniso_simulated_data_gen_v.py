#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 10:36:35 2025

author: mseifer1
"""

import numpy as np
import csv
import os


def regenerate_data(
    B0=0.3,
    B_vec=(0.2, 0.2, 0.6),
    n_sources=1000,
    decfilter=True,
    seed=None,
):
    δ = -1
    Bº = B0
    B_vec = np.asarray(B_vec, dtype=float)
    N_SOURCES = n_sources  # Number of data points to generate
    # decfilter: set to True to limit the number of sources with declination < 0° S,
    # to mimic MOJAVE data.
    if seed is not None:
        np.random.seed(seed)
    print("N_SOURCES = ", N_SOURCES)

    # Always write to project root, regardless of working directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    OUTPUT_FILE = os.path.join(project_root, "generated_sources.csv")

    # fns
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

    def w_true(v, v_hat, n_hat, v_c_val):

        dot = np.dot(n_hat, v_hat)  # n̂ ⋅ v̂

        numer = 1 + (v / v_c_val) * dot
        denom = v * np.sqrt(1 - dot**2)
        if denom == 0:
            return np.inf  # avoid error- divide by 0
        return numer / denom

    # generate data
    rows = []

    if decfilter:
        n_gen = (
            2 * N_SOURCES
        )  # Generate extra points to help get N_SOURCES datapoints in the end
    else:
        n_gen = N_SOURCES

    for _ in range(n_gen):

        # 1generate v_hat
        v_rand = np.random.normal(size=3)
        norm_vec = np.linalg.norm(v_rand)
        v_hat = v_rand / norm_vec

        # 3a generate raw velocity
        v_raw = np.random.uniform(0.0, 1.0)

        ##3b cerenkov braking if v > 1 / wc
        wc_values = solve_wc(δ, Bº, B_vec, v_hat)

        if wc_values:
            wc_val = min(wc_values)
            v_braked = min(v_raw, 1 / wc_val)
        else:
            wc_val = None
            v_braked = v_raw  # braking if wc is undefined wont occur

        # 4
        r_rand = np.random.normal(size=3)

        # 5 normalize r_rand to get n_hat
        r_norm = np.linalg.norm(r_rand)
        n_hat = r_rand / r_norm

        # 6 calculate wc(n_hat) with solve_wc -> (n_hat, parameteres)
        wc_nhat = solve_wc(δ, Bº, B_vec, n_hat)
        v_c_val = 1 / min(wc_nhat) if wc_nhat else 1.0

        # 7
        w_true_value = w_true(v_braked, v_hat, n_hat, v_c_val)

        if w_true_value < 0:
            print("Warning: w_true value = ", w_true_value)
            print(
                "Parameter values: v_raw = ",
                v_raw,
                "\n v_braked = ",
                v_braked,
                "\n wc_nhat = ",
                wc_nhat,
                "\n v_c_val = ",
                v_c_val,
                "\n v_hat = ",
                v_hat,
                "\n n_hat = ",
                n_hat,
                "\n Angle between v_hat & n_hat = ",
                np.arccos(np.dot(n_hat, v_hat)),
            )
        v_true_value = 1 / w_true_value
        # sigma_default = np.random.uniform(
        #     low=0.01, high=0.5, size=len(N_SOURCES)
        # ).tolist()
        v_sigma = abs(np.random.normal(loc=0, scale=0.1))
        # v_sigma = 0.2
        v_obs = np.random.normal(loc=v_true_value, scale=v_sigma)

        # ensure noise doesnt make v_obs negative
        min_velocity = 1e-12
        abs_v = abs(v_obs)
        v_obs = max(abs_v, min_velocity)

        w_obs = 1 / v_obs

        # data storage
        row = list(n_hat) + [v_obs, v_sigma, v_true_value]
        # nx, ny, nz, v_obs, v_sigma, v_true_value
        rows.append(row)

    print("Initial number of data points generated:", len(rows))
    # print(np.array(rows)[:,2])

    if decfilter:  # Apply declination filter if desired
        rows = [x for x in rows if np.random.rand() < 2 * x[2] + 1]
        # If in Southern Hemisphere, accept with prob. going from 0 at -30° to 1 at 0°
        # Note that x[2] = sin(dec)
        if len(rows) < N_SOURCES:
            print("WARNING: fewer than", N_SOURCES, "data points remain after filter.")
        else:
            rows = rows[:N_SOURCES]

    print("Final size of data set:", len(rows))

    # csv file
    with open(OUTPUT_FILE, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # header line: δ, Bº, B_vec
        # writer.writerow(["delta", "B0", "B_x", "B_y", "B_z"])
        writer.writerow([δ, Bº] + list(B_vec))

        # data header
        # writer.writerow(["n_x", "n_y", "n_z", "w_obs", "sigma", "w_true"])

        # rows
        writer.writerows(rows)


if __name__ == "__main__":
    regenerate_data()
