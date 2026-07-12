#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 09:38:26 2025

@author: ivanbijamov
"""
import numpy as np
import pandas as pd


def importCSV(filepath, filetype):
    """
    Import velocity data and directional unit vectors from CSV.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    filetype : str
        ``"Simulated"`` for generated data or ``"Mojave"`` for the MOJAVE
        survey (expects mojave_cleaned_radec.csv, which carries RA/Dec in
        decimal degrees as its last two columns).

    Returns
    -------
    list of list
        Each row is ``[r, theta, phi, vt, sigma, n_x, n_y, n_z]`` where:
        - indices 0-2: spherical coordinates (r, theta/declination, phi)
        - index 3: observed transverse velocity
        - index 4: velocity uncertainty (sigma)
        - indices 5-7: Cartesian unit vector n_hat (n_x, n_y, n_z)
    """
    dataImport = pd.read_csv(filepath, header=None, skiprows=1)

    if filetype == "Mojave":
        vindex, dvindex = 13, 14
    elif filetype == "Simulated":
        vindex, dvindex = 3, 4

    newList = []
    for i in range(len(dataImport)):
        # Compute spherical coords and n_hat from columns 0:3
        # (only numeric for simulated data)
        if filetype == "Simulated":
            pos = dataImport.iloc[i, 0:3].astype(float)
            n_x, n_y, n_z = pos[0], pos[1], pos[2]
            r = np.linalg.norm(pos)
            phi = np.arctan2(n_y, n_x)
            theta = np.arcsin(n_z / r)
        else:
            # RA/Dec (J2000, decimal degrees) are the last two columns of
            # mojave_cleaned_radec.csv
            ra = np.deg2rad(float(dataImport.iloc[i, -2]))
            dec = np.deg2rad(float(dataImport.iloc[i, -1]))
            n_x = np.cos(dec) * np.cos(ra)
            n_y = np.cos(dec) * np.sin(ra)
            n_z = np.sin(dec)
            r, theta, phi = 1.0, dec, ra

        newList.append(
            [
                r,
                theta,
                phi,
                dataImport.iloc[i, vindex],
                dataImport.iloc[i, dvindex],
                n_x,
                n_y,
                n_z,
            ]
        )

    return newList
