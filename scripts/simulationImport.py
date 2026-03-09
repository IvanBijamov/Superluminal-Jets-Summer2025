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
        ``"Simulated"`` for generated data or ``"Mojave"`` for the MOJAVE survey.

    Returns
    -------
    list of list
        Each row is ``[r, theta, phi, vt, sigma, n_x, n_y, n_z]`` where:
        - indices 0-2: spherical coordinates (r, theta/declination, phi)
        - index 3: observed transverse velocity
        - index 4: velocity uncertainty (sigma)
        - indices 5-7: Cartesian unit vector n_hat (n_x, n_y, n_z)

        For MOJAVE data, indices 0-2 and 5-7 are NaN (no sky-direction data
        in the CSV).
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
            # TODO: derive n_hat for MOJAVE data from B1950 source name (encodes RA/Dec).
            # Currently returns NaN — the anisotropic model will NOT work with MOJAVE data
            # until this is implemented.
            r, theta, phi = np.nan, np.nan, np.nan
            n_x, n_y, n_z = np.nan, np.nan, np.nan

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
