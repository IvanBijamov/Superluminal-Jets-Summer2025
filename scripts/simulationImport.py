#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 09:38:26 2025

@author: ivanbijamov
"""
import numpy as np
import pandas as pd


def importCSV(filepath, filetype):
    # Must modify file path to match where your version of the file is stored
    # Ensure that you correct the headers of the file you use, as by default only
    # 6 out of the 9 have headers, which causes it to break
    dataImport = pd.read_csv(filepath, header=None, skiprows=1)

    if filetype == "Mojave":
        vindex, dvindex = 13, 14
    elif filetype == "Simulated":
        vindex, dvindex = 3, 4

    newList = []
    for i in range(len(dataImport)):
        # Compute spherical coords from columns 0:3 (only numeric for simulated data)
        if filetype == "Simulated":
            pos = dataImport.iloc[i, 0:3].astype(float)
            r = np.linalg.norm(pos)
            phi = np.arctan2(pos[1], pos[0])
            theta = np.arcsin(pos[2] / r)
        else:
            # MOJAVE columns 0:2 are non-numeric (source name, ID, epochs)
            r, theta, phi = np.nan, np.nan, np.nan

        newList.append(
            [
                r,
                theta,
                phi,
                dataImport.iloc[i, vindex],
                dataImport.iloc[i, dvindex],
            ]
        )

    return newList


### TODO: ANGLES NEED VERIFICATION
