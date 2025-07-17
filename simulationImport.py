#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 09:38:26 2025

@author: ivanbijamov
"""
import numpy as np
import pandas as pd


def importCSV(filepath, filetype="Mojave"):
    # Must modify file path to match where your version of the file is stored
    # Ensure that you correct the headers of the file you use, as by default only
    # 6 out of the 9 have headers, which causes it to break
    dataImport = pd.read_csv(filepath, header=None, skiprows=1)
    radii = []
    # from x

    thetas = []
    # from z
    phis = []
    # TODO: find numpy function to make code more efficient with finding r
    for i in range(len(dataImport)):
        # pos = dataImport.iloc[i, 0:3]
        # r = np.linalg.norm(pos)
        # phi = np.arctan2(pos[1], pos[0])
        # # if arccos, it is theta angle, if arcsin, then declination
        # theta = np.arcsin(pos[2] / r)

        # THESE ARE PLACEHOLDERS, UNCOMMENT ABOVE TO GET ACTUAL POSITION STUFF WORKING

        r = 1.0
        phi = 1.0
        theta = 1.0
        phis.append(phi)
        thetas.append(theta)
        radii.append(r)

    newList = []
    i = 0
    # returns radius, theta/declination, phi, and inverse apparent velocity
    if filetype=="Mojave" :
        vindex = 13
        dvindex = 14
    elif filetype=="Schindler":
        vindex = 3
        dvindex = 4
        
    for radius, theta, phi in zip(radii, thetas, phis):
        # index 3 & 4 for warren data gen aniso
        # 8 for the mathematica iso version (take inverse of mathematica)
        # 13 and 14 for error and whatnot of mojave
        newList.append(
            [radius, theta, phi, dataImport.iloc[i, vindex], dataImport.iloc[i, dvindex]]
        )
        # print([radius, theta, phi, dataImport.iloc[i, 13], dataImport.iloc[i, 14]])
        i += 1

    return newList


# testrun
# test = importCSV('isotropic_sims/data_3957506368889_xx_1.2_yy_1.2_zz_1.2.csv')
# newList = importCSV('isotropic_sims/data_3957506368889_xx_1.2_yy_1.2_zz_1.2.csv')
# TEST
# maxVal = 0
# for i in newList:
#     if maxVal<i[3]:
#         maxVal = i[3]
# print(maxVal)
# theta is from -pi/2 to pi/2 and declination is -pi to pi NOPE
# declination is from -pi/2 to pi/2 and phi is -pi to pi NOPE


### TODO: ANGLES NEED VERIFICATION
