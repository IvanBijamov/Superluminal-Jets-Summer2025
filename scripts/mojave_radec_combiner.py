#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 17:01:45 2026

@author: mseifer1

Script to combine mojave_cleaned.csv (output of Schindler parsing code) and 
mojave_radec.csv (output of Claude code to scrape RA & Dec data from MOJAVE 
website) into a single dataset called mojave_cleaned_radec.csv.  In process,
converts RA & Dec as single string into two floats in order [RA, Dec], both in 
decimal degrees.
"""

import numpy as np
import pandas as pd
import re

def radecstringparse(radecstring):
    
    numberstrings = re.findall(r'\+|-|[\d]+', radecstring) #split string into digits & sign of dec
    signstring = numberstrings[4] # extract sign and then drop from list
    numberstrings.pop(4)
    numbers = [int(x) for x in numberstrings] #convert number strings into integers
    ra = numbers[0]*15 + numbers[1]/4 + numbers[2]/240 + numbers[3]/240000
    dec = numbers[4]+numbers[5]/60 + numbers[6]/3600 + numbers[7]/(3600000)
    if signstring=="-": #flip sign of dec
        dec = -dec
    return [ra,dec]
    

def main():
    radecfilepath = "../mojave_radec.csv" # in parent directory
    datafilepath = "../mojave_cleaned.csv" # in parent directory
    radecdata = pd.read_csv(radecfilepath, header=0)
    cleaneddata = pd.read_csv(datafilepath, header=0)
    
    # print(radecdata.dtypes)
    # print(cleaneddata.dtypes)
    
    radecstrings = radecdata['RA_Dec_J2000']
    radecvalues = [radecstringparse(x) for x in radecstrings]
    print(radecvalues)
    
    
    

if __name__ == "__main__":
    main()
