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
    
    # Takes string of form 0h5m57.175s+38d20'15.149" and returns decimal values
    # of RA & Dec as list.
    
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
    
    # Import CSV values into dataframes.
    radecfilepath = "../mojave_radec.csv" # in parent directory
    datafilepath = "../mojave_cleaned.csv" # in parent directory
    radecdata = pd.read_csv(radecfilepath, header=0)
    cleaneddata = pd.read_csv(datafilepath, header=0)
    
    # print(radecdata.columns)
    # print(cleaneddata.dtypes)
    
    # Add RA & Dec values to RAdec dataframe. 
    radecstrings = radecdata['RA_Dec_J2000']
    radecvalues = np.array([radecstringparse(x) for x in radecstrings])
    # print(radecvalues[:,0])
    radecdata.insert(3, "RA J2000", radecvalues[:,0])
    radecdata.insert(4, "Dec J2000", radecvalues[:,1])
    # Drop original string column
    radecdata = radecdata.drop(columns=["RA_Dec_J2000"])
    # Rename column for compatibility & clarity in final dataset
    radecdata = radecdata.rename(
        columns={"B1950_Name" : "B1950 Name",
                 "Source_URL" : "Main data page URL"})
    
    # Merge radecdata & cleaneddata
    newdata = pd.merge(cleaneddata,radecdata,how='left',on='B1950 Name')

    # Export to CSV file  
    newdata.to_csv("../mojave_cleaned_radec.csv",index=False)
    

if __name__ == "__main__":
    main()
