#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 12:18:58 2025

@author: warrenschindler
"""


import pandas as pd
import numpy as np
import requests
import html
import os

# Fetch HTML
url = "https://www.cv.nrao.edu/MOJAVE/velocitytableXVIII.html"
response = requests.get(url)
html_text = html.unescape(response.text)  # decodes &plusmn to ±

# read first table from HTML
df = pd.read_html(html_text)[0]

# drop unwanted headers
df = df.iloc[3:-1].reset_index(drop=True)

# parse by splitting
new_cols = []

for col in df.columns:
    if df[col].astype(str).str.contains("±").any():
        val_col = col + "_val"
        err_col = col + "_err"

        def split_pm(cell):
            if pd.isna(cell):
                return [np.nan, np.nan]
            parts = str(cell).split("±")
            if len(parts) == 2:
                return [parts[0].strip(), parts[1].strip()]
            else:
                return [parts[0].strip(), np.nan]

        split_vals = df[col].apply(split_pm)
        df[val_col] = split_vals.apply(lambda x: x[0])
        df[err_col] = split_vals.apply(lambda x: x[1])
        new_cols.extend([val_col, err_col])
    else:
        new_cols.append(col)

# clean and convert
df_clean = df[new_cols]
df_clean = df_clean.apply(pd.to_numeric, errors="ignore")


# save path
dir_path = os.path.dirname(os.path.realpath(__file__))
dataset = "/mojave_cleaned.csv"
csv_path = dir_path + dataset
# csv_path = "/Users/warrenschindler/Desktop/one/mojave_cleaned.csv"
df_clean.to_csv(csv_path, index=False)

print(f"Cleaned table saved as {csv_path}")
