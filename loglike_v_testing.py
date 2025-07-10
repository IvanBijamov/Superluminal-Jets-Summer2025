#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 17:10:01 2025

@author: mseifer1
"""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import os

def loglike(
    wt: pt.TensorVariable,
    wc: pt.TensorVariable,
    sigma=0.01,
    n_val=12,
) -> pt.TensorVariable:
    # wt = pt.vector("wt", dtype="float64")
    # wc = pt.scalar("wc", dtype="float64")

    vt = 1/wt

    sum = 0

    # configurables

    delta_v = sigma / 3

    for n in range(-n_val, n_val + 1):

        # check if inputs are corret
        # Compute squared terms
        # TODO fix naming, it's very jack hammered atm
        v = vt + n * delta_v

        def function(wt):

            # CDF VERSION

            # wt = w + delta_w / 2

            wt_squared = pt.pow(wt, 2)
            wc_squared = pt.pow(wc, 2)
            sum_squares = wc_squared + wt_squared

            # First expression (used when wc < 1)
            square_root = pt.sqrt(sum_squares - 1)
            at = pt.arctan(square_root)

            numer1 = wt * (square_root - at)
            denom = sum_squares
            expr1 = numer1 / denom

            at2 = pt.arctan(wt / wc)
            numer2 = wt**2 - wt * at2
            denom2 = sum_squares
            expr2 = numer2 / denom2

            fastslowcondition = pt.lt(wc, 1)

            result = pt.switch(fastslowcondition, expr1, expr2)

            # expr1 if wc < 1, expr2 if wc > 1

            negwtcondition = pt.lt(wt, 0)
            sumsquarecondition = pt.lt(sum_squares, 1)

            result = pt.switch(negwtcondition | sumsquarecondition, 0, result)

            # expr1 if wc < 1 & wt > 0, expr2 if wc > 1 & wt >0, 0 if wt < 0 or sum_squares < 1

            return result

        # coefficient = (-(n * sigma) / (pt.sqrt(2 * pt.pi) * sigma**3)) * pt.exp(
        #     -((n * sigma) ** 2) / (2 * sigma**2)
        # )
        coefficient = ((vt - v) / (pt.sqrt(2 * pt.pi) * sigma**3 * wt**2)) * pt.exp(
            (-((vt - v) ** 2)) / (2 * sigma**2)
        )

        # print(function(w+delta_w/2))
        # print(function(w+delta_w/2))
        # if pt.isnan(function(w + delta_w/2)):
        #     print("w + ∆w/2 is ", w+delta_w/2)
        # if pt.isnan(function(w - delta_w/2)):
        #     print("w - ∆w/2 is ", w-delta_w/2)
        sum += coefficient * function(wt) * delta_v
    # sum = pt.where(sum < 1, 0, sum)

    return pt.log(sum)
