#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for meter estimation notebook
"""
import os

import numpy as np

NOTEBOOK_DIR = os.path.dirname(os.path.abspath(__file__))


def compute_autocorrelation(x: np.ndarray) -> np.ndarray:
    """
    Compute non-normalized autocorrelation (consider only positive lags)

    Parameters
    x : np.ndarray
        Input array

    Returns
    -------
    result : np.ndarray
        Autocorrelation
    """
    result = np.correlate(x, x, mode="full")
    # Consider only positive lags
    result = result[result.shape[0] // 2 :]

    return result
