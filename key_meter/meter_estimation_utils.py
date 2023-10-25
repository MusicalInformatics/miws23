#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for meter estimation notebook
"""
import os

import numpy as np
import partitura as pt

NOTEBOOK_DIR = os.path.dirname(os.path.abspath(__file__))


def compute_autocorrelation(x: np.ndarray) -> np.ndarray:
    """
    Compute non-normalized autocorrelation (consider only positive lags)

    Parameters
    ----------
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


def get_frames_quantized(
    note_array: np.ndarray,
    framerate: int = 50,
    aggregation="num_notes",
    threshold: float = 0.0,
) -> np.ndarray:
    if aggregation not in ("num_notes", "sum_vel", "max_vel"):
        raise ValueError(
            "`aggregation` must be 'num_notes', 'sum_vel', 'max_vel' "
            f"but is {aggregation}"
        )

    if aggregation == "num_notes":
        # Count number of simultaneous notes
        binary = True
        agg_fun = np.sum
    elif aggregation == "sum_vel":
        binary = False
        agg_fun = np.sum
    elif aggregation == "max_vel":
        binary = False
        agg_fun = np.max

    onset_pr = pt.utils.music.compute_pianoroll(
        note_info=note_array,
        time_unit="sec",
        time_div=framerate,
        onset_only=True,
        binary=binary,
    ).toarray()

    frames = agg_fun(onset_pr, axis=0)

    if threshold > 0:
        frames[frames < threshold] = 0

    return frames


def get_frames_chordify(
    note_array: np.ndarray,
    framerate: int = 50,
    chord_spread_time: float = 1 / 12,
    aggregation="num_notes",
    threshold: float = 0.0,
) -> np.ndarray:

    if aggregation not in ("num_notes", "sum_vel", "max_vel"):
        raise ValueError(
            "`aggregation` must be 'num_notes', 'sum_vel', 'max_vel' "
            f"but is {aggregation}"
        )

    if aggregation == "num_notes":
        # Count number of simultaneous notes
        binary = True
        agg_fun = np.sum
    elif aggregation == "sum_vel":
        binary = False
        agg_fun = np.sum
    elif aggregation == "max_vel":
        agg_fun = np.max


    onsets = note_array["onset_sec"]
    sort_idx = np.argsort(onsets)

    onsets = onsets[sort_idx]
    velocity = note_array["velocity"][sort_idx]

    # (onset, agg_val)
    aggregated_notes = [(0, 0)]

    for (note_on, note_vel) in zip(onsets, velocity):
        prev_note_on = aggregated_notes[-1][0]
        prev_note_vel = aggregated_notes[-1][1]
        if abs(note_on - prev_note_on) < chord_spread_time:

            if aggregation == "num_notes":
                agg_val = 1
            elif aggregation == "sum_vel":
                agg_val = prev_note_vel + note_vel
            elif aggregation == "max_vel":
                agg_val = note_vel if note_vel > prev_note_vel else prev_note_vel
            
            aggregated_notes[-1] = (note_on, agg_val)
        else:

            if aggregation == "num_notes":
                agg_val = 1
            elif aggregation in ("sum_vel", "max_vel"):
                agg_val = note_vel

            aggregated_notes.append((note_on, agg_val))

    frames = np.zeros(int(onsets.max() * framerate) + 1)
    for note in aggregated_notes:
        frames[int((note[0]) * framerate)] += note[1]

    if threshold > 0:
        frames[frames  < threshold]

    return frames

