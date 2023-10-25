#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluates a submission on the challenge server,
computing the average tonal distance
"""
import argparse
import re
import numpy as np

key_pat = re.compile("([A-G])([xb\#]*)(m*)")

MAJOR_KEYS = [
    "F",
    "C",
    "G",
    "D",
    "A",
    "E",
    "B",
    "F#",
    "C#",
    "G#",
    "D#",
    "A#",
    "E#",
]

MINOR_KEYS_u = [k + "m" for k in MAJOR_KEYS]
MINOR_KEYS = MINOR_KEYS_u[3:] + MINOR_KEYS_u[:3]

KEY_COORDINATES = dict(
    [
        (key, np.array([np.cos(np.pi * i / 6), np.sin(np.pi * i / 6)]))
        for i, key in enumerate(MAJOR_KEYS)
    ]
)
KEY_COORDINATES.update(
    dict(
        [
            (key, np.array([np.cos(np.pi * i / 6), np.sin(np.pi * i / 6)]))
            for i, key in enumerate(MINOR_KEYS)
        ]
    )
)


def enharmonic_spelling(key: str) -> str:
    """
    Use enharmonic spelling to rename
    the labels to the list of expected keys.
    as specified in `MAJOR_KEYS` and `MINOR_KEYS`

    Parameter
    ---------
    key : str
        A string representing the key.

    Returns
    -------
    key : str
        The enharmonic spelling of the key appearing
        in the labels.
    """
    if key == "None":
        print(f"Invalid key: {key}")
        return "C"
    steps = ["C", "D", "E", "F", "G", "A", "B"]

    step, alter, mode = key_pat.search(key).groups()

    if step + alter == "B#":
        return "C"
    elif step + alter == "E#":
        return "F"
    elif step + alter == "Cb":
        return "B"
    elif step + alter == "Fb":
        return "E"

    if alter == "b":
        kstep = steps.index(step) - 1
        return steps[kstep] + "#" + mode
    else:
        return key


def compare_keys(prediction_key: str, ground_truth_key: str) -> float:
    """
    Tonal Distance between two keys.

    This method computes the tonal distance (in terms of closeness in
    the circle of fifths).

    Parameters
    ----------
    prediction_key: str
        Predicted key.

    ground_truth_key: str
        Ground truth key.

    Returns
    -------
    score: float
        Tonal distance.
    """
    pred_key = enharmonic_spelling(prediction_key)
    gt_key = enharmonic_spelling(ground_truth_key)

    mp = pred_key in MINOR_KEYS
    mg = gt_key in MINOR_KEYS

    vp = KEY_COORDINATES[pred_key]
    vy = KEY_COORDINATES[gt_key]

    cos_angle = np.dot(vp, vy)
    angle = np.arccos(np.clip(cos_angle, -1, 1))

    score = 6 * angle / np.pi + 1 if (mp != mg) else 6 * angle / np.pi

    return score


def load_submission(fn: str) -> dict:
    """
    Load a submission
    """

    gt = np.loadtxt(
        fn,
        dtype=str,
        delimiter=",",
        comments="//",
    )

    predictions = dict([(g[0], g[1]) for g in gt])

    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=str)
    parser.add_argument("--target", type=str, default=None)
    args = parser.parse_args()

    tonal_distance = []

    submission = load_submission(args.submission)

    target = load_submission(args.target)

    for piece, key in target.items():
        if piece in submission:
            td = compare_keys(submission[piece], key)
        else:
            # If the piece is not found, assume that
            # the maximal tonal distance.
            td = 7

        tonal_distance.append(td)
    mean_score = np.mean(tonal_distance)
    print(mean_score)
