#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluates a submission on the challenge server,
computing the average F-score
"""

import argparse
import numpy as np

from typing import List


def compare_alignments(
    prediction: List[dict],
    ground_truth: List[dict],
    types: List[str] = ["match", "deletion", "insertion"],
) -> (float, float, float):
    """
    Parameters
    ----------
    prediction: List of dicts
        List of dictionaries containing the predicted alignments
    ground_truth:
        List of dictionaries containing the ground truth alignments
    types: List of strings
        List of alignment types to consider for evaluation
        (e.g ['match', 'deletion', 'insertion']

    Returns
    -------
    f_score: float
       The F score
    """

    pred_filtered = list(filter(lambda x: x["label"] in types, prediction))
    gt_filtered = list(filter(lambda x: x["label"] in types, ground_truth))

    filtered_correct = [pred for pred in pred_filtered if pred in gt_filtered]

    n_pred_filtered = len(pred_filtered)
    n_gt_filtered = len(gt_filtered)
    n_correct = len(filtered_correct)

    if n_pred_filtered > 0 or n_gt_filtered > 0:
        precision = n_correct / n_pred_filtered if n_pred_filtered > 0 else 0.0
        recall = n_correct / n_gt_filtered if n_gt_filtered > 0 else 0
        f_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
    else:
        # no prediction and no ground truth for a
        # given type -> correct alignment
        precision, recall, f_score = 1.0, 1.0, 1.0

    return f_score


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=str)
    parser.add_argument("--target", type=str, default=None)
    args = parser.parse_args()

    submission = dict(np.load(args.submission, allow_pickle=True))

    target = dict(np.load(args.target, allow_pickle=True))

    f_scores = []

    for piece_name, gt_alignment in target.items():

        if piece_name in submission:
            pred_alignment = submission[piece_name].tolist()

            f_score = compare_alignments(
                prediction=pred_alignment,
                ground_truth=gt_alignment.tolist(),
            )

            f_scores.append(f_score)

        else:
            f_scores.append(0)

    mean_f_score = np.mean(f_scores)

    print(mean_f_score)

        
