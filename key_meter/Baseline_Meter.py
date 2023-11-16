#!/usr/bin/env python
# -*- coding: utf-8 -*-

# single script challenge submission template
import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from typing import Union, Tuple, Iterable, Optional, Dict

import numpy as np
import partitura as pt

from partitura.utils.misc import PathLike
from partitura.performance import PerformanceLike

from scipy.signal import find_peaks

from hiddenmarkov import HMM, ConstantTransitionModel, ObservationModel

from meter_estimation_utils import (
    get_frames_quantized,
    get_frames_chordify,
    compute_autocorrelation,
)

from meter_estimation_challenge import load_submission, compare_meter_and_tempo

import warnings

warnings.filterwarnings("ignore")


FRAMERATE = 8
CHORD_SPREAD_TIME = 0.05  # for onset aggregation


class MeterObservationModel(ObservationModel):
    def __init__(
        self,
        states: int = 20,
        downbeat_idx: Iterable = [0],
        beat_idx: Iterable = [50],
        subbeat_idx: Iterable = [25],
    ):
        super().__init__()
        self.states = states
        # observation 1 = note onset present, 0 = nothing present
        self.probabilities = np.ones((2, states)) / 100
        self.probabilities[0, :] = 0.99
        for idx in subbeat_idx:
            self.probabilities[:, idx] = [0.5, 0.5]
        for idx in beat_idx:
            self.probabilities[:, idx] = [0.3, 0.7]
        for idx in downbeat_idx:
            self.probabilities[:, idx] = [0.1, 0.9]
        self.db = downbeat_idx
        self.b = beat_idx
        self.sb = subbeat_idx

    def get_beat_states(self, state_sequence: np.ndarray) -> np.ndarray:
        state_encoder = np.zeros_like(state_sequence)
        for i, state in enumerate(state_sequence):
            if state in self.sb:
                state_encoder[i] = 1
            if state in self.b:
                state_encoder[i] = 2
            if state in self.db:
                state_encoder[i] = 3
        return state_encoder

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        if not self.use_log_probabilities:
            return self.probabilities[observation, :]
        else:
            return np.log(self.probabilities[observation, :])


def getTransitionMatrix(
    states: int,
    distribution: Iterable = [0.1, 0.8, 0.1],
) -> np.ndarray:
    """
    Compute transition Matrix
    """
    if states == 1:
        raise ValueError("The number of states should be > 1")
    transition_matrix = (
        np.eye(states, k=0) * distribution[0]
        + np.eye(states, k=1) * distribution[1]
        + np.eye(states, k=2) * distribution[2]
        + np.ones((states, states)) / 1e7
    )

    transition_matrix[-2, 0] = distribution[2]
    transition_matrix[-1, 0] = distribution[2] + distribution[1]

    return transition_matrix


def createHMM(
    tempo: float = 50,
    frame_rate: int = FRAMERATE,  # frames_per_beat
    beats_per_measure: int = 4,
    subbeats_per_beat: int = 2,
) -> Tuple[MeterObservationModel, ConstantTransitionModel]:
    """
    Create observation and transition models for the HMM

    Parameters
    ----------
    tempo : float
        Tempo in beats per minute
    frame_rate: int
        Number of frames per beat. Selecting a large frame_rate can result
        in very slow models!
    beats_per_measure: int
        Number of beats per measure (numerator of the time signature)
    subbeats_per_beat: int
        Number of divisions per beat (generally 2 or 3)

    Returns
    -------
    observation_model: MeterObservationModel
        The observation model of the HMM
    transition_model: ConstantTransitionModel
        The transition model of the HMM
    """
    frames_per_beat = 60 / tempo * frame_rate
    frames_per_measure = frames_per_beat * beats_per_measure
    states = int(frames_per_measure)
    downbeat_idx = [0]
    beat_idx = [int(states / beats_per_measure * k) for k in range(beats_per_measure)]
    subbeat_idx = [
        int(states / (beats_per_measure * subbeats_per_beat) * k)
        for k in range(beats_per_measure * subbeats_per_beat)
    ]

    observation_model = MeterObservationModel(
        states=states,
        downbeat_idx=downbeat_idx,
        beat_idx=beat_idx,
        subbeat_idx=subbeat_idx,
    )

    transition_matrix = getTransitionMatrix(states)
    transition_model = ConstantTransitionModel(transition_matrix)

    return observation_model, transition_model


def estimate_meter(
    filename: PathLike,
    beats_per_measure: Iterable[int] = [2, 3, 4],
    subbeats_per_beat: Iterable[int] = [2, 3],
    tempi: Union[Iterable[int], str] = "auto",
    frame_aggregation: str = "chordify",
    value_aggregation: str = "num_notes",
    framerate: int = FRAMERATE,
    frame_threshold: float = 0.0,
    chord_spread_time: float = 1 / 12,
    max_tempo: float = 250,
    min_tempo: float = 30,
) -> Tuple[int, float]:
    """
    Estimate tempo, meter (currently only time signature numerator)

    Parameters
    ----------
    note_array : structured array

    Returns
    -------
    meter_numerator: int
        The numerator of the time signature
    tempo: float
        The tempo in beats per minute
    """
    # get note array
    performance = pt.load_performance_midi(filename)
    note_array = performance.note_array()

    if frame_aggregation == "chordify":
        frames = get_frames_chordify(
            note_array=note_array,
            framerate=framerate,
            chord_spread_time=chord_spread_time,
            aggregation=value_aggregation,
            threshold=frame_threshold,
        )
    elif frame_aggregation == "quantize":
        frames = get_frames_quantized(
            note_array=note_array,
            framerate=framerate,
            aggregation=value_aggregation,
            threshold=frame_threshold,
        )

    if tempi == "auto":
        autocorr = compute_autocorrelation(frames)
        beat_period, _ = find_peaks(autocorr[1:], prominence=None)
        tempi = 60 * framerate / (beat_period + 1)
        tempi = tempi[np.logical_and(tempi <= max_tempo, tempi >= min_tempo)]

        if len(tempi) == 0:
            tempi = np.linspace(min_tempo, max_tempo, 10)

    likelihoods = []

    for ts_num in beats_per_measure:
        for sbpb in subbeats_per_beat:
            for tempo in tempi:
                observation_model, transition_model = createHMM(
                    tempo=tempo,
                    frame_rate=framerate,
                    beats_per_measure=ts_num,
                    subbeats_per_beat=sbpb,
                )

                hmm = HMM(
                    observation_model=observation_model,
                    transition_model=transition_model,
                )

                frames[frames < 1.0] = 0
                frames[frames >= 1.0] = 1

                observations = np.array(frames, dtype=int)
                _, log_lik = hmm.find_best_sequence(observations)

                likelihoods.append((ts_num, sbpb, tempo, log_lik))

    likelihoods = np.array(likelihoods)

    best_result = likelihoods[likelihoods[:, 3].argmax()]

    best_ts = int(best_result[0])
    best_tempo = best_result[2]

    return best_ts, best_tempo


def process_file(
    mfn: PathLike, ground_truth: Dict[str, Tuple[float, float]],
) -> Tuple[str, int, float, Optional[float], Optional[float]]:
    """
    Compute meter and get evaluation for
    """
    piece = os.path.basename(mfn)
    predicted_meter, predicted_tempo = estimate_meter(filename=mfn)

    meter_accuracy = None
    tempo_error = None
    if piece in ground_truth:
        expected_meter, expected_tempo = ground_truth[piece]
        meter_accuracy, tempo_error = compare_meter_and_tempo(
            predicted_meter,
            expected_meter,
            predicted_tempo,
            expected_tempo,
        )
    return (
        piece,
        predicted_meter,
        predicted_tempo,
        meter_accuracy,
        tempo_error,
    )


if __name__ == "__main__":
    import argparse
    import os
    import glob

    # DO NOT CHANGE THIS!
    parser = argparse.ArgumentParser(description="Meter Estimation")
    parser.add_argument(
        "--datadir",
        "-i",
        help="path to the input files",
        default=None,
        type=str,
    )

    parser.add_argument(
        "--challenge",
        "-c",
        help="Export results for challenge",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--outfile",
        "-o",
        help="Output file",
        type=str,
        default="meter_estimation.txt",
    )

    parser.add_argument(
        "--ground-truth",
        "-t",
        help="File with the ground truth labels",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    # Adapt this part as needed!
    midi_files = glob.glob(os.path.join(args.datadir, "*.mid"))
    midi_files.sort()

    ground_truth = {}
    if args.ground_truth:
        ground_truth = load_submission(args.ground_truth)

    # Parallel processing with concurrent.futures
    with ProcessPoolExecutor() as executor:
        # Using executor.map for parallel processing
        results_ = list(
            tqdm(
                executor.map(
                    process_file,
                    midi_files,
                    len(midi_files) * [ground_truth],
                ),
                total=len(midi_files),
            )
        )

    results = [res[:3] for res in results_]
    evaluation = [res[3:] for res in results_]

    for (piece, predicted_meter, predicted_tempo), (meter_accuracy, tempo_error) in zip(
        results, evaluation
    ):
        expected_meter, expected_tempo = ground_truth[piece]
        if meter_accuracy is not None:
            print(
                f"{piece}: "
                f"\tPredicted:{predicted_meter} {predicted_tempo:.2f}"
                f"\tExpected:{expected_meter} {expected_tempo:.2f}"
                f"\tTempo error:{tempo_error}"
            )

    mean_eval = np.mean(evaluation, 0)
    if len(evaluation) > 0:
        print("\n\nAverage Performance over dataset")
        print(f"\tMeter accuracy{mean_eval[0] * 100: .1f}")
        print(f"\tTempo error: {mean_eval[1]: .2f}")

    if args.challenge:
        # Export predictions for the challenge
        np.savetxt(
            args.outfile,
            np.array(results),
            fmt="%s",
            delimiter=",",
            comments="//",
            header="filename,ts_num,tempo(bpm)",
        )
