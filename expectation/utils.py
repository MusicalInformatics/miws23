from abc import ABC, abstractmethod
import logging
import os

from typing import List, Tuple, Union, Optional

import numpy as np
import glob
import partitura as pt
import warnings

try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False

HOME_DIR = "."

if IN_COLAB:
    HOME_DIR = "/content/miws23/expectation"

warnings.filterwarnings("ignore")

LOGGER = logging.getLogger(__name__)


def load_data(min_seq_length: int = 10) -> List[np.ndarray]:
    # load data
    files = glob.glob(os.path.join(HOME_DIR, "data", "*.mid"))
    files.sort()
    sequences = []
    for fn in files:
        seq = pt.load_performance_midi(fn)[0]
        if len(seq.notes) > min_seq_length:
            sequences.append(seq.note_array())
    return sequences

def find_nearest(array: np.ndarray, value: float) -> np.ndarray:
    """
    From https://stackoverflow.com/a/26026189
    """
    idx = np.clip(np.searchsorted(array, value, side="left"), 0, len(array) - 1)
    idx = idx - (np.abs(value - array[idx - 1]) < np.abs(value - array[idx]))
    return idx
