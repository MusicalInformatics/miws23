import os
from scipy.io import wavfile
from scipy.fftpack import fft
import numpy as np

from typing import Any, Tuple, List

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from partitura.utils.music import frequency_to_midi_pitch, midi_pitch_to_frequency
from partitura.utils.synth import additive_synthesis, SAMPLE_RATE

import IPython.display as ipd
from ipywidgets import interactive, fixed

PATH = os.path.dirname(os.path.realpath(__file__))
MIDI_PITCH_FREQS = midi_pitch_to_frequency(np.arange(128))


class VerticalLineOnClickPlot(object):
    """
    Interactive plot

    Add line: left click
    Remove last line: right click
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        figsize: Tuple[int, int] = (8, 6),
        xlabel: str = "x",
        ylabel: str = "y",
    ) -> None:
        self.x = x
        self.y = y
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.plot(
            x,
            y,
            color="navy"
        )
        self.lines: List[plt.Line2D] = []
        self.x_values: List[float] = []
        self.interp_fun = interp1d(
            x=x,
            y=y,
            kind="linear",
            fill_value="extrapolate",
        )
        self.cid = self.fig.canvas.mpl_connect(
            "button_press_event",
            self.onclick,
        )

        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)

    def onclick(self, event) -> None:
        if event.inaxes == self.ax:
            if event.button == 1:  # Left click
                self.add_line(event.xdata)
            elif event.button == 3:  # Right click
                self.remove_line()
            self.update_plot()

    def add_line(self, x_val: float) -> None:
        line = self.ax.axvline(x_val, color="r", linestyle="--")
        self.lines.append(line)
        self.x_values.append(x_val)
        self.x_values.sort()

    def remove_line(self) -> None:
        if self.lines:
            self.lines[-1].remove()
            del self.lines[-1]
            del self.x_values[-1]

    def update_plot(self) -> None:
        plt.draw()

    @property
    def y_values(self) -> np.ndarray:
        y_interpolated = self.interp_fun(self.x_values)
        return y_interpolated

    def show(self) -> None:
        plt.show()


def load_audio(filename: str) -> Tuple[np.ndarray, float]:
    """
    Load a Wav file

    Parameters
    ----------
    filename : str
        Wav file to be loaded.

    Returns
    -------
    signal : np.ndarray
        The audio signal as a 1D numpy array. If the audio file is not mono,
        the signal will be averaged into a single channel
    sample_rate: float
        Sample rate of the audio file
    """
    # Read the audio file
    sample_rate, signal = wavfile.read(filename)

    # Ensure audio is mono
    if len(signal.shape) > 1 and signal.shape[1] > 1:
        # Convert to mono by averaging channels
        signal = signal.mean(axis=1)

    return signal, sample_rate


def compute_magnitude_spectrogram(
    signal: np.ndarray,
    sample_rate: float,
    window_size: int = 2**14,
    hop_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the spectrogram of a signal using short time Fourier transform.

    Parameters
    ----------
    signal: np.ndarray
        Audio signal as a 1D array.
    sample_rate: float
        Sample rate of the audio signal (in Hz)
    window_size: int
        Size of the window (in samples)
    hop_size: int
        Step size (in samples)

    Returns
    -------
    spectrogram: np.ndarray
        A 2D magnitude spectrogram with shape (n_frequency_bins, n_steps)
    freqs: np.ndarray
        Frequency bins of the signal.
    """

    # Number of windows
    num_windows = 1 + (len(signal) - window_size) // hop_size

    # Initialize the spectrogram with zeros
    spectrogram = np.zeros((window_size // 2, num_windows))

    # Window function (Hamming window)
    window = np.hamming(window_size)

    # Compute STFT
    for w in range(num_windows):
        start = w * hop_size
        end = start + window_size
        audio_segment = signal[start:end] * window
        spectrum = fft(audio_segment)[
            : window_size // 2
        ]  # Take only the positive frequencies
        magnitude = np.abs(spectrum)
        spectrogram[:, w] = magnitude

    # Compute frequency bins
    freqs = np.linspace(0, sample_rate / 2, window_size // 2)

    return spectrogram, freqs


class InteractiveToneAnalyzer(VerticalLineOnClickPlot):
    def __init__(
        self,
        signal: np.ndarray,
        sample_rate: float,
        window_size: int = 1024,
        hop_size: int = 256,
        figsize: Tuple[int, int] = (8, 6),
    ) -> None:
        spectrogram, freqs = compute_magnitude_spectrogram(
            signal=signal,
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
        )

        spectrogram_dist = spectrogram.sum(axis=1)
        spectrogram_dist /= max(spectrogram_dist.sum(), 1e-10)
        super().__init__(
            x=freqs,
            y=spectrogram_dist,
            figsize=figsize,
        )
        self.spectrogram = spectrogram
        self.audio_widget = None

        self.fundamental_freq_raw = freqs[spectrogram_dist.argmax()]
        self.fundamental_freq = MIDI_PITCH_FREQS[
            frequency_to_midi_pitch(self.fundamental_freq_raw)
        ]
        self.x_values.append(self.fundamental_freq)

        line = self.ax.axvline(
            self.fundamental_freq_raw,
            color="firebrick",
            linestyle="--",
            linewidth=3,
        )
        self.update_plot()

    def harmonics_dist(self, freq: int) -> Tuple[np.ndarray, np.ndarray]:
        overtones = np.array(self.x_values) / self.fundamental_freq
        weights = self.y_values
        weights /= 2 * weights.sum()

        return overtones * freq, weights

    def __call__(self, arg: Any) -> None:
        pass

    def show(self) -> None:
        plt.show()
        self.synthesize()
        ipd.display(self.audio_widget)

    def synthesize(self) -> None:
        freqs, weights = self.harmonics_dist(self.fundamental_freq)
        audio_signal = additive_synthesis(
            freqs=freqs,
            duration=1,
            samplerate=SAMPLE_RATE,
            weights=weights,
        )
        self.audio_widget = ipd.Audio(
            data=audio_signal,
            rate=SAMPLE_RATE,
            normalize=False,
            element_id="audio_display",
        )

    def update_plot(self) -> None:
        plt.draw()
        self.synthesize()
