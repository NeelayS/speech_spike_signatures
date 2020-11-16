import matplotlib.pyplot as plt
import librosa
from librosa import display
import os
import scipy
import numpy as np
import json

data_root = "data/FSDD/recordings"
lower_bound = 52.0
upper_bound = 10000.0


class Dataset:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data = []

    # Preprocessing each speech signal
    def _frame(self, samples, sample_rate, window_ms):
        stride_ms = window_ms / 2
        stride_size = int(0.001 * sample_rate * stride_ms)
        window_size = int(0.001 * sample_rate * window_ms)

        # Extract strided windows
        truncate_size = (len(samples) - window_size) % stride_size
        samples = samples[: len(samples) - truncate_size]
        nshape = (window_size, (len(samples) - window_size) // stride_size + 1)
        nstrides = (samples.strides[0], samples.strides[0] * stride_size)
        windows = np.lib.stride_tricks.as_strided(
            samples, shape=nshape, strides=nstrides
        )

        assert np.all(
            windows[:, 1] == samples[stride_size : (stride_size + window_size)]
        )

        # Window weighting, squared Fast Fourier Transform (fft), scaling
        weighting = np.hanning(window_size)[:, None]

        fft = np.fft.fft(windows * weighting, axis=0)
        fft = np.absolute(fft)
        fft = fft ** 2
        scale = np.sum(weighting ** 2) * sample_rate
        fft[1:-1, :] *= 2.0 / scale
        fft[(0, -1), :] /= scale
        fft = np.log(fft)

        # Prepare fft frequency list
        freqs = float(sample_rate) / window_size * np.arange(fft.shape[0])

        frames = []
        for i in range(40):
            bands = []
            band0 = []
            band1 = []
            band2 = []
            band3 = []
            band4 = []
            j = 0
            for freq in freqs:
                if freq <= 333.3:
                    band0.append(fft[j][i])
                elif freq > 333.3 and freq <= 666.7:
                    band1.append(fft[j][i])
                elif freq > 666.7 and freq <= 1333.3:
                    band2.append(fft[j][i])
                elif freq > 1333.3 and freq <= 2333.3:
                    band3.append(fft[j][i])
                elif freq > 2333.3 and freq <= 4000:
                    band4.append(fft[j][i])

                j += 1
            bands.append(np.sum(band0) / np.shape(band0)[0])
            bands.append(np.sum(band1) / np.shape(band1)[0])
            bands.append(np.sum(band2) / np.shape(band2)[0])
            bands.append(np.sum(band3) / np.shape(band3)[0])
            bands.append(np.sum(band4) / np.shape(band4)[0])
            frames.append(bands)

        return frames

    def preprocess(self):
        N = 40
        overlap = 0.5
        for file in os.listdir(self.root_dir):
            samples, sampling_rate = librosa.load(
                os.path.join(data_root, file),
                sr=None,
                mono=True,
                offset=0.0,
                duration=None,
            )
            window_size = ((len(samples) * 1000) / sampling_rate) / (
                N * (1 - overlap) + overlap
            )
            self.data.append(self._frame(samples, sampling_rate, window_size))

        self.data = np.array(self.data).reshape(1500, 200)

        # Normalisation between [lower_bound, upper_bound]
        for index in range(self.data.shape[0]):

            min_v = np.min(self.data[index])
            max_v = np.max(self.data[index])
            self.data[index] -= min_v
            self.data[index] /= max_v - min_v
            self.data[index] *= upper_bound - lower_bound
            self.data[index] += lower_bound


dataset = Dataset(data_root)
dataset.preprocess()
np.save("data", dataset.data)
print(dataset.data.max(), dataset.data.min())
