import numpy as np
from scipy.fft import dct
from scipy.signal import stft
from scipy.fftpack import fft
from scipy.io import wavfile

#How It Works

#MFCC-like Approximation:
#         Uses the short-time Fourier transform (STFT) to calculate the spectrogram.
#          Applies a logarithmic scale to simulate the human auditory perception.
#Performs a discrete cosine transform (DCT) to approximate MFCCs.
#Zero-Crossing Rate:
#        Counts the number of times the signal crosses the zero line, normalized over the signal length.
#Spectral Centroid:
        #  Computes the "center of mass" of the frequency spectrum using the FFT.


import numpy as np
from scipy.signal import stft
from scipy.fft import fft
from scipy.io import wavfile
from scipy.fftpack import dct

def extract_features(file_path):
    try:
        sample_rate, audio_data = wavfile.read(file_path)

        # If stereo, take the first channel
        if audio_data.ndim > 1:
            audio_data = audio_data[:, 0]

        # Normalize the audio data
        audio_data = audio_data / np.max(np.abs(audio_data))

        # 1. MFCC-like Approximation
        _, _, Zxx = stft(audio_data, fs=sample_rate, nperseg=512)
        spectrogram = np.abs(Zxx)
        log_spectrogram = np.log(spectrogram + 1e-10)  # Avoid log(0)
        mfcc_approx = np.mean(dct(log_spectrogram, type=2, axis=0, norm='ortho')[:13], axis=1)

        # 2. Zero-Crossing Rate (ZCR)
        zcr = np.mean((audio_data[:-1] * audio_data[1:] < 0).astype(float))

        # 3. Spectral Centroid
        fft_spectrum = np.abs(fft(audio_data)[:len(audio_data) // 2])
        freqs = np.fft.fftfreq(len(audio_data), 1 / sample_rate)[:len(audio_data) // 2]
        spectral_centroid = np.sum(freqs * fft_spectrum) / np.sum(fft_spectrum)

        # 4. Spectral Roll-off (85% energy)
        cumulative_energy = np.cumsum(fft_spectrum)
        rolloff_threshold = 0.85 * cumulative_energy[-1]
        spectral_rolloff = freqs[np.searchsorted(cumulative_energy, rolloff_threshold)]

        # 5. Spectral Bandwidth
        spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid)**2) * fft_spectrum) / np.sum(fft_spectrum))

        # 6. RMS Energy
        rms_energy = np.sqrt(np.mean(audio_data**2))

        # Combine all features (update to 21 features as used in training)
        features = np.hstack((mfcc_approx, zcr, spectral_centroid, spectral_rolloff, spectral_bandwidth, rms_energy))

        # If additional features were used, ensure they are included here
        # e.g., Spectral Flatness, Pitch, etc.
        return features
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None
