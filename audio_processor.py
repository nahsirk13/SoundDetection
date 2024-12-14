import pyaudio
import numpy as np
import wave
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import joblib
from io import BytesIO
import scipy as sp
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

import extract_features
import sound_spectrum_analysis

#import mlpy
#import rpy2.robjects.numpy2ri
#rpy2.robjects.numpy2ri.activate()
#from rpy2.robjects.packages import importr
#import rpy2.robjects as robj

# load the pre-trained model

model = joblib.load("sound_classifier.pkl")

def record_audio(duration=5, fs=16000, filename="temp.wav"):
    """
    records audio for the specified duration and saves it to a file.
    """
    try:
        audio = pyaudio.PyAudio()

        # open a stream
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=fs, input=True, frames_per_buffer=1024)

        print("Recording...")
        frames = []

        for _ in range(0, int(fs / 1024 * duration)):
            data = stream.read(1024)
            frames.append(data)

        print("Recording finished.")

        # stop and close the stream
        stream.stop_stream()
        stream.close()
        audio.terminate()

        # save the recording to a WAV file
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(fs)
            wf.writeframes(b''.join(frames))

        return filename
    except Exception as e:
        print(f"Error during recording: {e}")
        return None

def extract_spectrogram_features(file_path, fs=16000):
    """
    extracts spectrogram-based features from an audio file.
    """
    try:
        with wave.open(file_path, 'rb') as wf:
            frames = wf.readframes(wf.getnframes())
            audio_data = np.frombuffer(frames, dtype=np.int16)

            # generate the spectrogram
            power_spectrum, freqs, times, _ = plt.specgram(audio_data, Fs=fs, NFFT=1024, noverlap=512)
            plt.close()

            # split frequencies into bands: low, mid, high
            low_band = power_spectrum[(freqs >= 0) & (freqs < 1000)].mean(axis=0)
            mid_band = power_spectrum[(freqs >= 1000) & (freqs < 5000)].mean(axis=0)
            high_band = power_spectrum[(freqs >= 5000)].mean(axis=0)

            # aggregate features
            features = np.array([
                np.mean(low_band),  # mean power in low frequencies
                np.mean(mid_band),  # mean power in mid frequencies
                np.mean(high_band)  # mean power in high frequencies
            ]).reshape(1, -1)

            # normalize features
            features = normalize(features)
            return features, audio_data
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        return None, None




def record_and_classify_with_spectrogram():
    """
    records audio, extracts spectrogram features, and classifies the sound.
    """
    try:
        # record audio
        audio_file = record_audio()
        if not audio_file:
            return "Recording failed.", None

        # extract features and audio data
        features, audio_data = extract_features.extract_features(audio_file)
        if features is None or audio_data is None:
            return "Feature extraction failed.", None

        # classify the sound
        prediction = model.predict(features)[0]

        # generate the spectrogram for visualization
        spectrogram = generate_spectrogram(audio_data)

        return prediction, spectrogram
    except Exception as e:
        print(f"Error during classification: {e}")
        return f"Error: {str(e)}", None

def generate_spectrogram(audio_data, fs=16000):
    """
    generates a spectrogram from audio data with proper axis scaling.
    """
    # create the figure and axes for the spectrogram
    fig, ax = plt.subplots(figsize=(6, 4))  # can tweak the size if needed
    spec, freqs, times, im = ax.specgram(audio_data, Fs=fs, NFFT=1024, noverlap=512, cmap='viridis')

    # make sure y-axis (frequencies) covers the full range up to nyquist
    ax.set_ylim(0, fs // 2)  # full frequency range, 0 to fs/2 (8000 Hz for 16kHz sampling rate)

    # make sure x-axis (time) covers the entire duration of the audio
    ax.set_xlim(0, len(audio_data) / fs)  # total time in seconds (samples / sampling rate)

    # turn off axis lines and labels for gui use
    ax.axis('off')

    # save the spectrogram as an image to use in the gui
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
    buffer.seek(0)

    # read the image into a numpy array for the gui
    image_array = plt.imread(buffer)
    buffer.close()
    plt.close(fig)

    return image_array




def direction_analysis_with_spectrogram():
    """
    performs direction analysis and returns the classification and spectrogram.
    """
    audio_file = record_audio()
    if sound_spectrum_analysis.microwave_different_height(audio_file) == 1:
        return "Different Height (Doppler Shift Found)"
    if sound_spectrum_analysis.microwave_different_height(audio_file) == -1:
        return "Same Height (Doppler Shift Not Found)"

#panneer functions

#calculate the entropy of fft values
def getEntropy(data):
	data_fft = sp.fft(data)
	data_fft_abs = np.abs(data_fft)
	data_fft_abs_sum = np.sum(data_fft_abs)
	data_fft_abs_norm = data_fft_abs/data_fft_abs_sum
	data_fft_abs_norm_log2 = np.log2(data_fft_abs_norm)
	result = - np.sum(data_fft_abs_norm * data_fft_abs_norm_log2)
	result = result/len(data_fft)
	return result

def getFFTCoeff(data):
	data_fft = abs(sp.fft(data))
	return data_fft[1:]

# get the first 5 largest fft coefficient
def getFFTPeaks(data):
	data_fft = sp.fft(data)
	result = np.sort(abs(data_fft))
	result = result[::-1]
	return result[1:6]

# calculate the second peak of autocorrelation of fft values
def getPitch(data):
	data_fft = sp.fft(data)
	result = np.correlate(data_fft, data_fft, 'full')
	result = np.sort(np.abs(result))
	return result[len(result)-2]

import sounddevice as sd
from scipy.io.wavfile import write
import os

def record_audio_to_file(duration=5, file_path="temp_audio.wav"):
    """Record audio and save to a temporary file."""
    try:
        print("Recording...")
        sample_rate = 44100  # CD-quality
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()  # Wait for recording to finish
        write(file_path, sample_rate, audio_data)  # Save as WAV file
        print(f"Recording saved to {file_path}")
        return file_path
    except Exception as e:
        print(f"Error recording audio: {e}")
        return None