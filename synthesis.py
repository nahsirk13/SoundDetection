import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt

def generate_tone(frequency, duration, sample_rate=44100, amplitude=0.5):
    """
    Generate a sine wave tone at a specific frequency.

    Args:
        frequency (float): Frequency of the tone in Hz.
        duration (float): Duration of the tone in seconds.
        sample_rate (int): Sampling rate in Hz (default is 44100).
        amplitude (float): Amplitude of the tone (0.0 to 1.0, default is 0.5).

    Returns:
        numpy.ndarray: The generated wave as a numpy array.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = amplitude * np.sin(2 * np.pi * frequency * t)
    return wave

def plot_spectrogram(wave, sample_rate, frequency):
    """
    Plot the spectrogram of the given wave.

    Args:
        wave (numpy.ndarray): The sound wave array.
        sample_rate (int): The sampling rate in Hz.
        frequency (float): The frequency of the tone for labeling.
    """
    plt.figure(figsize=(10, 4))
    plt.specgram(wave, Fs=sample_rate, NFFT=1024, noverlap=512, cmap='viridis')
    plt.title(f"Spectrogram for {frequency} Hz Tone")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.colorbar(label="Intensity [dB]")
    plt.show()

# Parameters
sample_rate = 44100
duration = 15   # seconds
frequencies = range(2000, 5500, 1000)  # 2000 Hz to 5000 Hz in steps of 500 Hz

# for freq in frequencies:
#     input(f"Press Enter to play the {freq} Hz tone...")
#     print(f"Playing {freq} Hz tone...")
    # Generate tone
wave = generate_tone(10000, duration, sample_rate)
# Play sound
sd.play(wave, samplerate=sample_rate)
sd.wait()
    # print(f"Displaying spectrogram for {freq} Hz tone...")
    # # Plot spectrogram
    # plot_spectrogram(wave, sample_rate, freq)
