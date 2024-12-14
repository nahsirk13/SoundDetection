import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, lfilter

def bandpass_filter(data, lowcut, highcut, sample_rate, order=5):
    """
    Apply a bandpass filter to the audio data.

    Parameters:
        data (numpy array): The audio data.
        lowcut (float): Lower cutoff frequency in Hz.
        highcut (float): Upper cutoff frequency in Hz.
        sample_rate (int): Sampling rate of the audio in Hz.
        order (int): Order of the filter. Higher values mean sharper filtering.

    Returns:
        numpy array: Filtered audio data.
    """
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = lfilter(b, a, data)
    return filtered_data

def create_spectrogram(file_path, lowcut, highcut, max_freq=6000):
    """
    Create a spectrogram from a .wav sound file after filtering frequencies.

    Parameters:
        file_path (str): Path to the .wav file.
        lowcut (float): Lower cutoff frequency in Hz.
        highcut (float): Upper cutoff frequency in Hz.
        max_freq (float): Maximum frequency to display in the spectrogram (default 6000 Hz).
    """
    # Read the wav file
    try:
        sample_rate, data = wavfile.read(file_path)
    except Exception as e:
        print(f"Error reading the file: {e}")
        return


    # If stereo, take only one channel
    if len(data.shape) > 1:
        data = data[:, 0]


    # Apply bandpass filter
    data = bandpass_filter(data, lowcut, highcut, sample_rate)

    # Create output folder
    output_folder = "spectrograms"
    os.makedirs(output_folder, exist_ok=True)

    # Generate spectrogram
    plt.figure(figsize=(10, 6))
    plt.specgram(data, Fs=sample_rate, NFFT=1024, noverlap=512, cmap='viridis')

    # Set frequency limit to max_freq (e.g., 6000 Hz)
    plt.ylim(0, max_freq)  # Limit frequency axis to 6000 Hz

    # Extract file name without extension for the title
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    # Add color bar, title, and labels
    plt.colorbar(label='Intensity [dB]')
    plt.title(f"Spectrogram of {file_name} (Filtered {lowcut}-{highcut} Hz)")
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')

    # Save the image
    file_name = os.path.splitext(os.path.basename(file_path))[0]  # Extract filename without extension
    output_path = os.path.join(output_folder, f"{file_name}.png")  # Keep the same name with .png extension
    plt.savefig(output_path)
    plt.show()
    plt.close()
    print(f"Spectrogram saved at: {output_path}")


def process_all_wav_files_in_folder(folder_path, lowcut, highcut, max_freq=6000):
    """
    Process all .wav files in the given folder.

    Parameters:
        folder_path (str): The path to the folder containing the .wav files.
        lowcut (float): Lower cutoff frequency for bandpass filter.
        highcut (float): Upper cutoff frequency for bandpass filter.
        max_freq (float): Maximum frequency to display in the spectrogram (default 6000 Hz).
    """
    # List all files in the folder
    for file_name in os.listdir(folder_path):
        # Check if the file is a .wav file
        if file_name.endswith(".wav"):
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing file: {file_path}")
            create_spectrogram(file_path, lowcut, highcut, max_freq)

# Example usage
folder_path = 'audio'  # Path to folder containing .wav files
lowcut = 100  # Lower cutoff frequency in Hz
highcut = 6000  # Upper cutoff frequency in Hz
process_all_wav_files_in_folder(folder_path, lowcut, highcut, max_freq=6000)