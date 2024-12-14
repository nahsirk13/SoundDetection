import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.io import wavfile
from scipy.signal import butter, lfilter, spectrogram, filtfilt, find_peaks
from scipy.signal import stft
from scipy.fftpack import fft, dct
from scipy.io import wavfile




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

def data_filter(data, cutoff_freq, fs, order=5,btype='high'):
    
    # Nyquist frequency
    nyq = 0.5 * fs
    # Normalized cutoff frequency
    normal_cutoff = cutoff_freq / nyq
    print("The normalized cut_off is {}".format(cutoff_freq))
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    # Apply the filter to the data
    filtered_data = filtfilt(b, a, data)
    return filtered_data


#Extract Features - We're getting 3 relevant features for sound identification mode:

#MFCC-like Approximation:
#         Uses the short-time Fourier transform (STFT) to calculate the spectrogram.
#          Applies a logarithmic scale to simulate the human auditory perception.
#Performs a discrete cosine transform (DCT) to approximate MFCCs.
#Zero-Crossing Rate:
#        Counts the number of times the signal crosses the zero line, normalized over the signal length.
#Spectral Centroid:
        #  Computes the "center of mass" of the frequency spectrum using the FFT.

def extract_features(file_path):
    try:
        # Load the audio file
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
        fft_spectrum = np.abs(fft(audio_data)[:len(audio_data) // 2])  # Use positive frequencies
        freqs = np.fft.fftfreq(len(audio_data), 1 / sample_rate)[:len(audio_data) // 2]
        spectral_centroid = np.sum(freqs * fft_spectrum) / np.sum(fft_spectrum)

        # 4. Spectral Roll-off (85% energy)
        cumulative_energy = np.cumsum(fft_spectrum)
        total_energy = cumulative_energy[-1]
        rolloff_threshold = 0.85 * total_energy
        spectral_rolloff = freqs[np.searchsorted(cumulative_energy, rolloff_threshold)]

        # 5. Spectral Bandwidth
        spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid)**2) * fft_spectrum) / np.sum(fft_spectrum))

        # 6. RMS Energy
        rms_energy = np.sqrt(np.mean(audio_data**2))

        # Combine all features into a single vector
        features = np.hstack((mfcc_approx, zcr, spectral_centroid, spectral_rolloff, spectral_bandwidth, rms_energy))
        return features
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

def create_spectrogram(file_path, lowcut=None, highcut=None, max_freq=12000):
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

    # data=data_filter(data,cutoff_freq=2500,fs=sample_rate,btype="low")
    # Apply bandpass filter
    # data = bandpass_filter(data, lowcut, highcut, sample_rate)

    # Create output folder
    # output_folder = "spectrograms"
    # os.makedirs(output_folder, exist_ok=True)
    # f,t,spect=spectrogram(data,fs=sample_rate,nfft=5120,nperseg=5120,noverlap=2048)
    plt.figure(figsize=(10, 6))
    spect, f, t, _ =plt.specgram(data, Fs=sample_rate, NFFT=5120, noverlap=2048, cmap='viridis')
    plt.ylim(0,8000)
    plt.show()
    plt.close()
    spect=np.log(spect)
    plt.imshow(spect)
    plt.show()
    plt.tight_layout()
    print(spect.shape)
    spect_2000=spect[200:300,:]
    spect_4000=spect[400:500,:]
    spect_6000=spect[600:700,:]
    f_2000=f[200:300]
    f_4000=f[400:500]
    f_6000=f[600:700]
    print(spect_2000.shape,spect_4000.shape,spect_6000.shape)
    plt.close()
    print(spect.shape)
    fig,axes=plt.subplots(1,3,figsize=(10,5))
    axes[0].imshow(spect_2000,cmap='viridis')
    axes[1].imshow(spect_4000,cmap='viridis')
    axes[2].imshow(spect_6000,cmap='viridis')
    # plt.colorbar()  
    plt.title(file_path)
    plt.show()
    exit(0)
    max_freq=np.argmax(spect,axis=0)
    max_frequencies_obtained=[]
    for i in range(max_freq.shape[0]):
        max_frequencies_obtained.append(f[max_freq[i]])
    diff=np.abs(4000-np.array(max_frequencies_obtained))
    fig,axes=plt.subplots(2,1,figsize=(10,5))
    axes[0].plot(diff,label="diff")
    axes[0].legend()
    axes[0].set_ylim(0,50)
    axes[1].plot(max_frequencies_obtained,label="original_Freq")
    axes[1].legend()
    plt.title(file_path)
    plt.show()

    print(max_freq.shape)
    print(spect.shape)
    # Generate spectrogram
    # data=bandpass_filter(data,1,3000,sample_rate)
    
    # Set frequency limit to max_freq (e.g., 6000 Hz)
    plt.ylim(3000, max_freq)  # Limit frequency axis to 6000 Hz

    # Add color bar, title, and labels
    plt.colorbar(label='Intensity [dB]')
    plt.title(f'Spectrogram (Filtered {lowcut}-{highcut} Hz)')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.show()
    # Save the image
    # file_name = os.path.splitext(os.path.basename(file_path))[0]  # Extract filename without extension
    # output_path = os.path.join(output_folder, f"{file_name}.png")  # Keep the same name with .png extension
    # plt.savefig(output_path)
    # plt.close()
    # print(f"Spectrogram saved at: {output_path}")



def compute_doppler_shifts(file_path,fundamental=2000,first_harmonics=4000,second_harmonics=6000):
    """

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

    spect, f, t, _ =plt.specgram(data, Fs=sample_rate, NFFT=5120, noverlap=2048, cmap='viridis')
    plt.close()
    spect=np.log(spect)
    #We separate the fundamental, first and second harmonics part of the sepctrogram
    fundamental_indices=np.where((f >= 1950) & (f <= 2050))[0]
    first_harmonics_indices=np.where((f >= 3950) & (f <= 4050))[0]
    second_harmonics_indices=np.where((f >= 5950) & (f <= 6050))[0]
    spect_fundamental=spect[fundamental_indices,:]
    spect_first_haromics=spect[first_harmonics_indices,:]
    spect_second_harmonics=spect[second_harmonics_indices,:]
    f_fundamental=f[fundamental_indices]
    f_first_haromics=f[first_harmonics_indices]
    f_second_harmonics=f[second_harmonics_indices]
    print(f_fundamental)
    print(f_first_haromics)
    print(f_second_harmonics)

    #Compute the variance in fundamental frequency
    max_freq_val_fundamental=np.max(spect_fundamental,axis=0) # Gives you the maximum value in time
    max_freq_val_fundamental[max_freq_val_fundamental<0]=0 # Zero values which are less than a particular threshold
    max_freq_fundamental=np.argmax(spect_fundamental,axis=0) # Gives you the index of the maximum value in time
    max_frequencies_obtained_fundamental=[]
    for i in range(max_freq_fundamental.shape[0]):
        max_frequencies_obtained_fundamental.append(f_fundamental[max_freq_fundamental[i]])#Get the frequency for the maximum value over time
    diff_fundamental=np.abs(2000-np.array(max_frequencies_obtained_fundamental)) # Compute difference between the known frequencey and the maximum the doppler shift
    diff_fundamental=diff_fundamental*max_freq_val_fundamental # Scale the doppler shift to 
    diff_fundamental[diff_fundamental<0]=0
    #Compute the variance in fundamental frequencyW
    max_first_val_haromics=np.max(spect_first_haromics,axis=0)
    max_first_val_haromics[max_first_val_haromics<0]=0
    max_first_haromics=np.argmax(spect_first_haromics,axis=0)
    max_frequencies_obtained_first_haromics=[]
    for i in range(max_first_haromics.shape[0]):
        max_frequencies_obtained_first_haromics.append(f_first_haromics[max_first_haromics[i]])
    diff_first_haromics=4000-np.array(max_frequencies_obtained_first_haromics)
    diff_first_haromics=diff_first_haromics*max_first_val_haromics
    diff_first_haromics[diff_first_haromics<0]=0
    #Compute the variance in fundamental frequency
    max_second_val_haromics=np.max(spect_second_harmonics,axis=0)
    max_second_val_haromics[max_second_val_haromics<0]=0
    max_second_haromics=np.argmax(spect_second_harmonics,axis=0)
    max_frequencies_obtained_second_haromics=[]
    for i in range(max_second_haromics.shape[0]):
        max_frequencies_obtained_second_haromics.append(f_second_harmonics[max_second_haromics[i]])
    diff_second_haromics=6000-np.array(max_frequencies_obtained_second_haromics)
    diff_second_haromics=diff_second_haromics*max_second_val_haromics
    diff_second_haromics[diff_second_haromics<0]=0
    fig,axes=plt.subplots(3,1,figsize=(10,5))
    axes[0].plot(diff_fundamental,label="diff_fundamental")
    axes[0].legend()
    # axes[0].set_ylim(-50,50)
    axes[1].plot(diff_first_haromics,label="diff_first_harmonics")
    axes[1].legend()
    # axes[1].set_ylim(-50,50)
    axes[2].plot(diff_second_haromics,label="diff_second_harmonics")
    axes[2].legend()
    # axes[2].set_ylim(-50,50)
    plt.title(file_path)
    plt.show()

    #find average of array to compare and print
    print(np.mean(diff_second_haromics))
    print(np.mean(diff_first_haromics))

    #return the average of first harmonics with second harmonics for thre
    print(f"{file_path} : Average Difference Fundamental:", np.mean(diff_fundamental))
    print(f"{file_path} : Average Difference Second Harmonics:", np.mean(diff_second_haromics))
    print(f"{file_path} : Average Difference Third Harmonic:", np.mean(diff_first_haromics))
    return (np.mean(diff_second_haromics))


def microwave_different_height(file_path):
    #find fundamental frequency
    fundamental = find_fundamental_frequency(file_path)
    #threshhold - most microwaves are around 2000Hz, and the mean of seconds harmonic doppler shift is greater than 2
    if compute_doppler_shifts(file_path, fundamental, fundamental * 2, fundamental * 3) > 2:
        return 1
    else:
        return -1


def find_fundamental_frequency(file_path):
    """Finds the fundamental frequency of a WAV file."""

    # Read the WAV file
    sample_rate, data = wavfile.read(file_path)

    # Perform Fourier Transform
    fft_data = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(data), 1/sample_rate)

    # Find the positive frequencies and their corresponding magnitudes
    positive_freqs = freqs[freqs > 0]
    positive_fft_data = np.abs(fft_data[freqs > 0])

    # Find the peak frequencies
    peaks, _ = find_peaks(positive_fft_data, height=0.1 * np.max(positive_fft_data))

    # If no peaks are found, return 0
    if len(peaks) == 0:
        return 0

    # The fundamental frequency is the first peak
    fundamental_frequency = positive_freqs[peaks[0]]

    return fundamental_frequency
    

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
if __name__=="__main__":
    compute_doppler_shifts("audio/up_down_1ft_faster.wav")
    compute_doppler_shifts("audio/still_1ft.wav")

    create_spectrogram("audio/up_down_1ft_faster.wav")


# create_spectrogram("../audio/still_1ft.wav")
# folder_path = 'audio'  # Replace with the path to your folder containing .wav files
# lowcut = 100  # Lower cutoff frequency in Hz
# highcut = 1500  # Upper cutoff frequency in Hz
# process_all_wav_files_in_folder(folder_path, lowcut, highcut, max_freq=1500)
