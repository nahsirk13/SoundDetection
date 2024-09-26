

import wave
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    obj = wave.open('Death.wav')

    print("Number of Channels: ", obj.getnchannels())
    print("Sample width: ", obj.getsampwidth())
    print("Framerate: ", obj.getframerate())
    print("Number of frames: ", obj.getnframes())
    print("Parameters: ", obj.getparams())

    time_audio = (obj.getnframes() / obj.getframerate())
    print("Duration of audio: ", time_audio)

    frames = obj.readframes(-1)  #-1 reads all frames
    print(type(frames), type(frames[0]))
    print(len(frames))
    obj.close() #close wave object once done

    #open same wav in "write binary" mode
    obj_wb = wave.open('Death.wav', 'wb')
    obj_wb.setnchannels(1)
    obj_wb.setsampwidth(2)
    obj_wb.setframerate(16000)
    obj_wb.writeframes(frames)
    obj_wb.close()

    obj = wave.open('Death.wav', 'rb') #read binary mode
    sample_freq = obj.getframerate()
    n_samples = obj.getnframes()
    signal_wave = obj.readframes(-1)
    obj.close()
    t_audio = n_samples/sample_freq
    print(t_audio)
    signal_array = np.frombuffer(signal_wave, dtype=np.int16)
    times = np.linspace(0, t_audio, num=n_samples)

    plt.figure(figsize=(15, 5))
    plt.plot(times, signal_array)
    plt.title("Death Audio Signal")
    plt.ylabel("Signal wave")
    plt.xlabel("Time (s)")
    plt.xlim(0, t_audio)
    plt.show()
