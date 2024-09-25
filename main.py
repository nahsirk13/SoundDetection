# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.

import wave

obj = wave.open('Death.wav')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
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
