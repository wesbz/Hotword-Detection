import librosa
import pyaudio
import wave
import datetime
import numpy as np


class Listener:
    """docstring for Listener."""
    def __init__(self):
        self.CHUNK = 1600
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 2
        self.RATE = 16000
        self.WAVE_OUTPUT_FILENAME = "buffer.wav"


    def listen_continuous(self):
        self.pa = pyaudio.PyAudio()

        def callback(in_data, frame_count, time_info, flag):
            audio_data = np.fromstring(in_data, dtype=np.int16)
            # Instead of printing, process here the audio chunk 'audio_data' with libROSA
            # [...]
            print(audio_data)
            return None, pyaudio.paContinue

        stream = self.pa.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNK,
                        stream_callback=callback)

        stream.start_stream()

        while stream.is_active():
            time.sleep(0.25)

        stream.close()
        pa.terminate()


    def listen_sec(self, length_seconds):
        self.pa = pyaudio.PyAudio()
        stream = self.pa.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNK)

        frames = []

        print('recording...')

        for i in range(0, int(self.RATE / self.CHUNK * length_seconds)):
            data = stream.read(self.CHUNK)
            frames.append(data)

        print('done')

        stream.stop_stream()
        stream.close()
        self.pa.terminate()
        return frames

    def save_wav(self, frames):
        timestamp = '{:%Y-%m-%d-%H-%M-%S}'.format(datetime.datetime.now())
        waveFile = wave.open('../record-'+timestamp+'.wav', 'wb')
        waveFile.setnchannels(self.CHANNELS)
        waveFile.setsampwidth(self.pa.get_sample_size(self.FORMAT))
        waveFile.setframerate(self.RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()
        return '../record-'+timestamp+'.wav'
