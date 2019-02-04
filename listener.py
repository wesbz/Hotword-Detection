import librosa
import pyaudio
import wave


class Listener:
    """docstring for Listener."""
    def __init__(self):
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 2
        self.RATE = 16000
        self.WAVE_OUTPUT_FILENAME = "buffer.wav"


    def listen_continuous(self):
        pa = pyaudio.PyAudio()

        def callback(in_data, frame_count, time_info, flag):
            audio_data = np.fromstring(in_data, dtype=np.float32)
            # Instead of printing, process here the audio chunk 'audio_data' with libROSA
            # [...]
            print(audio_data)
            return None, pyaudio.paContinue

        stream = pa.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        stream_callback=callback)

        stream.start_stream()

        while stream.is_active():
            time.sleep(0.25)

        stream.close()
        pa.terminate()


    def listen_sec(self,length_seconds):
        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNK)

        frames = []

        for i in range(0, int(self.RATE / self.CHUNK * length_seconds)):
            data = stream.read(self.CHUNK)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        self.p.terminate()
        return frames
