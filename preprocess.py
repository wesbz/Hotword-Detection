import librosa
import numpy as np
import os
from scipy.io import wavfile

class AudioPreprocessor(object):
    def __init__(self, sr=16000, n_dct_filters=40, n_mels=40, f_max=4000, f_min=20, n_fft=480, hop_length=160):
        super().__init__()
        self.n_mels = n_mels
        self.dct_filters = librosa.filters.dct(n_dct_filters, n_mels)
        self.sr = sr
        self.f_max = f_max if f_max is not None else sr // 2
        self.f_min = f_min
        self.n_fft = n_fft
        self.hop_length = sr // 1000 * hop_ms

    def compute_mfccs(self, audio_data):
        S = librosa.feature.melspectrogram( audio_data,
                                            sr=self.sr,
                                            n_mels=self.n_mels,
                                            hop_length=self.hop_length,
                                            n_fft=self.n_fft,
                                            fmin=self.f_min,
                                            fmax=self.f_max)
        S = librosa.power_to_db(S)
        S = [np.matmul(self.dct_filters, x) for x in np.split(data, data.shape[1], axis=1)]
        S = np.array(S, order='F').reshape(1, S.shape[0], S.shape[1]).astype(np.float32)
        return S

    def compute_mfccs2(self, audio_data):
        S = librosa.feature.mfcc(   audio_data
                                    sr=self.sr,
                                    n_mfcc=self.n_mels,
                                    n_fft=self.n_fft,
                                    hop_length=self.hop_length,
                                    fmin=self.f_min,
                                    fmax=self.f_max)
        S = S.reshape(1, S.shape[0], S.shape[1]).astype(np.float32)
        return S



class AudioSnippet(object):
    def __init__(self, byte_data=b"", dtype=np.int16):
        self.byte_data = byte_data
        self.dtype = dtype
        self._compute_amps()


    def save(self, filename):
        with wave.open(filename, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(16000)
            f.writeframes(self.byte_data)


    def _compute_amps(self):
        self.amplitudes = np.frombuffer(self.byte_data, self.dtype).astype(float) / np.iinfo(self.dtype).max

noises = dict()
for file in os.listdir('../speech_commands/_background_noise_'):
    sample_rate, samples = wavfile.read('../speech_commands/_background_noise_/'+file)
    if file.endswith(".wav"):
        filename = os.path.splitext(file)[0]
        noises[filename] = samples.astype(float) / np.iinfo(samples.dtype).max


noises = {  "doing_the_dishes": wavfile.read('../speech_commands/_background_noise_/doing_the_dishes.wav')[1].astype(float) / np.iinfo(np.int16).max,
            "dude_miaowing": wavfile.read('../speech_commands/_background_noise_/dude_miaowing.wav')[1].astype(float) / np.iinfo(),
            "exercise_bike": wavfile.read('../speech_commands/_background_noise_/exercise_bike.wav')[1].astype(float) / np.iinfo(),
            "pink_noise": wavfile.read('../speech_commands/_background_noise_/pink_noise.wav')[1].astype(float) / np.iinfo(),
            "running_tap": wavfile.read('../speech_commands/_background_noise_/running_tap.wav')[1].astype(float) / np.iinfo(),
            "white_noise": wavfile.read('../speech_commands/_background_noise_/white_noise.wav')[1].astype(float) / np.iinfo() }
