import librosa
import numpy as np
import os

class AudioPreprocessor(object):
    def __init__(self, sr=16000, n_dct_filters=40, n_mels=40, f_max=4000, f_min=20, n_fft=480, hop_length=160):
        super().__init__()
        self.n_mels = n_mels
        self.dct_filters = librosa.filters.dct(n_dct_filters, n_mels)
        self.sr = sr
        self.f_max = f_max if f_max is not None else sr // 2
        self.f_min = f_min
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.noises = dict()
        for file in os.listdir('../speech_commands/_background_noise_'):
            if file.endswith(".wav"):
                samples, sample_rate = librosa.load(os.path.join('../speech_commands/_background_noise_/'+file))
                filename = os.path.splitext(file)[0]
                self.noises[filename] = samples

    def compute_mfccs(self, audio_data):
        S = librosa.feature.melspectrogram( audio_data,
                                            sr=self.sr,
                                            n_mels=self.n_mels,
                                            hop_length=self.hop_length,
                                            n_fft=self.n_fft,
                                            fmin=self.f_min,
                                            fmax=self.f_max)
        # S = librosa.power_to_db(S)
        # S = [np.matmul(self.dct_filters, x) for x in np.split(audio_data, audio_data.shape[1], axis=1)]
        # S = np.array(S, order='F').reshape(1, S.shape[0], S.shape[1]).astype(np.float32)
        S =librosa.feature.mfcc(S=librosa.power_to_db(S),
                                n_mfcc=self.n_mels)
        return S


    def compute_mfccs2(self, audio_data):
        S = librosa.feature.mfcc(   audio_data,
                                    sr=self.sr,
                                    n_mfcc=self.n_mels,
                                    n_fft=self.n_fft,
                                    hop_length=self.hop_length,
                                    fmin=self.f_min,
                                    fmax=self.f_max)
        # S = S.reshape(1, S.shape[0], S.shape[1]).astype(np.float32)
        return S


    def pad_short_data(self, data, up_to=16000):
        p = np.zeros(up_to)

        offset = np.random.randint(up_to - len(data))

        p[offset:offset+len(data)] += data

        return p


    def noise(self, data, up_to=16000, padding='zeros', factor=0.1):
        """
        padding : method to use to fill the data with.
                -'zeros' : fill with zeros
                -'doing_the_dishes'
                -'dude_miaowing'
                -'exercise_bike'
                -'pink_noise'
                -'running_tap'
                -'white_noise'
        """
        if padding == 'zeros':
            p = np.zeros(up_to)
        elif padding in self.noises:
            noise_offset = np.random.randint(len(self.noises[padding]) - up_to + 1)
            p = self.noises[padding][noise_offset:noise_offset+up_to]
        else:
            raise ValueError('wrong type of padding.')

        offset = np.random.randint(up_to - len(data) + 1)

        p *= factor
        p[offset:offset+len(data)] += data

        return p



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
