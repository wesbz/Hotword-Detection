# Juste pour avoir les longueurs des enregistrements audios

import os
from scipy.io import wavfile
from tqdm import tqdm

list_dir = next(os.walk('../speech_commands/'))[1]

too_short = 0
length_16k = 0
too_long = 0
wrong_samplerate = 0

for dir in tqdm(list_dir):
    if dir[0] != '_':
        for file in tqdm(os.listdir('../speech_commands/'+dir)):
            sample_rate, samples = wavfile.read('../speech_commands/'+dir+'/'+file)
            wrong_samplerate += int(sample_rate == 16000)
            too_short += int(len(samples) < 16000)
            too_long += int(len(samples) > 16000)
            length_16k += int(len(samples) == 16000)


# length_16k = 95394
# too_long = 0
# too_short = 10435
