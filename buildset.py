import librosa
import os
from tqdm import tqdm
import preprocess
import torch.utils.data as utils

SAMPLING_DURATION = 16000

def build_dataset():

    dataset = []
    audio_preproc = preprocess.AudioPreprocessor()

    for folder in tqdm(next(os.walk('../speech_commands/'))[1]):
        tmp = []

        if folder != "_background_noise_":

            commande = int(folder in ["one","two","three"])

            for filename in tqdm(os.listdir('../speech_commands/'+folder)):
                
                y, _ = librosa.load(os.path.join("../speech_commands/"+folder+"/"+filename), sr=None)

                if len(y) < SAMPLING_DURATION:
                    y = audio_preproc.pad_short_data(y, up_to=SAMPLING_DURATION, padding='zeros')

                S = audio_preproc.compute_mfccs(y)

                tmp.append([S, commande])
            dataset.append(tmp)

    return dataset

ds = build_dataset()


def split(dataset, train_prop, val_prop, test_prop):
