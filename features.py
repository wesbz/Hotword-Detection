import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import csv

import pyaudio
import wave








audio="corpus/backward/0a2b400e_nohash_0.wav"


def add_csv(audio,commande):
    y, sr = librosa.load(audio)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
    mfcc_delta = librosa.feature.delta(mfcc)
    aux=[]
    for x in range(0,mfcc.shape[0]):
        aux+=list(mfcc[x])
    for z in range(0, mfcc_delta.shape[0]):
        aux += list(mfcc_delta[z])
    aux.append(commande)
    my_rounded_list = [round(elem, 2) for elem in aux]
    if((len(aux)==881)):

        with open('corpus_round.csv', 'a') as csvFile:
            writer = csv.writer(csvFile)

            writer.writerow(my_rounded_list)

        csvFile.close()


a=0
for folder in os.listdir("corpus_2"):
    l=[]
    for filename in os.listdir(os.path.join("corpus_2", folder)):
            commande=0
            if (folder=="one" or folder=="two" or folder=="three"):
                commande=1
            add_csv(os.path.join("corpus_2", folder,filename),commande)
            a+=1
            pc=round(a/26500,2)*100
            print(pc,"%")



