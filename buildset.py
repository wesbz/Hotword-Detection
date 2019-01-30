import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import csv

import pyaudio
import wave
import pandas as pd


df = pd.read_csv('corpus_round.csv', header=None)


print(df.count)

ds = df.sample(frac=1)

ds.to_csv('corpus_rand.csv')