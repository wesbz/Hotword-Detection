import librosa
import os
from tqdm import tqdm
import preprocess
import torch.utils.data as utils
import numpy as np
import torch

SAMPLING_DURATION = 16000
COMMANDS = ["one","two","three"]

def build_dataset():
    dataset = []
    dataset_append = dataset.append
    audio_preproc = preprocess.AudioPreprocessor()
    librosa_load = librosa.load # For performance purpose.
    os_path_join = os.path.join # For performance purpose

    for folder in tqdm(next(os.walk('../speech_commands/'))[1]):
        tmp = []
        tmp_append = tmp.append
        speakers = {}

        if folder != "_background_noise_":

            commande = int(folder in COMMANDS)

            for filename in tqdm(os.listdir('../speech_commands/'+folder)):

                if not commande:
                    prob_skip = np.random.rand()
                    if prob_skip > 0.2:
                        continue
                y, _ = librosa_load(os_path_join("../speech_commands/"+folder+"/"+filename), sr=None)

                if len(y) < SAMPLING_DURATION:
                    y = audio_preproc.pad_short_data(y, up_to=SAMPLING_DURATION)

                S = audio_preproc.compute_mfccs(y)

                tmp_append([S, np.array([commande])])
            dataset += tmp

    return dataset


def dataset_augment():
    dataset = []
    dataset_append = dataset.append
    audio_preproc = preprocess.AudioPreprocessor()
    librosa_load = librosa.load # For performance purpose.
    os_path_join = os.path.join # For performance purpose

    for folder in tqdm(COMMANDS):
        tmp = []
        tmp_append = tmp.append

        if folder != "_background_noise_":

            commande = 1

            for filename in tqdm(os.listdir('../speech_commands/'+folder)):

                y, _ = librosa_load(os_path_join("../speech_commands/"+folder+"/"+filename), sr=None)

                if len(y) < SAMPLING_DURATION:
                    y = audio_preproc.pad_short_data(y, up_to=SAMPLING_DURATION)

                S = audio_preproc.compute_mfccs(y)

                tmp_append([S, np.array([commande])])

                for padding in ['doing_the_dishes', 'dude_miaowing', 'exercise_bike', 'pink_noise', 'running_tap', 'white_noise']:
                    y_aug = audio_preproc.noise(y, up_to=SAMPLING_DURATION, padding=padding)
                    S = audio_preproc.compute_mfccs(y_aug)
                    tmp_append([S, np.array([commande])])

            dataset += tmp

    return dataset

# ds = build_dataset()
# np.save('../dataset_40mfcc_16k', ds)
#ds = np.load('../dataset_40mfcc_16k.npy')


def split(dataset, train_prop=0.8, val_prop=0.1, test_prop=0.1):
    N = len(dataset)

    shuffled_indices = list(range(N))
    np.random.shuffle(shuffled_indices)

    split_1, split_2 = int(train_prop*N), int((train_prop+val_prop)*N)
    train_indices, val_indices, test_indices = shuffled_indices[:split_1], shuffled_indices[split_1:split_2], shuffled_indices[split_2:]


    train_set = []
    val_set = []
    test_set = []

    tensor_x = torch.stack([torch.Tensor(dataset[i][0]) for i in train_indices])
    tensor_y = torch.stack([torch.Tensor(dataset[i][1]) for i in train_indices])
    train_set = utils.TensorDataset(tensor_x, tensor_y)

    tensor_x = torch.stack([torch.Tensor(dataset[i][0]) for i in val_indices])
    tensor_y = torch.stack([torch.Tensor(dataset[i][1]) for i in val_indices])
    val_set = utils.TensorDataset(tensor_x, tensor_y)

    tensor_x = torch.stack([torch.Tensor(dataset[i][0]) for i in test_indices])
    tensor_y = torch.stack([torch.Tensor(dataset[i][1]) for i in test_indices])
    test_set = utils.TensorDataset(tensor_x, tensor_y)

    return train_set, val_set, test_set
