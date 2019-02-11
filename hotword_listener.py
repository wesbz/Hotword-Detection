import pyaudio
import wave
from network import MyConvolutionalNetwork
import torch
import datetime
import librosa
import librosa.display
import numpy as np
import torch.utils.data as utils
from nnet import CNNTradPool1Conv1
from nnet import CNNTradPool1Conv2
from nnet import CNNTradPool1Conv3


#path of the trained models
path_cnntp_1conv = 'models/CNNTradPool1/one_convo_reduced_dataset_40MFCC'
path_cnntp_2conv = 'models/CNNTradPool1/two_convo_reduced_dataset_40MFCC'
path_cnntp_3conv = 'models/CNNTradPool1/three_convo_reduced_dataset_40MFCC'

#create an instance of the Neural Network
netCNNTradPool1Conv1 = CNNTradPool1Conv1()
netCNNTradPool1Conv2 = CNNTradPool1Conv2()
netCNNTradPool1Conv3 = CNNTradPool1Conv3()

#load the trained model of the NN
netCNNTradPool1Conv1.load_state_dict(torch.load(path_cnntp_1conv, map_location='cpu'))
net = netCNNTradPool1Conv1
netCNNTradPool1Conv2.load_state_dict(torch.load(path_cnntp_2conv, map_location='cpu'))
netCNNTradPool1Conv3.load_state_dict(torch.load(path_cnntp_3conv, map_location='cpu'))

#choose the net to be tested
# choose between :
# - netCNNTradPool1Conv1
# - netCNNTradPool1Conv2
# - netCNNTradPool1Conv3

net = netCNNTradPool1Conv3

batch_size = 64
size_mffc = 101 

def resize_mfcc(mfcc):
    mfcc2 = []
    missing = size_mffc - len(mfcc[0])
    if missing > 0:
        for i in range(len(mfcc)):
            mfcc2.append(np.append(mfcc[i], np.zeros(missing)))
        mfcc = np.array(mfcc2)
    return mfcc 

def predict(net, path_file):
    pred = test_file(net, path_file)
    if pred:
        print('hotword detected')
    else:
        print('not a hotword')

def test_file(net, path_file):
    y, sr = librosa.load(path_file, None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft = 480, hop_length = 160, fmin = 20, fmax = 4000)

    mfcc = resize_mfcc(mfcc)

    test_set_list = []
    for _ in range(batch_size):
        test_set_list.append(mfcc)
    test_set = np.array(test_set_list) 

    my_x = [x for x in test_set] 
    tensor_x = torch.stack([torch.Tensor(i) for i in my_x]) 

    output_net = net(tensor_x)
    print(output_net[0])
    if output_net[0] > 0.5:
        prediction = True
    else:
        prediction = False
    return prediction


timestamp = '{:%Y-%m-%d-%H-%M-%S}'.format(datetime.datetime.now())

FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 16000 
CHUNK = 1024
RECORD_SECONDS = 1
WAVE_OUTPUT_FILENAME = "recordings/recording-"+ timestamp+ ".wav"
 
audio = pyaudio.PyAudio()
 
# start Recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)
print("recording...")
frames = []
 
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)
print("finished recording")
 
 
# stop Recording
stream.stop_stream()
stream.close()
audio.terminate()
 
# save the audio file
waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(frames))
waveFile.close()

# launch a prediction on the new recorded audio file
predict(net, WAVE_OUTPUT_FILENAME)
