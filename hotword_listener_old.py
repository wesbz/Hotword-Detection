import pyaudio
import wave
from network import MyConvolutionalNetwork
import torch
import datetime
import librosa
import librosa.display
import numpy as np
import torch.utils.data as utils
import network_old
path_1_conv_old = "models/model_one_convolution_old"
path_1_conv = "models/model_one_convolution"
path_3_conv = "models/model_three_convolutions"

#create an instance of the Neural Network
net = network_old.MyConvolutionalNetwork()
#load the trained model of the NN
net.load_state_dict(torch.load(path_1_conv_old, map_location='cpu'))

path_file_to_test = "recordings/recording-2019-02-07-16-38-20.wav"
path_one = 'data/corpus_test/one/0a2b400e_nohash_0.wav'
path_happy = 'data/corpus_test/happy/0a5636ca_nohash_0.wav'
path_happy = 'data/corpus_test/happy/0a5636ca_nohash_0.wav'

batch_size = 4
size_mffc = 44 #101 

def resize_mfcc(mfcc):
    #print('size mfcc[0] : ', len(mfcc[0]))
    mfcc2 = []
    missing = size_mffc - len(mfcc[0])
    if missing > 0:
        for i in range(len(mfcc)):
            mfcc2.append(np.append(mfcc[i], np.zeros(missing)))
        mfcc = np.array(mfcc2)
    #print('size mfcc[0] : ', len(mfcc[0]))
    return mfcc 

def test_file(net, path_file):
    y, sr = librosa.load(path_file, None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)


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
RATE = 22050 #16000 #22050
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
 
waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(frames))
waveFile.close()




def predict(net, path_file):
    pred = test_file(net, path_file)
    if pred:
        print('hotword detected')
    else:
        print('not a hotword')

predict(net, WAVE_OUTPUT_FILENAME)


#print('One')
#predict(net, path_one)
#print('Happy')
#predict(net, path_happy)
