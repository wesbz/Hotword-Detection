listener = Listener()
frames = listener.listen_sec(1)
name = listener.save_wav(frames)
y, sr = librosa.load(name, None)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft =480, hop_length = 160, fmin = 20, fmax = 4000)

test = torch.stack([torch.Tensor(mfcc), torch.Tensor(mfcc), torch.Tensor(mfcc)])

output = net(test)
print(output)
