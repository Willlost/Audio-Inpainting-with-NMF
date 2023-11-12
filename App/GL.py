import librosa, librosa.display, soundfile
import numpy as np

audio,sr = librosa.load('train_data.wav',sr=44100)
mel = np.abs(librosa.feature.melspectrogram(audio))
gl = librosa.feature.inverse.mel_to_audio(mel)
gl = np.pad(gl, (0,audio.size-gl.size), 'constant')
loss = np.sum(np.abs(audio-gl))/audio.size
print(loss)
soundfile.write('gl.wav', gl, sr)
soundfile.write('aud.wav', audio, sr)