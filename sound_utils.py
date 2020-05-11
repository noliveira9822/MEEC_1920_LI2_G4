import soundfile
import numpy as np
import librosa
import glob
import os
from sklearn.model_selection import train_test_split


def feature_extract(filename, **kwargs):
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")

    with soundfile.SoundFile(filename) as sound_file:
        x = sound_file.read(dtype="float32")
        samplerate = sound_file.samplerate

        # normaliza o sinal
        x = librosa.util.normalize(x)

        if chroma or contrast:
            stft = np.abs(librosa.stft(x))
        result = np.array([])

        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=x, sr=samplerate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))

        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=samplerate).T, axis=0)
            result = np.hstack((result, chroma))

        if mel:
            mel = np.mean(librosa.feature.melspectrogram(x, sr=samplerate).T, axis=0)
            result = np.hstack((result, mel))

        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(x), sr=samplerate).T, axis=0)
            result = np.hstack((result, tonnetz))

    return result


def load_commands_data(test_size=0.2):
    x, words = [], []
    for file in glob.glob("sound/*/*.wav"):
        basename = os.path.basename(file)
        comando = basename.split("_")[1]
        words.append(comando)
        # grupo.append([basename.split('_')[0]])

        features = feature_extract(file, mfcc=True, chroma=True, mel=True)
        x.append(features)
    print(words)
    return train_test_split(np.array(x), words, test_size=test_size, random_state=8)


def load_groups_data(test_size=0.2):
    x, grupos = [], []
    for file in glob.glob("sound/*/*.wav"):
        basename = os.path.basename(file)
        grupo = basename.split("_")[0]
        grupos.append(grupo)
        # grupo.append([basename.split('_')[0]])

        features = feature_extract(file, mfcc=True, chroma=True, mel=True)
        x.append(features)
    print(grupos)
    return train_test_split(np.array(x), grupos, test_size=test_size, random_state=8)
