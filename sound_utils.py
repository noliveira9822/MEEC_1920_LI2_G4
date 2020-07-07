import soundfile
import numpy as np
import librosa
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def feature_extract_classify(data, samplerate, **kwargs):
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")

    x_norm = librosa.util.normalize(data)

    if chroma or contrast:
        stft = np.abs(librosa.stft(x_norm))
    result = np.array([])

    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=x_norm, sr=samplerate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))

    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=samplerate).T, axis=0)
        result = np.hstack((result, chroma))

    if mel:
        mel = np.mean(librosa.feature.melspectrogram(x_norm, sr=samplerate).T, axis=0)
        result = np.hstack((result, mel))

    if tonnetz:
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(x_norm), sr=samplerate).T, axis=0)
        result = np.hstack((result, tonnetz))

    return result


def feature_extract_train(filename, shift_dir, **kwargs):
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")

    with soundfile.SoundFile(filename) as sound_file:
        x = sound_file.read(dtype="float32")
        samplerate = sound_file.samplerate

        '''# adds noise
        if (noise):
            noise_to_Apply = np.random.randn(len(x))
            augmented_data = x + 0.2 * noise_to_Apply
            # Cast back to same data type
            augmented_data = augmented_data.astype(type(x[0]))
            x = augmented_data
'''
        to_shift = np.random.randint(samplerate * 2)
        if shift_dir == 'right':
            to_shift = -to_shift
        elif shift_dir == 'both':
            direction = np.random.randint(0, 2)
            if direction == 1:
                to_shift = -to_shift

        augmented_data = np.roll(x, to_shift)

        # Set to silence for heading/ tailing
        if to_shift > 0:
            augmented_data[:to_shift] = 0
        else:
            augmented_data[to_shift:] = 0

        # normaliza o sinal
        print(x)
        x_norm = librosa.util.normalize(x)
        # scaler = StandardScaler()
        # print(x.shape) 41984,
        # scaler.fit(x.reshape(-1, 1))
        # x_norma = scaler.transform(x.reshape(-1, 1))
        # x_norm = x_norma.reshape(-1, )
        # print(x_norm.shape)
        print(x_norm)

        if chroma or contrast:
            stft = np.abs(librosa.stft(x_norm))
        result = np.array([])

        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=x_norm, sr=samplerate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))

        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=samplerate).T, axis=0)
            result = np.hstack((result, chroma))

        if mel:
            mel = np.mean(librosa.feature.melspectrogram(x_norm, sr=samplerate).T, axis=0)
            result = np.hstack((result, mel))

        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(x_norm), sr=samplerate).T, axis=0)
            result = np.hstack((result, tonnetz))

    return result


def load_commands_data(test_size=0.25):
    x, y = [], []
    for folder in glob.glob("sound\G*"):
        for file in glob.glob(folder + "\*.wav"):
            basename = os.path.basename(file)
            command = basename.split("_")[1]
            with soundfile.SoundFile(file) as sound_file:
                data = sound_file.read(dtype="float32")
                if len(data) == 0:
                    print("Ficheiro: " + file + " vazio")
                    continue
            features = feature_extract_train(file, shift_dir="right", mfcc=True, chroma=True, mel=True)
            x.append(features)
            y.append(command)

    return train_test_split(np.array(x), y, test_size=test_size, random_state=7)


def load_groups_data(test_size=0.25):
    x, y = [], []
    for folder in glob.glob("sound\G*"):
        grupo = folder.split("\\")[1]
        print(grupo)
        for file in glob.glob(folder + "\*.wav"):
            with soundfile.SoundFile(file) as sound_file:
                data = sound_file.read(dtype="float32")
                if len(data) == 0:
                    print("Ficheiro: " + file + " vazio")
                    continue
            features = feature_extract_train(file, shift_dir='both', mfcc=True, chroma=True, mel=True)
            x.append(features)
            y.append(grupo)

    return train_test_split(np.array(x), y, test_size=test_size, random_state=7)
