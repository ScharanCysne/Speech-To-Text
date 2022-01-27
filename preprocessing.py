'''
Helpful Links:

https://towardsdatascience.com/audio-deep-learning-made-simple-part-1-state-of-the-art-techniques-da1d3dff2504
https://towardsdatascience.com/audio-deep-learning-made-simple-part-2-why-mel-spectrograms-perform-better-aad889a93505
https://towardsdatascience.com/audio-deep-learning-made-simple-part-3-data-preparation-and-augmentation-24c6e1f6b52
https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5
https://towardsdatascience.com/audio-deep-learning-made-simple-automatic-speech-recognition-asr-how-it-works-716cfce4c706
https://towardsdatascience.com/foundations-of-nlp-explained-visually-beam-search-how-it-works-1586b9849a24
'''

import os
import sklearn
import warnings
import librosa, librosa.display

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

warnings.filterwarnings("ignore")

WAV_PATH = 'C:\\Users\\nicho\\OneDrive\\Desktop\\Projetos\\Kaggle Datasets\\LJSpeech-1.1\\wavs\\'
DURATION = 264600

def data_cleaning(samples):
    pad_width = DURATION - len(samples)
    if pad_width % 2 == 0:
        return np.pad(samples, (pad_width/2, pad_width/2), 'constant', constant_values=(0,0))
    else:
        return np.pad(samples, (pad_width//2, pad_width//2 + 1), 'constant', constant_values=(0,0))


def preprocess(wav_file, debug=False):
    # Load the audio file
    samples, sample_rate = librosa.load(WAV_PATH + wav_file, sr=None)
    # Convert to uniform dimensions: sample rate, channels, and duration
    samples = data_cleaning(samples)

    # Generate Simple Spectrogram with Short-Term Fourier Transform
    spectrogram = librosa.stft(samples)
    # Spectrogram in Mel Scale instead of linear frequency
    spectrogram_mag, _ = librosa.magphase(spectrogram)
    mel_scale_spectrogram = librosa.feature.melspectrogram(S=spectrogram_mag, sr=sample_rate)
    # Decibel Scale to get the final Mel Spectrogram
    mel_spectrogram = librosa.amplitude_to_db(mel_scale_spectrogram, ref=np.min)
    
    # Mel Frequency Cepstral Coefficients
    mfcc = librosa.feature.mfcc(samples, sr=sample_rate)
    # Center MFCC coefficient dimensions to the mean and unit variance
    mfcc = sklearn.preprocessing.scale(mfcc, axis=1)

    if debug:
        plt.figure(figsize=(14, 5))
        librosa.display.waveplot(samples, sr=sample_rate)
        plt.show()

        plt.figure(figsize=(14, 5))
        librosa.display.specshow(mel_spectrogram, sr=sample_rate, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.show()

        plt.figure(figsize=(14, 5))
        librosa.display.specshow(mfcc, sr=sample_rate, x_axis='time')
        plt.show()

    # Save Spectogram and MFCC
    fig = plt.Figure()
    FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.axis("off") 
    librosa.display.specshow(mel_spectrogram, sr=sample_rate, ax=ax, x_axis='time', y_axis='mel')
    fig.savefig(f'.\\spectograms\\{wav_file[:-4]}.png', bbox_inches='tight', pad_inches=0)

    fig = plt.Figure()
    FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.axis("off") 
    librosa.display.specshow(mfcc, sr=sample_rate, ax=ax, x_axis='time')
    fig.savefig(f'.\\mfcc\\{wav_file[:-4]}.png', bbox_inches='tight', pad_inches=0)


def main():
    counter = 0
    # Pre-process all wav files in WAV_PATH
    for wav_file in os.listdir(WAV_PATH):
        f = os.path.join(WAV_PATH, wav_file)
        # checking if it is a file
        if not os.path.isfile(f):
            print("Error in " + f)
            continue
        preprocess(wav_file, debug=False)
        counter += 1
        if counter % 1000 == 0:
            print(f'Preprocessed {counter}th .wav file from {len(os.listdir(WAV_PATH))} files')


if __name__ == '__main__':
    main()