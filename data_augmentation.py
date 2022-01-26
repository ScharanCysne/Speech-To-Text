import os
import warnings
import soundfile
import librosa, librosa.display

from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift

warnings.filterwarnings("ignore")

WAV_PATH = 'C:\\Users\\nicho\\OneDrive\\Desktop\\Projetos\\Kaggle Datasets\\LJSpeech-1.1\\wavs\\'

def augment_audio(wav_file, augment, type=''):
    # Load the audio file
    samples, sample_rate = librosa.load(WAV_PATH + wav_file, sr=None)
    
    # Augment/transform/perturb the audio data
    augmented_samples = augment(samples=samples, sample_rate=sample_rate)
    soundfile.write(WAV_PATH + wav_file[:-4] + type + '.wav', augmented_samples, sample_rate)


def main():
    counter = 0
    wav_files = os.listdir(WAV_PATH)
    num_files = len(wav_files)

    # Pre-process all wav files in WAV_PATH
    for wav_file in wav_files:
        f = os.path.join(WAV_PATH, wav_file)
        # checking if it is a file
        if not os.path.isfile(f):
            print("Error in " + f)
            continue
        
        # Add Gaussian Noise
        augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        ])
        augment_audio(wav_file, augment, '_noise')

        # Add Time Stretch
        augment = Compose([
            TimeStretch(min_rate=0.7, max_rate=1.3, p=0.5),
        ])
        augment_audio(wav_file, augment, '_stretch')

        # Add Pitch Shift
        augment = Compose([
            PitchShift(min_semitones=-5, max_semitones=5, p=0.5)
        ])
        augment_audio(wav_file, augment, '_shift')

        # Add Gaussian Noise and Time Stretch
        augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            TimeStretch(min_rate=0.7, max_rate=1.3, p=0.5),
        ])
        augment_audio(wav_file, augment, '_noise_stretch')

        # Add Gaussian Noise and Pitch Shift
        augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            PitchShift(min_semitones=-5, max_semitones=5, p=0.5)
        ])
        augment_audio(wav_file, augment, '_noise_shift')

        # Add Time Stretch and Pitch Shift
        augment = Compose([
            TimeStretch(min_rate=0.7, max_rate=1.3, p=0.5),
            PitchShift(min_semitones=-5, max_semitones=5, p=0.5)
        ])
        augment_audio(wav_file, augment, '_stretch_shift')

        # Add Gaussian Noise, Time Stretch and Pitch Shift
        augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            TimeStretch(min_rate=0.7, max_rate=1.3, p=0.5),
            PitchShift(min_semitones=-5, max_semitones=5, p=0.5)
        ])
        augment_audio(wav_file, augment, '_noise_stretch_shift')

        counter += 1
        print(f'Augmented {counter}th .wav file from {num_files} files')


if __name__ == '__main__':
    main()