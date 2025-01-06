import numpy as np
import librosa
import librosa.display
import soundfile
import matplotlib.pyplot as plt
import gc
import pandas as pd
from io import BytesIO


def extract_features_from_spectrogram(y, sr):
    """
    Extract features from the audio signal.
    """
    features = []

    # BPM
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features.extend(tempo)

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features.extend([np.mean(zcr), np.median(zcr), np.std(zcr)])

    # Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.extend([np.mean(spectral_centroid), np.median(spectral_centroid), np.std(spectral_centroid)])

    # Spectral Contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features.extend([np.mean(spectral_contrast), np.median(spectral_contrast), np.std(spectral_contrast)])

    # Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.extend([np.mean(spectral_bandwidth), np.median(spectral_bandwidth), np.std(spectral_bandwidth)])

    # Spectral Rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.extend([np.mean(spectral_rolloff), np.median(spectral_rolloff), np.std(spectral_rolloff)])

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for coeff in mfcc:
        features.extend([np.mean(coeff), np.median(coeff), np.std(coeff)])

    # MFCC Derivation
    mfcc_delta = librosa.feature.delta(mfcc)
    for delta in mfcc_delta:
        features.extend([np.mean(delta), np.median(delta), np.std(delta)])

    return features

def generate_spectrogram(mp3_file, output_img=None, generate_mfcc = False, mfcc_output = None):
    """
    Generate a mel spectrogram from an MP3 file.
    """
    # Load the audio (y - np array of loaded audio, sr - sampling rate)
    try: 
        y, sr = librosa.load(mp3_file)

        if y is None or len(y) == 0:
            raise ValueError(f"Failed to load audio data from file {mp3_file}")

        # Check if the audio duration is too short
        duration = librosa.get_duration(y=y, sr=sr)
        if duration < 1.0:
            raise ValueError(f"Audio file {mp3_file} is too short (duration: {duration:.2f} seconds).")

        
        # Create the mel spectrogram and convert it to display dB
        n_fft = 2048
        n_mels = 128
        n_ts = 256
        hop_length = int((len(y) - n_fft) / (n_ts - 1))

        S = librosa.feature.melspectrogram(y=y, sr=sr,n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Plot the spectrogram
        # plt.figure(figsize=(10,4))
        plt.axis('off')
        plt.tight_layout(pad=0)

        features = None
        if generate_mfcc:
            features = extract_features_from_spectrogram(y, sr)
            if mfcc_output is not None:
                np.save(mfcc_output, features)

        librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None)
        if output_img is not None:
            plt.savefig(output_img, bbox_inches='tight', pad_inches=0)

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)

        plt.clf()
        plt.close()

    except Exception as e:
        raise Exception(f"File corrupted, could not generate Spectrogram: {e}")

    plt.clf()
    gc.collect()

    return buf, features
