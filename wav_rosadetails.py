import librosa
import numpy as np
import os
import sys

def analyze_audio(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")

    # Load audio
    y, sr = librosa.load(file_path, sr=None)

    # Basic properties
    duration = librosa.get_duration(y=y, sr=sr)
    rms = librosa.feature.rms(y=y).mean()
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    zero_crossings = librosa.zero_crossings(y, pad=False).sum()
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)

    # Compile results
    results = {
        'filename': os.path.basename(file_path),
        'sample_rate': sr,
        'duration_sec': duration,
        'rms_energy': rms,
        'tempo_bpm': tempo,
        'zero_crossing_rate': zero_crossings / len(y),
        'spectral_centroid': spectral_centroid,
        'spectral_rolloff': spectral_rolloff,
        'spectral_bandwidth': spectral_bandwidth,
        'mfcc_mean': mfccs.mean(axis=1).tolist(),
        'chroma_stft_mean': chroma_stft.mean(axis=1).tolist()
    }

    return results

# Example usage
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <path_to_wav_file>")
        # Example usage:
        wav_file_path = 'filepath.wav'  # Replace with your actual path

    else:
        wav_file_path = sys.argv[1]
    
    audio_info = analyze_audio(wav_file_path)
    for k, v in audio_info.items():
        print(f"{k}: {v}")
