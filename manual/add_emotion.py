import librosa
import soundfile as sf
import numpy as np

def add_emotion(input_file, output_file, emotion="happy", intensity=1.0):
    y, sr = librosa.load(input_file)
    if emotion == "happy":
        y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=1*intensity)
        y_processed = librosa.effects.time_stretch(y_shifted, rate=1.0 + 0.1*intensity)
    elif emotion == "sad":
            y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=-1*intensity)
            y_speed = librosa.effects.time_stretch(y_shifted, rate=0.9 - 0.1*intensity)
            # y_reverb = librosa.effects.reverb(y_speed, sr=sr)
            y_processed = y_speed
    elif emotion == "angry":
        # y_processed = y * (1 + 0.5 * intensity)
        y_processed = librosa.effects.pitch_shift(y, sr=sr, n_steps=-1*intensity)
        y_processed = librosa.effects.time_stretch(y_processed, rate=1.1 + 0.1*intensity)
        # y_processed = np.clip(y_processed, -1, 1)
    elif emotion == "fearful":
        y_processed = librosa.effects.pitch_shift(y, sr=sr, n_steps=3 * intensity)
        # y_processed = librosa.effects.echo(y_shifted, sr=sr, delay=[0.2, 0.4], lengths=[0.1, 0.1])
    sf.write(output_file, y_processed, sr)

if __name__ == "__main__":
    add_emotion("testing.wav", "output_happy.wav", emotion="happy", intensity=0.6)
    add_emotion("testing.wav", "output_sad.wav", emotion="sad", intensity=0.5)
    add_emotion("testing.wav", "output_angry.wav", emotion="angry", intensity=0.6)
    add_emotion("testing.wav", "output_fearful.wav", emotion="fearful", intensity=0.5)