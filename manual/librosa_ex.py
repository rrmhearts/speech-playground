import librosa
import numpy as np
import soundfile as sf
import wave
import sounddevice as sd

filename = "vowels.wav"
# Set the sampling frequency and duration of the recording
sampling_frequency = 44100
duration = 5  # in seconds

# Record audio
# print("Recording...")
# audio = sd.rec(int(sampling_frequency * duration), samplerate=sampling_frequency, channels=1)
# sd.wait()  # Wait until recording is finished
# print("Finished recording")

# # Save the recorded audio to a WAV file
# sf.write('voice.wav', audio, sampling_frequency)

# with wave.open(filename, 'rb') as wav_file:
#     channels_number, sample_width, framerate, frames_number, compression_type, compression_name = wav_file.getparams()
#     frames = wav_file.readframes(frames_number)
#     # frames datatype not supported by s
#     audio_signal = np.frombuffer(frames, dtype='<i2')

#     sd.play(audio_signal, framerate)
#     sd.wait()

# load the audio signal and its sample rate
signal, sample_rate = librosa.load(filename)

import librosa.display
import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 3))
# librosa.display.waveshow(signal, sr=sample_rate) # use waveplot should waveshow be unavailable
# plt.show()

# # Compute the mel-spectrogram
# mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sample_rate)

# # Plot the mel-spectrogram
# plt.figure(figsize=(10, 4))
# librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max), sr=sample_rate, hop_length=512, y_axis="mel", x_axis="time")
# plt.colorbar(format="%+2.0f dB")
# plt.title("Mel-spectrogram")
# plt.tight_layout()
# plt.show()

# # MFCC: extract MFCCs
# mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate)

# plt.figure(figsize=(10, 4))
# librosa.display.specshow(mfccs, sr=sample_rate, hop_length=512, y_axis="mel", x_axis="time")
# plt.colorbar(format="%+2.0f dB")
# plt.title("MFCCs")
# plt.tight_layout()
# plt.show()

# Same as MFCC above
# Compute the mel-spectrogram
# mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sample_rate)

# Compute the Mel-frequency cepstral coefficients (MFCCs)
# mfccs = librosa.feature.mfcc(S=librosa.power_to_db(mel_spectrogram), sr=sample_rate)

# # Compute the spectral contrast features
# spectral_contrast = librosa.feature.spectral_contrast(y=signal, n_fft=2048, hop_length=512)

# # Plot the spectral contrast features
# plt.figure(figsize=(10, 4))
# librosa.display.specshow(spectral_contrast, sr=sample_rate, hop_length=512, x_axis="time")
# plt.title("Spectral contrast features")
# plt.tight_layout()
# plt.show()

# # Compute the chroma features
# chroma_features = librosa.feature.chroma_stft(y=signal, sr=sample_rate)

# # Plot the chroma features
# plt.figure(figsize=(10, 4))
# librosa.display.specshow(chroma_features, sr=sample_rate, hop_length=512, x_axis="time", y_axis="chroma")
# plt.title("Chroma features")
# plt.tight_layout()
# plt.show()

# Compute pitch using the PEPLOs algorithm
f0, voiced_flag, voiced_probs = librosa.pyin(signal, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))

# Plot pitch contour
plt.figure(figsize=(12, 4))
librosa.display.waveshow(signal, sr=sample_rate, alpha=0.5)
plt.plot(librosa.frames_to_time(range(len(f0))), f0, color='r')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Pitch Contour')
plt.show()

# Shift the pitch down by two semitones
pitch_shifted_waveform = librosa.effects.pitch_shift(y=signal, sr=sample_rate, n_steps=-2.0)
# Normalize the output signal
pitch_shifted = librosa.util.normalize(pitch_shifted_waveform)

# Save the output signal to a WAV file
sf.write("voice_lower.wav", pitch_shifted, sample_rate)

# Stretch the time by a factor of 2
pitch_shifted_waveform = librosa.effects.pitch_shift(y=signal, sr=sample_rate, n_steps=3)
time_stretched_waveform = librosa.effects.time_stretch(pitch_shifted_waveform, rate=2)
sf.write("voice_high_slow.wav", time_stretched_waveform, sample_rate)

f0, voiced_flag, voiced_probs = librosa.pyin(pitch_shifted, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))

# Plot pitch contour
plt.figure(figsize=(12, 4))
librosa.display.waveshow(pitch_shifted, sr=sample_rate, alpha=0.5)
plt.plot(librosa.frames_to_time(range(len(f0))), f0, color='r')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Pitch SHIFTED Contour')
plt.show()