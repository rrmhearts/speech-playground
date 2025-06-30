import soundfile as sf
import librosa
import numpy as np

# Create some dummy audio data (e.g., a sine wave)
samplerate = 8000  # Mu-law is typically used with 8kHz
duration = 1  # seconds
frequency = 440  # Hz
t = np.linspace(0, duration, int(samplerate * duration), endpoint=False)
data = 0.5 * np.sin(2 * np.pi * frequency * t)

data, samplerate = sf.read('vs_000.wav')

if samplerate != 8000:
    data = librosa.resample(data, samplerate, 8000)
# Define the output filename
output_filename = 'mulaw_audio.wav'

# Write the data to a WAV file with mu-law encoding
# 'WAV' is the format, and 'ULAW' is the subtype for mu-law encoding
sf.write(output_filename, data, samplerate, format='WAV', subtype='ULAW')

print(f"Mu-law encoded audio saved to {output_filename}")