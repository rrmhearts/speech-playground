import numpy as np
import wave
import struct

def add_emotion_to_wav(input_wav, output_wav, emotion_type, intensity=1.0):
    """
    Adds emotion to a WAV file by manipulating its audio data.

    Args:
        input_wav (str): Path to the input WAV file.
        output_wav (str): Path to save the modified WAV file.
        emotion_type (str): Type of emotion to add ('happy', 'sad', 'angry').
        intensity (float): Intensity of the emotion effect (0.0 to 1.0).
    """
    try:
        with wave.open(input_wav, 'rb') as wf:
            num_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frame_rate = wf.getframerate()
            num_frames = wf.getnframes()
            audio_data = wf.readframes(num_frames)
    except wave.Error as e:
         raise Exception(f"Could not open or read WAV file: {e}")

    # Convert audio data to numerical array
    if sample_width == 2:
        audio_array = np.array(struct.unpack(f"<{num_frames * num_channels}h", audio_data))
    elif sample_width == 1:
         audio_array = np.array(struct.unpack(f"<{num_frames * num_channels}B", audio_data)) - 128
    else:
        raise ValueError("Unsupported sample width")


    # Apply emotion-based modification
    if emotion_type == 'happy':
        audio_array = audio_array * (1 + 0.2 * intensity)  # Increase volume/energy
    elif emotion_type == 'sad':
        audio_array = audio_array * (1 - 0.3 * intensity)  # Decrease volume/energy
    elif emotion_type == 'angry':
         audio_array = audio_array * (1 + 0.5 * intensity) # Increase volume/energy
    else:
        raise ValueError("Invalid emotion type")

    audio_array = np.clip(audio_array, -32768, 32767)
    modified_audio_data = struct.pack(f"<{len(audio_array)}h", *audio_array.astype(np.int16))


    try:
        with wave.open(output_wav, 'wb') as wf_new:
            wf_new.setnchannels(num_channels)
            wf_new.setsampwidth(sample_width)
            wf_new.setframerate(frame_rate)
            wf_new.writeframes(modified_audio_data)
    except wave.Error as e:
         raise Exception(f"Could not open or write WAV file: {e}")

if __name__ == '__main__':
    input_file = 'audio.wav'
    output_file = 'audio_with_emotion.wav'
    add_emotion_to_wav(input_file, output_file, 'happy', 0.7)