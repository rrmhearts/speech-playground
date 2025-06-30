import numpy as np
from scipy.io.wavfile import read, write
from pathlib import Path

def convert_float_wav_to_pcm(input_filepath: str, output_filepath=None):
    """
    Converts a float WAV file to a 16-bit PCM WAV file.

    Args:
        input_filepath (str): Path to the input float WAV file.
        output_filepath (str): Path to save the output 16-bit PCM WAV file.
    """

    if output_filepath is None:
        output_filepath = input_filepath.replace(".wav", "_pcm16.wav")
    try:
        samplerate, float_data = read(input_filepath)

        # Scale the float data to the range of int16
        # np.iinfo(np.int16).max gives the maximum value for int16 (32767)
        # Ensure the data is within -1 to 1 before scaling to prevent clipping
        print(f"min: {float_data.min()}, max: {float_data.max()}")
        float_data = np.clip(float_data, -1.0, 1.0)
        pcm_data = (float_data * np.iinfo(np.int16).max).astype(np.int16)

        write(output_filepath, samplerate, pcm_data)
        print(f"Successfully converted '{input_filepath}' to '{output_filepath}' (16-bit PCM).")

    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_filepath}'")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
# Assuming 'input_float_audio.wav' is a 32-bit float WAV file
# convert_float_wav_to_pcm('input_float_audio.wav', 'output_pcm_audio.wav')

def apply_function_to_files(root_directory, filetype, function, **kwargs):
    """
    Iterates through a directory and its subdirectories to find .wav files
    and applies a specified function to each found file.
    """
    root_path = Path(root_directory)
    # Use glob with "**/*.wav" to recursively find all .wav files
    for file_path in root_path.glob(f"**/*.{filetype}"):
        file_path_str = str(file_path)
        if filetype in file_path_str and file_path.is_file():  # Ensure it's a file and not a directory
            print(f"Processing file: {file_path}")
            function(file_path_str, **kwargs)

if __name__ == "__main__":
    # input = Path("data/EmoV_DB/bea_Amused/amused_1-15_0001.wav")
    # convert_float_wav_to_pcm(input, 'output_pcm_audio.wav')

    apply_function_to_files("data/EmoV_DB", "wav", convert_float_wav_to_pcm)    