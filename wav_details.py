#!/usr/bin/python

import wave
import sys

def get_wav_info(file_path):
    """
    Opens a WAV file and prints its key properties.

    Args:
        file_path (str): The path to the WAV file.
    """
    try:
        with wave.open(file_path, 'rb') as wav_file:
            print("Compression name:", wav_file.getcomptype())
            print(f"File: {file_path}")
            print(f"Number of channels: {wav_file.getnchannels()}")
            print(f"Sample width (bytes): {wav_file.getsampwidth()}")
            print(f"Frame rate (samples/sec): {wav_file.getframerate()}")
            print(f"Number of frames: {wav_file.getnframes()}")
            
            # Calculate duration in seconds
            duration_seconds = wav_file.getnframes() / wav_file.getframerate()
            print(f"Duration (seconds): {duration_seconds:.2f}")

    except wave.Error as e:
        print(f"Error opening or reading WAV file: {e}")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <path_to_wav_file>")
        # Example usage:
        get_wav_info("your_audio_file.wav") 
    else:
        wav_file_path = sys.argv[1]
        get_wav_info(wav_file_path)