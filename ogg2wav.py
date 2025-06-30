from pydub import AudioSegment
import requests

def convert_ogg_to_alaw_wav(ogg_filepath, alaw_wav_filepath, codec="pcm_alaw"):
    try:
        # Load the OGG file
        audio = AudioSegment.from_ogg(ogg_filepath)

        # Export as A-law WAV
        audio.export(alaw_wav_filepath, format="wav", codec=codec)
        print(f"Successfully converted '{ogg_filepath}' to '{alaw_wav_filepath}' (A-law WAV).")
    except Exception as e:
        print(f"Error converting file: {e}")

if __name__ == "__main__":

    # Pull OGG off the internet
    url = "https://file-examples.com/storage/fe803e9596685d587a3e84a/2017/11/file_example_OOG_1MG.ogg"
    input_ogg = "file_example_OOG_1MG.ogg"  # Replace with your OGG file path
    output_alaw_wav = "file_example_OOG_1MG.wav" # Desired output WAV file path

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)

        with open(input_ogg, 'wb') as f:
            for chunk in response.iter_content(chunk_size=4096):
                f.write(chunk)
        print(f"File downloaded successfully to {input_ogg}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")

    # Convert file from OGG to a-law wav
    convert_ogg_to_alaw_wav(input_ogg, output_alaw_wav)