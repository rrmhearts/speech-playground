# Ryan McCoppin
# Goal: Convert RAVDESS dataset to Emotional Speech Dataset format

import os
import shutil
from pathlib import Path

# Paths
ravdess_path = Path("./data/RAVDESS")  # Input RAVDESS directory
output_path = Path("./data/RAVDESS_ESD")       # Output directory for ESD
emotion_offsets = {'Neutral': 0, 'Angry': 350, 'Happy': 700, 'Sad': 1050, 'Surprise': 1400}

# Emotion map from RAVDESS codes to ESD names
emotion_map = {
    "01": "Neutral",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "08": "Surprise"
}

# Create output directories
def create_esd_dirs(speaker_id):
    speaker_dir = output_path / speaker_id
    for emotion in emotion_map.values():
        emotion_dir = speaker_dir / emotion
        emotion_dir.mkdir(parents=True, exist_ok=True)

# Convert files
def convert_ravdess_to_esd():
    for wav_file in ravdess_path.rglob("*.wav"):
        parts = wav_file.stem.split('-')
        if len(parts) != 7:
            continue

        emotion_code = parts[2]
        actor_id = int(parts[6])
        statement = "Kids are talking by the door." if parts[4] == "01" else \
                    "Dogs are sitting by the door."

        esd_offset = 20
        speaker_id = f"{actor_id+esd_offset:04d}"

        if emotion_code not in emotion_map:
            continue  # Skip unused emotions

        emotion_name = emotion_map[emotion_code]
        create_esd_dirs(speaker_id)

        # Create new filename
        if parts[5] == "01": # only one of the two repetitions
            # Modality: {parts[0]}, Vocal Channel (speech, song): {parts[1]}, Statement: {parts[4]}
            # statement, intensity, # Emotion (4, 3, 2)
            identifier_num = (int(parts[4])-1) * 2 + (int(parts[3])-1) # emotion is dir wise

            # identifier = f"{parts[2]}{parts[3]}{parts[4]}" # Repetition: {parts[5]}"
            identifier = f"{identifier_num + emotion_offsets[emotion_name]:06d}"

            new_filename = f"{speaker_id}_{identifier}.wav"
            no_extension, _ = os.path.splitext(new_filename)

            dest_file = output_path / speaker_id / emotion_name / new_filename
            shutil.copy2(wav_file, dest_file)

            with open(output_path / speaker_id / f"{speaker_id}.txt", "a") as file:
                # Write the new content to the file
                file.write(f"{no_extension}\t{statement}\t{emotion_name}\n")
            with open(output_path / "filelist.txt", "a") as file:
                # Write the new content to the file
                file.write(f"{Path(speaker_id) / emotion_name / new_filename}|{statement}|{emotion_name}\n".replace("\\", "/"))

            print(f"Copied {wav_file.name} -> {dest_file}")

if __name__ == "__main__":
    convert_ravdess_to_esd()
