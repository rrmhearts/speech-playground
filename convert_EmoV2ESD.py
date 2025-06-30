import os
import shutil
import re
from collections import defaultdict

# Paths
SRC_DIR = 'EmoV_DB'
DST_DIR = 'ESD'
SENTENCE_FILE = os.path.join(SRC_DIR, 'cmu_sentences.txt')

# Step 1: Parse cmu_sentences.txt to map IDs to text
def parse_cmu_sentences(path):
    id_to_text = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.match(r'\(\s*arctic_a0*(\d+)\s+"(.+?)"\s*\)', line)
            if match:
                file_id, text = match.groups()
                id_to_text[file_id.zfill(4)] = text
    return id_to_text

# Step 2: Map speakers to numeric IDs
def get_speaker_id_map(directories):
    speaker_names = sorted(set(dir_name.split('_')[0] for dir_name in directories))
    return {name: str(i+1).zfill(4) for i, name in enumerate(speaker_names)}

# Step 3: Process each file
def convert_dataset():
    os.makedirs(DST_DIR, exist_ok=True)
    dirs = [d for d in os.listdir(SRC_DIR) if os.path.isdir(os.path.join(SRC_DIR, d))]
    speaker_id_map = get_speaker_id_map(dirs)
    id_to_text = parse_cmu_sentences(SENTENCE_FILE)

    for dir_name in dirs:
        speaker_name, emotion = dir_name.split('_')
        speaker_id = speaker_id_map[speaker_name]
        src_emotion_dir = os.path.join(SRC_DIR, dir_name)
        dst_emotion_dir = os.path.join(DST_DIR, speaker_id, emotion)
        os.makedirs(dst_emotion_dir, exist_ok=True)

        for fname in os.listdir(src_emotion_dir):
            if not fname.endswith('.wav'):
                continue
            # Example filename: anger_1-28_0001.wav
            parts = fname.split('_')
            if len(parts) != 3:
                continue
            file_id = parts[2].replace('.wav', '')
            # Ensure ID has four digits to match cmu_sentences.txt
            file_id_padded = file_id.zfill(4)
            new_fname = f"{speaker_id}_{file_id_padded}.wav"
            shutil.copy(
                os.path.join(src_emotion_dir, fname),
                os.path.join(dst_emotion_dir, new_fname)
            )

    print(f"Conversion completed. Output at: {DST_DIR}")

if __name__ == '__main__':
    convert_dataset()
