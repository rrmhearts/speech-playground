# speech-playground
Attempts at speech conversion 

* [without DL](./manual/)
* [training w/ librosa](./librosa_v/)
* [training w/ PyTorch](./audio2torchaudio.py)
* [classification problem](./urbansound_v/)
* [taco example from PyTorch](./taco_v/)
* [UNFINISHED emotion conversion](./emo.py)

Download RAVDESS dataset into `data/`. Copy directory files to top-level for processing. Current top level application requires LJSpeech.

The secret Microsoft doesn't want you to know about. **"Shortcuts"** are not supported by VSCode for some reason.
`New-Item -Path <new_symlink_name> -ItemType SymbolicLink -Value <source_of_symlink>`
```sh
# e.g.
New-Item -Path .\EmoV_DB -ItemType SymbolicLink -Value ..\..\Datasets\EmoV_DB\
```