import soundfile # to read audio file
import numpy as np
import librosa # to extract speech features
import glob
import os
import pickle # to save model after training
from sklearn.model_selection import train_test_split # for splitting training and testing
from sklearn.neural_network import MLPClassifier # multi-layer perceptron model
from sklearn.metrics import accuracy_score # to measure how good we are
import scipy.sparse as sps

def extract_feature(file_name, **kwargs):
    """
    Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
        e.g:
        `features = extract_feature(path, mel=True, mfcc=True)`
    """
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
            # print("mfccs shape: ", mfccs.shape)
            # print("result shape: ", result.shape)
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, chroma))
            # print("chroma shape: ", chroma.shape)
            # print("result shape: ", result.shape)
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
            result = np.hstack((result, mel))
            # print("mel shape: ", mel.shape)
            # print("result shape: ", result.shape)
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, contrast))
            # print("contrast shape: ", contrast.shape)
            # print("result shape: ", result.shape)
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
            result = np.hstack((result, tonnetz))
            # print("tonnetz shape: ", tonnetz.shape)
            # print("result shape: ", result.shape)
        # print("result shape: ", result.shape)
    return result

# all emotions on RAVDESS dataset
int2emotion = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# we allow only these emotions ( feel free to tune this on your need )
AVAILABLE_EMOTIONS = {
    "angry",
    "sad",
    "neutral",
    "happy"
}

def load_data(test_size=0.2):
    X, y = [], []
    for file in glob.glob("data/Actor_*/*.wav"):
        # get the base name of the audio file
        basename = os.path.basename(file)
        # get the emotion label
        emotion = int2emotion[basename.split("-")[2]]
        # we allow only AVAILABLE_EMOTIONS we set
        if emotion not in AVAILABLE_EMOTIONS:
            continue
        # extract speech features
        try:
            features = extract_feature(file, mfcc=True, chroma=True, mel=True)
            # add to dataset
            X.append(features)
            y.append(emotion)
        except ValueError as e:
            print(f"ValueError, failed to compute MFCC for {file}: {e}")
    # split the data to training and testing and return it
    return train_test_split(np.array(X), y, test_size=test_size, random_state=7)

if __name__ == "__main__":
    # load RAVDESS dataset, 75% training 25% testing
    try:
        X_train, X_test, y_train, y_test = pickle.load(open("output/X_data.pkl", "rb"))
    except FileNotFoundError:
        print("[+] Loading data for the first time...")
        X_train, X_test, y_train, y_test = load_data(test_size=0.25)

        # save the data to use it in other scripts
        with open('output/X_data.pkl', 'wb') as f:
            pickle.dump((X_train, X_test, y_train, y_test), f)

    # number of samples in training data
    print("[+] Number of training samples:", X_train.shape[0])
    # number of samples in testing data
    print("[+] Number of testing samples:", X_test.shape[0])
    # number of features used
    # this is a vector of features extracted 
    # using extract_features() function
    print("[+] Number of features:", X_train.shape[1])

    # best model, determined by a grid search
    model_params = {
        'alpha': 0.01,
        'batch_size': 256,
        'epsilon': 1e-08, 
        'hidden_layer_sizes': (200,), 
        'learning_rate': 'adaptive', 
        'max_iter': 5000, 
    }

    # initialize Multi Layer Perceptron classifier
    # with best parameters ( so far )
    model = MLPClassifier(**model_params)
    # train the model
    print("[*] Training the model...")
    model.fit(X_train, y_train)

    # predict 25% of data to measure how good we are
    y_pred = model.predict(X_test)

    # calculate the accuracy
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

    print("Accuracy: {:.2f}%".format(accuracy*100))

    # now we save the model
    # make result directory if doesn't exist yet
    if not os.path.isdir("output"):
        os.mkdir("output")

    pickle.dump(model, open("output/mlp_classifier.model", "wb"))