import torch
import torchaudio

from cnn import CNNNetwork
from urbansound import UrbanSoundDataset
from train import AUDIO_DIR, ANNOTATIONS_FILE, SAMPLE_RATE, NUM_SAMPLES, create_data_loader

class_mapping = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music"
]


def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load back the model
    cnn = CNNNetwork()
    state_dict = torch.load("feedforwardnet.pth")
    cnn.load_state_dict(state_dict)

    # load urban sound dataset dataset
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = UrbanSoundDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)

    test_dataloader = create_data_loader(usd, 1)

    # get a sample from the urban sound dataset for inference
    input, target = usd[0][0], usd[0][1] # [batch size, num_channels, fr, time]
    input.unsqueeze_(0)

    # make an inference
    predicted, expected = predict(cnn, input, target,
                                  class_mapping)
    print(f"Predicted: '{predicted}', expected: '{expected}'")

    correct, total = 0, 0
    for input, target in test_dataloader:
        input, target = input.to(device), target.to(device)

        predicted, expected = predict(cnn, input, target, class_mapping)
        # print(f"Predicted: '{predicted}', expected: '{expected}'")

        if predicted == expected:
            correct += 1
        total += 1
        if correct % 100 == 0:
            print("Accuracy: ", correct/total)
