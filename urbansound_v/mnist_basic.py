import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import os

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
class_mapping = list(str(i) for i in range(10))

class FeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.denseLayers = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.denseLayers(x)
        return self.softmax( x )

def download_MNST():
    train_data = datasets.MNIST(
        root="data",
        download=True,
        train=True,
        transform=ToTensor()
    )
    validation_data = datasets.MNIST(
        root="data",
        download=True,
        train=False,
        transform=ToTensor()
    )
    return train_data, validation_data

def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    for inputs, targets in train_data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # calculate loss
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)
        # backpropagate loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"loss: {loss.item()}")

def train(model, data_loader, loss_fn, optimizer, device, num_epochs):
    model.train()
    for i in range(num_epochs):
        print(f"Epoch: {i+1}")
        train_one_epoch(model, data_loader, loss_fn, optimizer, device)
        print(f"-----------------------")

    print("Training Complete")

def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        preds = model(input)
        # tensor (1, 10, )
        pred_index = preds[0].argmax(0)
        pred = class_mapping[pred_index]
        expected = class_mapping[target]
    return pred, expected

if __name__ == "__main__":
    train_data, validation_data = download_MNST()
    print(len(train_data))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = FeedForwardNet().to(device)

    if not os.path.exists("feedforwardnet.pth"):
        train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE)

        # Instantiate loss function
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE )

        # Train model
        train(model, train_data_loader, loss_fn, optimizer, device, num_epochs=EPOCHS)

        torch.save(model.state_dict(), "feedforwardnet.pth")

        print("Model trained")
    else:
        model.load_state_dict(torch.load("feedforwardnet.pth"))
        # torch.load("feedforwardnet.pth")
    # get a sample for infernence
    input, target = validation_data[0][0], validation_data[0][1]

    predicted, expected = predict(model, input, target, class_mapping)

    print(f"Predicted: {predicted}, Expected: {expected}")