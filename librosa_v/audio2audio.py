import os
import glob
import librosa
import torch
from torch.utils.data import Dataset
import numpy as np

class LJSpeechMelDataset(Dataset):
    def __init__(self, root_dir, sr=22050, n_mels=80, hop_length=256, win_length=1024, duration=2.0):
        self.file_paths = sorted(glob.glob(os.path.join(root_dir, "wavs", "*.wav")))
        self.sr = sr
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.win_length = win_length
        self.duration = duration

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        y, _ = librosa.load(path, sr=self.sr, duration=self.duration)
        y = librosa.util.fix_length(y, size=int(self.duration * self.sr))
        mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels,
                                             hop_length=self.hop_length, win_length=self.win_length)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)
        mel_tensor = torch.tensor(mel_db.T, dtype=torch.float32)  # shape: (T, 80)
        return mel_tensor, mel_tensor

import torch.nn as nn

class AudioTransformerBlock(nn.Module):
    def __init__(self, dim, heads, ff_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

class AudioConvBlock(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, in_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

class MelFeatureAutoencoder(nn.Module):
    def __init__(self, dim=80):
        super().__init__()
        self.input_proj = nn.Linear(dim, dim)
        self.trans1 = AudioTransformerBlock(dim, heads=4, ff_dim=256)
        self.conv = AudioConvBlock(in_channels=1)
        self.trans2 = AudioTransformerBlock(dim, heads=4, ff_dim=256)
        self.output_proj = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.trans1(x)

        x_conv = x.transpose(1, 2).unsqueeze(1)  # (B, 1, F, T)
        x_conv = self.conv(x_conv)
        x = x_conv.squeeze(1).transpose(1, 2)  # back to (B, T, F)

        x = self.trans2(x)
        x = self.output_proj(x)
        return x

def mel_to_waveform(mel_db, sr=22050, n_fft=1024, hop_length=256, n_iter=60):
    mel_db = mel_db.T  # (n_mels, time)
    mel_power = librosa.db_to_power(mel_db)
    inv_mel_filter = librosa.feature.inverse.mel_to_stft(mel_power, sr=sr, n_fft=n_fft)
    waveform = librosa.griffinlim(inv_mel_filter, hop_length=hop_length, n_iter=n_iter)
    return waveform


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = LJSpeechMelDataset("data/LJSpeech-1.1")
    loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MelFeatureAutoencoder().to(device)

    if os.path.exists("a2a.pth"):
        model.load_state_dict(torch.load("a2a.pth"))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    for epoch in range(10):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = loss_fn(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        torch.save(model.state_dict(), "a2a.pth")
        print(f"Epoch {epoch+1} | Loss: {total_loss / len(loader):.4f}")


    import soundfile as sf

    model.eval()
    x, _ = dataset[0]
    x = x.unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
    out = out.squeeze(0).cpu().numpy()

    # Reconstruct waveform
    wave = mel_to_waveform(out)
    sf.write("output.wav", wave, 22050)
