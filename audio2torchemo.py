import os
import glob
import torch
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn

class LJSpeechMelDataset(Dataset):
    def __init__(self, root_dir, sample_rate=22050, n_mels=80, win_length=1024, hop_length=256, duration=5.0):
        self.paths = sorted(glob.glob(os.path.join(root_dir, 'wavs', '*.wav')))
        self.sr = sample_rate
        self.n_mels = n_mels
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = win_length
        self.duration = duration
        self.num_samples = int(sample_rate * duration)

        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            power=2.0,
            normalized=True,
        )

        # self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        waveform, sr = torchaudio.load(path)
        if waveform.shape[1] > self.num_samples:
            waveform = waveform[:, :self.num_samples]
        elif waveform.shape[1] < self.num_samples:
            waveform = torch.nn.functional.pad(waveform, (0, self.num_samples - waveform.shape[1]))

        if sr != self.sr:
            resampler = torchaudio.transforms.Resample(sr, self.sr)
            waveform = resampler(waveform)
        mel = self.mel_spec(waveform)  # (1, n_mels, time)
        # mel_db = self.amplitude_to_db(mel)  # Log scale
        # mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)
        mel_db = mel.squeeze(0).transpose(0, 1)  # (T, F)
        return mel_db, mel_db  # input = target

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
        x2, _ = self.attn(x, x, x)
        x = self.norm1(x + x2)
        x2 = self.ff(x)
        x = self.norm2(x + x2)
        return x

class AudioConvBlock(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, in_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):  # (B, C, F, T)
        return self.conv(x)

class MelFeatureAutoencoder(nn.Module):
    def __init__(self, dim=80):
        super().__init__()
        # self.input_proj = nn.Linear(dim, dim)
        self.transformer1 = AudioTransformerBlock(dim, 4, 256)
        self.conv_block = AudioConvBlock()
        self.transformer2 = AudioTransformerBlock(dim, 4, 256)
        self.conv_block2 = AudioConvBlock()

        # self.output_proj = nn.Linear(dim, dim)

    def forward(self, x):  # (B, T, F)
        # x = self.input_proj(x)
        x = self.transformer1(x)

        x_conv = x.transpose(1, 2).unsqueeze(1)  # (B, 1, F, T)
        x_conv = self.conv_block(x_conv)
        x_conv = self.conv_block2(x_conv)

        x = x_conv.squeeze(1).transpose(1, 2)  # (B, T, F)

        x = self.transformer2(x)
        return x

def mel_to_waveform(mel_db, sr=22050, n_fft=1024, win_length=1024, hop_length=256, n_mels=80):
    # [8, 431, 80]
    # mel_db = mel_db.unsqueeze(0)  # (1, T, F)
    mel_db = mel_db.transpose(0,1)  # (1, F, T)
    print(mel_db.shape)

    # db_to_amp = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80).inverse
    # mel_to_spec = torchaudio.transforms.MelScale(
    #     n_stft=n_fft // 2 + 1,
    #     n_mels=n_mels,
    #     sample_rate=sr,
    # ).inverse

    # mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=sr)

    power_spec = torchaudio.transforms.InverseMelScale(n_stft=n_fft//2+1, n_mels=n_mels, sample_rate=sr)
    power_val = power_spec(mel_db)
    
    griffin_lim = torchaudio.transforms.GriffinLim(n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    waveform = griffin_lim(power_val)
    return waveform

if __name__ == "__main__":
    dataset = LJSpeechMelDataset("./data/LJSpeech-1.1")
    loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MelFeatureAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    if os.path.exists("a2atorch.pth"):
        model.load_state_dict(torch.load("a2atorch.pth"))

    for epoch in range(10):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(loader):.4f}")
        torch.save(model.state_dict(), "a2atorch.pth")

    ## evaluation
    model.eval()
    # x, _ = dataset[0]
    # x = x.unsqueeze(0).to(device)

    # with torch.no_grad():
    #     out = model(x).squeeze(0).cpu()

    for x, y in loader:
        with torch.no_grad():
            out = model(x).squeeze(0).cpu()
        waveform = mel_to_waveform(out[0])
        torchaudio.save("output.wav", waveform.unsqueeze(0), 22050)
        exit()