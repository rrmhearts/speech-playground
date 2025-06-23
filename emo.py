import os
import torch
import torchaudio
from torch.utils.data import Dataset
import glob

class EmotionSpeechDataset(Dataset):
    def __init__(self, data_root, emotions, sample_rate=22050, n_mels=80, duration=2.0, win_length=1024, hop_length=256):
        self.data_root = data_root
        self.emotions = emotions
        self.sr = sample_rate
        self.n_fft = win_length
        self.hop_length = hop_length
        self.duration = duration
        self.max_samples = int(sample_rate * duration)
        self.n_mels = n_mels

        self.pairs = []
        for emotion in emotions:
            neutral_files = glob.glob(os.path.join(data_root, 'neutral', '*.wav'))
            for neutral_path in neutral_files:
                basename = os.path.basename(neutral_path)
                target_path = os.path.join(data_root, emotion, basename)
                if os.path.exists(target_path):
                    self.pairs.append((neutral_path, target_path, emotion))

        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=win_length,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            normalized=True
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()

        self.emotion_to_id = {e: i for i, e in enumerate(emotions)}

    def __len__(self):
        return len(self.pairs)

    def _process(self, wav_path):
        waveform, _ = torchaudio.load(wav_path)
        if waveform.shape[1] > self.max_samples:
            waveform = waveform[:, :self.max_samples]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, self.max_samples - waveform.shape[1]))
        mel = self.db_transform(self.mel_spec(waveform))
        mel = (mel - mel.mean()) / (mel.std() + 1e-9)
        return mel.squeeze(0).transpose(0, 1)  # (T, F)

    def __getitem__(self, idx):
        neutral_path, target_path, emotion = self.pairs[idx]
        x = self._process(neutral_path)
        y = self._process(target_path)
        e_id = self.emotion_to_id[emotion]
        return x, y, e_id
import torch.nn as nn

class SpeakerEncoder(nn.Module):
    def __init__(self, mel_dim=80, hidden_dim=256, emb_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(mel_dim, hidden_dim, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(hidden_dim, emb_dim)

    def forward(self, mel):  # (B, T, F)
        x = mel.transpose(1, 2)  # (B, F, T)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)  # (B, H)
        emb = self.proj(x)            # (B, emb_dim)
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-9)  # normalize
        return emb

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

class EmotionMelModel(nn.Module):
    def __init__(self, mel_dim=80, emotion_dim=8, num_emotions=4):
        super().__init__()
        self.emotion_emb = nn.Embedding(num_emotions, emotion_dim)
        self.input_proj = nn.Linear(mel_dim + emotion_dim, mel_dim)

        self.trans1 = AudioTransformerBlock(mel_dim, heads=4, ff_dim=256)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.ReLU()
        )
        self.trans2 = AudioTransformerBlock(mel_dim, heads=4, ff_dim=256)
        self.output_proj = nn.Linear(mel_dim, mel_dim)

    def forward(self, x, emotion_id):
        B, T, F = x.shape
        e = self.emotion_emb(emotion_id).unsqueeze(1).expand(-1, T, -1)  # (B, T, E)
        x = torch.cat([x, e], dim=-1)
        x = self.input_proj(x)
        x = self.trans1(x)

        x_conv = x.transpose(1, 2).unsqueeze(1)  # (B, 1, F, T)
        x_conv = self.conv(x_conv)
        x = x_conv.squeeze(1).transpose(1, 2)

        x = self.trans2(x)
        return self.output_proj(x)
    
def mel_to_waveform(mel_db, sr=22050, n_fft=1024, hop_length=256, win_length=1024, n_mels=80):
    mel_db = mel_db.unsqueeze(0).transpose(1, 2)  # (1, F, T)
    db_to_amp = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80).inverse
    mel_to_spec = torchaudio.transforms.MelScale(
        n_stft=n_fft // 2 + 1,
        n_mels=n_mels,
        sample_rate=sr
    ).inverse
    spec = mel_to_spec(db_to_amp(mel_db))
    waveform = torchaudio.transforms.GriffinLim(n_fft=n_fft, hop_length=hop_length, win_length=win_length)(spec)
    return waveform


from torch.utils.data import DataLoader

emotions = ['happy', 'sad', 'angry']
dataset = EmotionSpeechDataset('EmoSpeech', emotions)
loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionMelModel(num_emotions=len(emotions)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

speaker_encoder = SpeakerEncoder().to(device)
speaker_encoder.eval()  # freeze identity encoder

alpha = 1.0  # weight for identity loss

for epoch in range(10):
    model.train()
    total_loss = 0.0
    for x, y, e_id in loader:
        x, y, e_id = x.to(device), y.to(device), e_id.to(device)
        y_hat = model(x, e_id)

        # Reconstruction loss
        recon_loss = loss_fn(y_hat, y)

        # Voice identity loss (cosine embedding)
        with torch.no_grad():
            emb_x = speaker_encoder(x)
        # want the new voice to match original for this embedding
        emb_yhat = speaker_encoder(y_hat)        

        identity_loss = 1.0 - torch.nn.functional.cosine_similarity(emb_x, emb_yhat, dim=-1).mean()

        loss = recon_loss + alpha * identity_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Total Loss = {total_loss / len(loader):.4f}")
