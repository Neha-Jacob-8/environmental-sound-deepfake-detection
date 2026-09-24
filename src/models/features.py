"""Log-Mel front-end shared by the CNN model (and available to any other
model that wants a spectrogram view of the same raw waveform input).

64 mels x 251 frames for a 4.0 s / 16 kHz / 64,000-sample clip, matching the
"Feature extraction" box in the methodology diagram:
    STFT -> Mel filterbank -> log, n_fft=1024, hop=256, 64 mels
    (64000 / 256 = 250 hops -> 251 frames with center=True padding)
"""

import torch
import torch.nn as nn
import torchaudio


class LogMelExtractor(nn.Module):
    def __init__(self, sample_rate=16_000, n_fft=1024, hop_length=256, n_mels=64):
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)

    def forward(self, waveform):
        """waveform: (B, 1, T) -> (B, 1, n_mels, n_frames), per-clip standardized."""
        mel = self.melspec(waveform)  # (B, 1, n_mels, n_frames)
        log_mel = self.to_db(mel)
        # Per-clip standardization so clips at different loudness/generators
        # land on a comparable scale for the CNN.
        mean = log_mel.mean(dim=(-2, -1), keepdim=True)
        std = log_mel.std(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
        return (log_mel - mean) / std
