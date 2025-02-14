import torch
import torch.nn as nn
import torchaudio.transforms as T

class MelSpectrogram(nn.Module):
    def __init__(
        self,
        sample_rate,
        n_fft,
        win_length,
        hop_length,
        f_min,
        f_max,
        n_mels,

        norm_mean=0.0,
        norm_std=1.0
    ) -> None:
        super().__init__()

        self.spec_layer = T.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
            f_min=f_min, f_max=f_max, n_mels=n_mels
        )
        self.to_db = T.AmplitudeToDB()
        
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        
    # ORIGINAL
    # def forward(self, x):
    #     s = self.spec_layer(x)
    #     s = self.to_db(s)

    #     s = (s - self.norm_mean) / self.norm_std

    #     return s
    
    # MODIFIED
    def forward(self, x):
        # Ensure the MelSpectrogram's spectrogram components are on the same device as input
        if self.spec_layer.spectrogram.window is not None:
            self.spec_layer.spectrogram.window = self.spec_layer.spectrogram.window.to(x.device)

        # Ensure the MelScale filter bank is on the same device
        self.spec_layer.mel_scale.fb = self.spec_layer.mel_scale.fb.to(x.device)

        # Perform the Mel-spectrogram transformation
        s = self.spec_layer(x)
        s = self.to_db(s)

        # Normalize the spectrogram
        s = (s - self.norm_mean) / self.norm_std

        return s
