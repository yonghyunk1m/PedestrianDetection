import torch
import torch.nn as nn
import torchaudio.transforms as T
import torchaudio.functional as F


TARGET_SR = 16000

N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160

F_MIN = 125
F_MAX = 7500
N_MELS = 64

EPS = 0.01


class VGGishMel(nn.Module):
    def __init__(self, sample_rate) -> None:
        super().__init__()

        self.target_sr = TARGET_SR
        
        self.n_fft = N_FFT
        self.win_length = WIN_LENGTH
        self.hop_length = HOP_LENGTH

        self.f_min = F_MIN
        self.f_max = F_MAX
        self.n_mels = N_MELS

        self.eps = EPS

        self.resample = T.Resample(sample_rate, self.target_sr)
        self.register_buffer("window", torch.hann_window(self.win_length))
        self.register_buffer("mel_fb", F.melscale_fbanks(n_freqs=self.n_fft // 2 + 1, f_min=self.f_min, f_max=self.f_max, n_mels=self.n_mels, sample_rate=self.target_sr, mel_scale= "htk"))

    def forward(self, x):
        x = self.resample(x)

        frames = x.unfold(dimension=-1, size=self.win_length, step=self.hop_length)
        windowed_frames = frames * self.window
        s = torch.abs(torch.fft.rfft(windowed_frames, self.n_fft))
        mel = torch.matmul(s, self.mel_fb)
        return torch.log(mel + self.eps).transpose(-1, -2)
