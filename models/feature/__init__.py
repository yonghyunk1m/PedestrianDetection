from typing import Dict

import torch.nn as nn

from .vggish_mel import VGGishMel
from .mel_spec import MelSpectrogram

ALL_FEATURES: Dict[str, nn.Module] = dict(
    vggish_mel=VGGishMel,
    mel=MelSpectrogram
)
