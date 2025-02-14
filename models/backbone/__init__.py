from typing import Dict

import torch.nn as nn

from .vggish import VGGish, VggishLayers
from .my_ast import ASTBackbone
from .resnet2d import ResNet2d
from .densenet import DenseNet2d


ALL_BACKBONES: Dict[str, nn.Module] = dict(
    vggish=VGGish,
    vggish_layers=VggishLayers,
    ast=ASTBackbone,
    resnet2d=ResNet2d,
    densenet2d=DenseNet2d
)