"""
VGGish

Adapted from https://github.com/harritaylor/torchvggish/tree/master
"""
'''
This code implements an audio embedding model based on the VGGish model.
Here, the VGGish model is implemented in various variations to take audio data as input,
extract features, and learn temporal relationships.
'''
from collections import OrderedDict

import torch
import torch.nn as nn


VGGISH_WEIGHTS = "https://github.com/harritaylor/torchvggish/" \
                 "releases/download/v0.1/vggish-10086976.pth"


# Defines the basic structure of the VGG network, where the 'features' module extracts features.
class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        # self.embeddings = nn.Sequential(
        #     nn.Linear(512 * 4 * 6, 4096),
        #     nn.ReLU(True),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(True),
        #     nn.Linear(4096, 128),
        #     nn.ReLU(True),
        # )

    def forward(self, x):
        x = self.features(x)
        # # Transpose the output from features to
        # # remain compatible with vggish embeddings
        # x = torch.transpose(x, 1, 3)
        # x = torch.transpose(x, 1, 2)
        # x = x.contiguous()
        # x = x.view(x.size(0), -1)
        # x = self.embeddings(x)
        return x

# Constructs a standard VGG structure. Each layer consists of a convolutional layer, ReLU activation function, and max pooling layer.
def make_layers():
    layers = []
    in_channels = 1
    for v in [64, "M", 128, "M", 256, 256, "M", 512, 512, "M"]:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

# Used when the per_second flag is true; adjusts downsampling along the time axis by using (2, 5) max pooling sizes in some layers.
def make_layer_1s():
    layers = []
    in_channels = 1
    for v in [64, "M", 128, "M", 256, 256, "M5", 512, 512, "M5"]:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == "M5":
            layers += [nn.MaxPool2d(kernel_size=(2, 5), stride=(2, 5))]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

# A class that uses a pre-trained VGGish model.
class VGGish(nn.Module):
    def __init__(self, pretrained=True, freeze=False, per_second=False) -> None:
        super().__init__()
        
        self.output_channel = 512
        self.model = VGG(make_layers()) if not per_second else VGG(make_layer_1s())
        if pretrained: # If pretrained is true, loads weights via load_pretrained_model method.
            self.load_pretrained_model()
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
    
    def forward(self, x): # Passes input data through convolutional layers, then takes the mean along the time axis and returns the final output in (N, C, T) shape.
        x = x.unsqueeze(dim=1)  # N, 1, F, T
        x = self.model(x)
        x = torch.mean(x, dim=2)  # N, C, T
        return x

    def load_pretrained_model(self):
        state_dict = torch.hub.load_state_dict_from_url(VGGISH_WEIGHTS, progress=True)
        
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith("features"):
                new_state_dict[key] = value

        self.model.load_state_dict(new_state_dict)

# Inherits from VGGish and adds skip connections to utilize multi-level features.
class VggishLayers(VGGish):
    def __init__(self, pretrained=True, freeze=False, per_second=False) -> None:
        super().__init__(pretrained, freeze, per_second)

        self.output_1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(32, 1), groups=64),
            nn.GELU(),
            nn.Conv2d(64, 512, 1)
        )

        self.output_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(16, 1), groups=128),
            nn.GELU(),
            nn.Conv2d(128, 512, 1)
        )

        self.output_3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(8, 1), groups=256),
            nn.GELU(),
            nn.Conv2d(256, 512, 1)
        )

        self.output_4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(4, 1), groups=512),
            nn.GELU(),
            nn.Conv2d(512, 512, 1)
        )

        self.skip_connections = nn.ModuleList([
            self.output_1, self.output_2, self.output_3, self.output_4])

    # forward method: Sequentially passes input data through each layer in self.model.features, extracts features at each MaxPool2d layer using skip_connections.
    # These features are averaged along the time axis and summed at the end to obtain the final output.
    def forward(self, x):
        outputs = []
        x = x.unsqueeze(dim=1)  # N, 1, F, T
        output_idx = 0
        for layer in self.model.features:
            x = layer(x)
            if isinstance(layer, nn.MaxPool2d):
                outputs.append(self.skip_connections[output_idx](x).mean(dim=-1))
                output_idx += 1
        return sum(outputs)


if __name__ == "__main__":
    model = VggishLayers(pretrained=True, per_second=True)
    print(model)

    x = torch.randn(4, 64, 1000)
    result = model(x)
    print(result.shape) # [4, 512, 1]
    print(f"result[0][0]: {result[0][0]}")