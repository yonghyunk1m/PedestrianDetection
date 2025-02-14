"""
VGGish

Adapted from https://github.com/harritaylor/torchvggish/tree/master
"""
'''
이 코드는 VGGish 모델을 기반으로 한 오디오 임베딩 모델을 구현한다.
여기서 VGGish 모델을 다양한 변형으로 구현하여 오디오 데이터를 입력으로 받아 특징을 추출하고,
이를 바탕으로 시간적 관계를 학습할 수 있도록 설계한다.
'''
from collections import OrderedDict

import torch
import torch.nn as nn


VGGISH_WEIGHTS = "https://github.com/harritaylor/torchvggish/" \
                 "releases/download/v0.1/vggish-10086976.pth"


# VGG네트워크의 기본 구조를 정의하며, features 네트워크르 모듈을 사용하여 특징을 추출한다.
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

# 표준 VGG 구조를 만든다. 각 레이어는 Convolution 레이어와 ReLU 활성화 함수, 그리고 Max Pooling 레이어로 구성된다.
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

# per_second 플래그가 참일때 사용된다; 일부 Max Pooling 레이어에서 (2, 5) 크기를 사용하여 시간 축에서 다운샘플링을 조정한다.
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

# 사전학습된 VGGish 모델을 사용하는 클래스이다!
class VGGish(nn.Module):
    def __init__(self, pretrained=True, freeze=False, per_second=False) -> None:
        super().__init__()
        
        self.output_channel = 512
        self.model = VGG(make_layers()) if not per_second else VGG(make_layer_1s())
        if pretrained: # pretrained가 참이면 load_pretrained_model 메서드를 통해 가중치를 로드한다.
            self.load_pretrained_model()
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
    
    def forward(self, x): # 입력 데이터를 Convolution 레이어를 거친 후, 시간 축에서 평균을 취해 최종 출력 (N, C, T 형태)을 반환한다.
        x = x.unsqueeze(dim=1)  # N, 1, F, T
        x = self.model(x)
        #print(f"x.shape: {x.shape}")
        x = torch.mean(x, dim=2)  # N, C, T
        return x

    def load_pretrained_model(self):
        state_dict = torch.hub.load_state_dict_from_url(VGGISH_WEIGHTS, progress=True)
        
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith("features"):
                new_state_dict[key] = value

        self.model.load_state_dict(new_state_dict)

# VGGish를 상속받아 Skip Connection을 추가하여 다중 레벨 특징을 사용한다.
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

    # forward 메서드: 입력 데이터를 self.model.features의 각 레이어를 순차적으로 통과하면서, MaxPool2d 레이어에서 skip_connections을 사용해 각 단계의 특징을 추출한다.
    # 이 특징들은 시간 축에서 평균이 취해지고, 마지막에 모두 더해져 최종 출력이 된다.
    def forward(self, x):
        outputs = []
        x = x.unsqueeze(dim=1)  # N, 1, F, T
        output_idx = 0
        #print(f"x.shape: {x.shape}")
        for layer in self.model.features:
            x = layer(x)
            #print(f"x.shape: {x.shape}")
            if isinstance(layer, nn.MaxPool2d):
                #breakpoint()
                outputs.append(self.skip_connections[output_idx](x).mean(dim=-1))
                output_idx += 1
        # breakpoint()
        # print(f"len(outputs): {len(outputs)}") # 4
        # print(f"outputs[0].shape: {outputs[0].shape}") # torch.Size([4, 512, 1])
        # print(f"sum(outputs).shape: {sum(outputs).shape}")
        # print(f"sum(outputs[0][0][]): {sum(outputs[0][0])}")
        return sum(outputs)


if __name__ == "__main__":
    model = VggishLayers(pretrained=True, per_second=True)
    print(model)

    x = torch.randn(4, 64, 1000)
    result = model(x)
    print(result.shape) # [4, 512, 1]
    print(f"result[0][0]: {result[0][0]}")