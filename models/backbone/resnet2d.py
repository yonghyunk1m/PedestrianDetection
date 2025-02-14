import torch
import torch.nn as nn
import timm


class ResNet2d(nn.Module):
    def __init__(self, resnet_name="resnet18d", pretrained=True) -> None:
        super().__init__()
        
        self.output_channel = 512
        self.model = timm.create_model(resnet_name, pretrained=pretrained, in_chans=1, num_classes=0, global_pool="")
    
    def forward(self, x):
        x = x.unsqueeze(dim=1)  # N, 1, F, T
        x = self.model(x)
        x = torch.mean(x, dim=2)  # N, C, T
        return x

if __name__ == "__main__":
    model = ResNet2d(resnet_name="resnet18d", pretrained=True)
    print(model)

    x = torch.randn(4, 128, 300)
    print(model(x).shape)