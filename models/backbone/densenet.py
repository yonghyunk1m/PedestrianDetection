import torch
import torch.nn as nn
import timm


class DenseNet2d(nn.Module):
    def __init__(self, model_name="densenet121", pretrained=True) -> None:
        super().__init__()
        
        self.output_channel = 1024
        self.model = timm.create_model(model_name, pretrained=pretrained, in_chans=1, num_classes=0, global_pool="")
    
    def forward(self, x):
        x = x.unsqueeze(dim=1)  # N, 1, F, T
        x = self.model(x)
        x = torch.mean(x, dim=2)  # N, C, T
        return x

if __name__ == "__main__":
    model = DenseNet2d(model_name="densenet121", pretrained=True)
    print(model)

    x = torch.randn(4, 128, 300)
    print(model(x).shape)