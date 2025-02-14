import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "ast"))

import torch
import torch.nn as nn

from src.models import ASTModel
os.environ['TORCH_HOME'] = os.path.join(os.path.dirname(__file__), 'ast/pretrained_models')

class ASTBackbone(nn.Module):
    def __init__(self, input_tdim=100, input_fdim=128, imagenet_pretrain=True, audioset_pretrain=False, freeze=False) -> None:
        super().__init__()

        self.output_channel = 768
        self.ast = ASTModel(
            label_dim=527, input_tdim=input_tdim, input_fdim=input_fdim, imagenet_pretrain=imagenet_pretrain, audioset_pretrain=audioset_pretrain)
        self.ast.mlp_head = nn.Identity()

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = x.transpose(-1, -2)
        x = self.ast(x)
        return x.unsqueeze(dim=-1)

if __name__ == "__main__":
    input_tdim = 100
    label_dim = 2
    input_fdim = 128

    test_input = torch.rand([10, input_fdim, input_tdim]).cuda()

    ast_mdl = ASTBackbone(input_fdim=input_fdim, imagenet_pretrain=True, audioset_pretrain=True).cuda()
    test_output = ast_mdl(test_input)
    print(test_output.shape)