import torch
from torch import nn




class Gamma_net(nn.Module):
    def __init__(self,inputbands):
        super(Gamma_net, self).__init__()
        self.net = nn.Sequential(nn.Linear(inputbands, 32),
                                 nn.Sigmoid(),
                                 nn.Linear(32, 32),
                                 nn.Sigmoid(),
                                 nn.Linear(32, inputbands),)

    def forward(self, ms):
        out = self.net(ms.permute(0, 2, 3, 1))
        return out.permute(0, 3, 1, 2)
