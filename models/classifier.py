import torch
from torch import nn
from torch.nn import functional as F
from .types_ import *


class Classifier(nn.Module):


    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 num_outputs: int=10,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(Classifier, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512, 1024]
        self.hidden_dims = hidden_dims

        # Build Encoder
        for h_dim in hidden_dims[:2]:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        
        for h_dim in hidden_dims[2:]:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        self.fc = nn.Sequential(
                        nn.LeakyReLU(),
                        nn.Linear(hidden_dims[-1], num_outputs),
                    )
    
    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x