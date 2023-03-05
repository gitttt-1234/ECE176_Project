import torch
from torch import nn
from torch.nn import functional as F
from .types_ import *

class Actor(nn.Module):


    def __init__(self,
                 latent_dim: int,
                 hidden_dim: int,
                 layer_num: int = 4,
                 num_label: int = 10,
                 **kwargs) -> None:
        super(Actor, self).__init__()

        self.layer_num = layer_num
        self.num_label = num_label
        modules = []
        modules.append(nn.Sequential(
                    nn.Linear(latent_dim + hidden_dim, hidden_dim),
                    nn.LeakyReLU()))
        
        for i in range(self.layer_num - 1):
            modules.append(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LeakyReLU()))
        self.actor = nn.Sequential(*modules)
        self.actor_gate = nn.Linear(hidden_dim, latent_dim)
        self.actor_dz = nn.Linear(hidden_dim, latent_dim)
        self.actor_cond_layer = nn.Linear(num_label, hidden_dim)
    
    def forward(self, input: Tensor, label: Tensor):
        """
        Acts on the Latent
        :param input: (Tensor) Input tensor to encoder [N x D]
        :return: (Tensor) List of latent codes
        """
        x = self.actor_cond_layer(label)
        x = torch.cat((x, input), dim=-1)
        x = self.actor(x)
        ig = F.sigmoid(self.actor_gate(x))
        dz = self.actor_dz(x)
        z_dash = (1 - ig) * input + ig * dz
        return z_dash