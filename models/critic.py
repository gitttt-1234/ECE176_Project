import torch
from torch import nn
from torch.nn import functional as F
from .types_ import *


class Critic(nn.Module):


    def __init__(self,
                #  vae: nn.Module,
                 latent_dim: int,
                 hidden_dim: int,
                 layer_num: int = 4,
                 num_label: int = 10,
                 num_output: int = 10,
                 **kwargs) -> None:
        super(Critic, self).__init__()
        self.layer_num = layer_num
        modules = []
        modules.append(nn.Sequential(
                    nn.Linear(latent_dim + hidden_dim, hidden_dim),
                    nn.LeakyReLU()))

        for i in range(self.layer_num - 1):
            modules.append(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LeakyReLU()))
        modules.append(nn.Sequential(
                    nn.Linear(hidden_dim, 1)))
        self.real_critic = nn.Sequential(*modules)
        self.real_critic_cond_layer = nn.Linear(num_label, hidden_dim)
    
    def forward(self, input: Tensor, label: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        x = self.real_critic_cond_layer(label)
        x = torch.cat((input, x), dim=-1)
        x = self.real_critic(x)
        return F.sigmoid(x)