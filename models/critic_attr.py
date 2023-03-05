import torch
from torch import nn
from torch.nn import functional as F
from .types_ import *


class Critic_Attr(nn.Module):


    def __init__(self,
                 latent_dim: int,
                 hidden_dim: int,
                 num_output: int = 10,
                 **kwargs) -> None:
        super(Critic_Attr, self).__init__()
        modules = []
        modules.append(nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.LeakyReLU()))
        
        for i in range(self.layer_num  - 1):
            modules.append(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LeakyReLU()))
        modules.append(nn.Sequential(
                    nn.Linear(hidden_dim, num_output)))
        self.fake_critic = nn.Sequential(*modules)
    
    def forward(self, input: Tensor) -> List[Tensor]:
        x = self.fake_critic(input)
        return F.sigmoid(x)