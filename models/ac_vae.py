import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class ACVAE(BaseVAE):


    def __init__(self,
                #  vae: nn.Module,
                 latent_dim: int,
                 hidden_dim: int,
                 layer_num: int = 4,
                 num_label: int = 10,
                 num_output: int = 10,
                 **kwargs) -> None:
        super(ACVAE, self).__init__()

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

    def actor_forward(self, input: Tensor, label: Tensor) -> List[Tensor]:
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
    
    def real_critic_forward(self, input: Tensor, label: Tensor) -> Tensor:
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

    def fake_critic_forward(self, input: Tensor) -> List[Tensor]:
        x = self.fake_critic(input)
        return F.sigmoid(x)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def critic_loss_function(self,
                      pred,
                      target) -> dict:
        """
        :return:
        """
        loss = F.binary_cross_entropy(pred, target)
        return {
                    'loss': loss, 
                    'critic_loss': loss
                }
    
    def actor_loss_function(self, var,
                                z, 
                                fake_z, 
                                real_z, 
                                actor_z, 
                                z_critic_out, 
                                z_critic_real, 
                                actor,):
        weight_var = torch.mean(var, 0, True)
        distance_penalty = torch.mean(torch.sum((1 + (actor_z - fake_z).pow(2)).log() * weight_var.pow(-2),1),0)
        distance_penalty += torch.mean(torch.sum((1 + (real_z - z).pow(2)).log() * weight_var.pow(-2),1),0)
        actor_loss = F.binary_cross_entropy(z_critic_out, actor, size_average=False) \
                        + F.binary_cross_entropy(z_critic_real, actor, size_average=False)
        return {
                    'distance_penalty': distance_penalty, 
                    'actor_loss': actor_loss, 
                    'loss': actor_loss + distance_penalty
                }

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]