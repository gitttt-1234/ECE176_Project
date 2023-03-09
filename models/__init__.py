from .base import *
from .vanilla_vae import *
from .vae_mnist import *
from .ac_vae import *
from .actor import *
from .critic import *
from .critic_attr import *
from .classifier import *


# Aliases
VAE = VanillaVAE
GaussianVAE = VanillaVAE
VanillaVAEMNIST = VanillaVAEMNIST

vae_models = {
              'VanillaVAE':VanillaVAE,
               'VanillaVAEMNIST':VanillaVAEMNIST,
              'ACVAE': ACVAE,
              'Actor': Actor,
              'Critic': Critic,
              'Critic_Attr': Critic_Attr,
              'Classifier': Classifier,
              }