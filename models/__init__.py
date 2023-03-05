from .base import *
from .vanilla_vae import *
from .cvae import *
from .ac_vae import *
from .actor import *
from .critic import *
from .critic_attr import *
from .classifier import *


# Aliases
VAE = VanillaVAE
GaussianVAE = VanillaVAE
CVAE = ConditionalVAE

vae_models = {
              'VanillaVAE':VanillaVAE,
              'ConditionalVAE':ConditionalVAE,
              'ACVAE': ACVAE,
              'Actor': Actor,
              'Critic': Critic,
              'Critic_Attr': Critic_Attr,
              'Classifier': Classifier,
              }
